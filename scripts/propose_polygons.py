# The basic semantic segmentation as outlined in the pytorch flash documentation [here](https://lightning-flash.readthedocs.io/en/latest/reference/semantic_segmentation.html)


import torch
import numpy as np
import os.path
from argparse import ArgumentParser
from pathlib import Path

# from flash.core.data.utils import download_data
from csupl.model import Model
import albumentations as A
from albumentations.pytorch import ToTensorV2

# json for parsing the config file
import json

from csupl.utils import load_image, get_colour_decoder, get_image_list, overlay_images, plot_overlaid, write_image, extract_new_size, pad_image, resize_img, plot_images
from csupl.generate_masks import get_polygons_from_labels

import cv2

def parse_args():
    parser = ArgumentParser(description="Proposing polygons for Semantic Segmentation model")
    # Model settings
    parser.add_argument("-m", "--model", type=str, help="Which model to load. If None given, will look for <best> in <results> folder", default=None)
    parser.add_argument("-i", "--input", help="Input. If directory, will look for .png and .JPG files within this directory. If file, will attempt to load file.", type=str, required=True)
    # parser.add_argument("-o", "--output", type=str, help="Output name to be used for the file in the input directory.", required=True)    
    args = parser.parse_args()
    return vars(args)

def rescale_image(img : torch.Tensor, msg: str) -> torch.Tensor:
    """
        function to pad the image if necessary by the architecture
    """
    nshape = extract_new_size(msg)
    nimg = pad_image(img, nshape)
    return nimg


def get_cnts(cnts):
    """
        utility function based on the fucking pyimagesearch thing because OpenCV has changed their shitty interface. Mofos
        https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py#L154
    """
    if len(cnts) == 2:
        cnts = cnts[0]
    elif len(cnts) == 3:
        cnts = cnts[1]
    else: raise ValueError("Some shit happened to OpenCV. Good luck")
    return cnts
    

def model_pass(model : Model, img : np.ndarray, augmentations : A.Compose, device : torch.device) -> np.ndarray:
    """
        ! model size 100% breaks the GPU memory on my computer ( > 6 GB) -> rescaling image necessary.
        TODO: could break into a batch of 4 (25%) -> run as one batch, then restitch
        Binary search by hand resulted in **44 %** being the largest possible size. 
        For images of size (5460, 8192) that results in input image of size (2402, 3604) -> padded to (2432, 3616)
    """

    if len(img.shape) == 3:
        x = augmentations(image=img)['image'].to(device)
        x = x.unsqueeze(dim=0)
    else:
        # x = torch.Tensor()
        for i in range(img.shape[0]):
            img_i = img[i,...]
            x_i = augmentations(image=img_i)['image'].to(device)
            if i == 0:
                x_shape = tuple([img.shape[0]] + list(x_i.shape))
                x = torch.empty((x_shape), device=device)
            x[i,...] = x_i
    # x.to(device)
    with torch.no_grad():
        # y_hat = model(x)
        try:
            y_hat = model(x)
        except RuntimeError as e:
            nx = rescale_image(x, e)
            # nx.to(device)
            y_hat = model(nx)
    labels = model.get_labels(y_hat)

    return labels.cpu().numpy().astype(np.int8)

def too_large(img : np.ndarray) -> bool:
    return True if (img.shape[0] > 1024 or img.shape[1] > 1024)  else False

def to_quadrants(img : np.ndarray) -> np.ndarray:
    """
        Function to turn an image into a batch from images
        top left is 0, top right is 1, bottom left is 2, bottom right is 3
    """
    half_v = img.shape[0] // 2
    half_h = img.shape[1] // 2
    l_t = img[: half_v, :half_h]
    l_b = img[half_v :, :half_h]
    r_t = img[: half_v, half_h :]
    r_b = img[half_v:, half_h : ]
    nimg = np.zeros((4, half_v, half_h, img.shape[2])).astype(np.uint8)
    nimg[0,...] = l_t
    nimg[1, ...] = r_t
    nimg[2, ...] = l_b
    nimg[3, ...] = r_b
    return nimg

def from_quadrants(img : np.ndarray) -> np.ndarray:
    """
        Function to return a batch of 4 images into a single image again.
        top left is 0, top right is 1, bottom left is 2, bottom right is 3
    """
    img_h = img.shape[1]
    img_v = img.shape[2]
    nimg_shape = [img_h *2, img_v * 2]
    nimg_sh = nimg_shape if len(img.shape) == 3 else nimg_shape + [img.shape[3]] 
    nimg = np.zeros(nimg_sh).astype(np.uint8)
    nimg[: img_h, : img_v] = img[0,...]
    nimg[: img_h, img_v :] = img[1, ...]
    nimg[img_h:, : img_v] = img[2,...]
    nimg[img_h:, img_v :] = img[3, ...]
    return nimg

if __name__=="__main__":
    #Setup
    args = parse_args()
    fdir = os.path.abspath(os.path.dirname(__file__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    # Argument unpacking
    if args["model"] is None:
        resdir = os.path.join(fdir, '..', 'results')
        model_f = os.path.join(resdir, 'best')
    else:
        model_f = args["model"]
    model = Model.load_from_checkpoint(model_f)
    print("Using model: {} from file: {}".format(model, os.path.realpath(model_f)))

    # model setup
    model.eval()
    model.to(device)
    preprocess_params = model.get_preprocessing_parameters()
    mean = tuple(preprocess_params['mean'])
    std = tuple(preprocess_params['std'])
    # height = 512
    # width = 512
    augmentations = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True)
    ])

    # config json file as base file for the format
    confdir = os.path.join(fdir, '..', 'config')
    base_json_f = os.path.join(confdir, 'via_base.json')
    with open(base_json_f, 'r') as json_f:
        json_dict = json.load(json_f)
    
    # check input format
    if os.path.isdir(args["input"]):
        # get file list
        print("{} is a directory. Reading all files".format(args["input"]))
        img_list, f_ext = get_image_list(args["input"])
        print("Found {} {} images. Going through them individually".format(
            len(img_list), f_ext
        ))
        img_dir = args["input"]
       
    else:
        print("{} is an image. Reading image".format(args["input"]))
        img_dir = os.path.dirname(args["input"])
        img_list = [str(Path(args["input"]).stem)]
        _, f_ext = os.path.splitext(args["input"])
    cdec = get_colour_decoder()

    # go through the image list
    for img_f in img_list:
        fpath = os.path.join(img_dir, (img_f + f_ext))
        img = load_image(fpath)
        assert img is not None, "Not an image File, None object. Ensure {} exists".format(fpath)
        x = img.copy()
        # model pass
        if too_large(img):
            x = to_quadrants(x)
            labels = model_pass(model, x, augmentations, device)
            labels = from_quadrants(labels)
        else:
            labels = model_pass(model, x, augmentations, device)

        # turn the labels into a binary image
        bin_img = np.copy(labels).astype(np.uint8)
        # bin_img -= 1 # turn to maximum value - alternative: bin_img -= 1 -> should turn 0 into 255 and 1 into 0

        cnts, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = get_cnts(cnts)

        drawing = img.copy().astype(np.uint8)
        col = (255, 0, 0)
        for i in range(len(cnts)):
            cv2.drawContours(drawing, cnts, -1, col, 3) #, cv2.LINE_8, hierarchy, 0)
        plot_images(img, drawing, "", "")
        # cv2.imshow("Contours", drawing)
        # 
        mask = cdec(labels)
        mask = overlay_images(img, mask, alpha=0.5)

        # Step 1: identify Polygons
        # out_img = get_polygons_from_labels(img, labels)
        # plot_overlaid(mask)

        # Step 2: write the polygons into the json dictionary

    # Step 3: write the json file out again
    # with open(args["output"], "w") as json_f:
    #     json.dump(json_dict)