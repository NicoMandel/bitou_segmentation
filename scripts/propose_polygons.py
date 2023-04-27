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
import os

import cv2

def parse_args():
    parser = ArgumentParser(description="Proposing polygons for Semantic Segmentation model")
    # Model settings
    parser.add_argument("-m", "--model", type=str, help="Which model to load. If None given, will look for <best> in <results> folder", default=None)
    parser.add_argument("-i", "--input", help="Input. If directory, will look for .png and .JPG files within this directory. If file, will attempt to load file.", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default=None, help="Output name to be used for the file in the input directory. If not given, will plot image with polygons")    
    parser.add_argument("-p", "--param", type=int, default=20, help="Number of pixels under which to remove polygons. default is 20")
    parser.add_argument("--shape", type=int, default=256, help="model shape to be used during inference. Should be larger (a multiple of) training. Defaults to 512")
    parser.add_argument("--halo", type=int, default=128, help="halo to be used by the model during inference. Defaults to 256, but could be as small as 128")
    parser.add_argument("--batch", type=int, default=6, help="Batch Size for GPU inference. Defaults to 6")
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

def model_pass_reduced(model : Model, img : np.ndarray, augmentations : A.Compose, device : torch.device) -> np.ndarray:
    # for idx in range(img_batch.shape[0]):
    #     img_i = img_batch[idx,...]
    #     x_i = augmentations(image=img_i)['image'].to(device)
    #     if idx == 0:
    #         x_shape = tuple([img_batch.shape[0]] + list(x_i.shape))
    #         x = torch.empty((x_shape), device=device)
    #     x[idx,...] = x_i
    x = augmentations(image=img)['image'].to(device)
    x = x.unsqueeze(dim=0)
    with torch.no_grad():
        y_hat = model(x)
    labels = model.get_labels(y_hat)
    return labels.cpu().numpy().astype(np.uint8)

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

def get_regions(cnts : np.ndarray) -> list:

    rglist = []
    for rg in cnts:
        # rg = rg.squeeze()
        # if len(rg.shape) >= 2:
        rg_x, rg_y = get_region(rg)
        region = {
            "name" : "polygon",
            "all_points_x" : rg_x,
            "all_points_y" : rg_y,
        }
        rgdict = {
            "shape_attributes" : region,
            "region_attributes": {"plant" : "bitou_bush"}
            }
        rglist.append(rgdict)
    return rglist

def get_region(rg : np.ndarray) -> tuple:
    rg_x = rg[:,0].tolist()
    rg_y = rg[:,1].tolist()
    assert len(rg_x) == len(rg_y), "Not same size, cannot be right indexing"
    return rg_x, rg_y

def get_image_dict(img_f, img_fname, cnts : np.ndarray ) -> tuple:
    sz = os.stat(img_f).st_size
    img_id = img_fname + str(sz)
    regs = get_regions(cnts)
    img_dict = {
        "filename" : img_fname,
        "size" : sz,
        "regions" : regs,
        "file_attributes" : {}
    }

    return img_id, img_dict 

def insert_into_dict(json_dict : dict, img_id : str, img_dict : dict) -> dict:
    # add the image to the number of images 
    imglist = json_dict["_via_image_id_list"]
    imglist.append(img_id)
    json_dict["_via_image_id_list"] = imglist
    curr_imgs = json_dict["_via_img_metadata"]
    curr_imgs[img_id] = img_dict
    json_dict["_via_img_metadata"] = curr_imgs      # in-place addition
    # return json_dict

def write_dict(json_dict : dict, out_file : str): 
    """
        writing dictionary into file
    """
    with open(out_file, 'w') as fp:
        json.dump(json_dict, fp)

def get_polygons_from_binary(bin_img: np.ndarray, param : tuple) -> np.ndarray:
    """
        function to get polygons from the binary image.
        Playing around with settings to show polygons
        TODO: consider using RETR_CCOMP as a setting - to get polygons inside polygons: https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    """
    # cleaning - use Gaussian filter

    # use polygon approximation? 

    # base function
    cnts, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_TREE
    
    n_cts = [cnt for cnt in cnts if not is_too_small(cnt, param)]

    return n_cts

def is_too_small(cnt : np.ndarray, param : tuple) -> bool:
    """
        function to check if a polygon is too small by checking the extent along x and y
    """
    rg = cnt.squeeze()
    if len(rg.shape) >= 2:
        d_x = rg[:,0].max() - rg[:,0].min()
        d_y = rg[:,1].max() - rg[:,1].min()
    else:
        return True
    return True if (d_x < param[0] and d_y < param[1]) else False 

# Tiling functions
def get_tile_numbers(img_shape : tuple, model_shape : tuple) -> tuple:
    """
        Function to get the number of tiles that should be used.
        model_width and height are required
    """
    model_h = model_shape[0]
    model_w = model_shape[1]
    
    h = img_shape[0]
    w = img_shape[1]

    leftover_w = w % model_w
    leftover_h = h % model_h

    # n times the image for the width
    n_w = w // model_w if leftover_w == 0 else (w // model_w) +1
    n_h = h // model_h if leftover_h == 0 else (h // model_h) +1

    return n_w * n_h, n_h

def get_padding(img_shape : tuple, model_shape : tuple, halo : int = 256) -> tuple:
    """
        function to get the padding in the format:
        pad_left, pad_right, pad_top, pad_bottom
        the img_shape and model_shape indices hav to coincide
    """
    model_h = model_shape[0]
    model_w = model_shape[1]

    h = img_shape[0]
    w = img_shape[1]

    leftover_w = w % model_w
    leftover_h = h % model_h
    extra_x = model_w - leftover_w
    extra_y = model_h - leftover_h
    
    pad_left = int(halo)
    pad_right = int(extra_x + halo)
    pad_top = int(halo)
    pad_bottom = int(extra_y + halo)

    return pad_left, pad_right, pad_top, pad_bottom

def get_window_dims(model_shape : tuple, halo : int = 256) -> tuple:
    """
        Function to get the dimensions of the window
    """
    return model_shape[0] + 2*halo, model_shape[1] + 2*halo 

def get_stride(model_shape : tuple) -> tuple:
    """
        Function to get the stride of each dimension
    """
    return model_shape

def pad_image(img : np.ndarray, pad_top : int, pad_left : int, pad_right : int, pad_bottom : int) -> np.ndarray:
    padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT) # BORDER_REPLICATE, BORDER_REFLECT_101, BORDER_WRAP
    return padded

def get_out_shape(n_tot : int, n_w : int, model_shape : tuple) -> tuple:
    """
        get the shape of the image output - depends on n_x and n_y
    """
    n_h = int(n_tot / n_w)
    return (n_h * model_shape[0], n_w * model_shape[1])

def get_final_image(out_img : np.ndarray, img_shape : tuple) -> np.ndarray:
    """
        Get the final image from the oversized overhanging image
    """
    return out_img[:img_shape[0], :img_shape[1]]

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
    augmentations = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True)
    ])

    # Settings for tiling the image
    model_shape = (args["shape"], args["shape"])
    halo = args["halo"]
    window_shape = get_window_dims(model_shape, halo)
    stride = get_stride(model_shape)
    batch_size = args["batch"]

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

    px_threshold = (args["param"], args["param"])
    out = args["output"]
    if out is None:
        cdec = get_colour_decoder()
    # go through the image list
    for img_f in img_list:
        fpath = os.path.join(img_dir, (img_f + f_ext))
        img = load_image(fpath)
        assert img is not None, "Not an image File, None object. Ensure {} exists".format(fpath)
        x = img.copy()

        # insert here
        # model pass
        if too_large(img):
            # calculate number of tiles
            img_shape = img.shape[:-1]

            pad_left, pad_right, pad_top, pad_bottom = get_padding(img_shape, model_shape, halo)
            n_tot, n_w  = get_tile_numbers(img_shape, model_shape)
            out_img_shape = get_out_shape(n_tot, n_w, model_shape)
            out_labels = np.zeros(out_img_shape, dtype=np.uint8)        
            padded = pad_image(img, pad_top, pad_left, pad_right, pad_bottom)
            in_batch_size = [batch_size] + list(window_shape) + [3]
            in_batch = np.zeros(tuple(in_batch_size), dtype=np.uint8)
            # ! np indexing: rows, cols := h, w
            for k in range(n_tot):
                # ! i is rows (h), j is columns (w)
                i = k // n_w
                j = k % n_w

                # l is the index in the batch
                # l = k % batch_size
                
                # window locations
                h_window = i * stride[0]
                w_window = j * stride[1]

                # take the window
                window = padded[h_window : h_window + window_shape[0], w_window : w_window + window_shape[1]]
                # insert it into the batch
                # in_batch[l, ...] = window

                ## From here - no batch processing
                label_out = model_pass_reduced(model, window, augmentations, device)
                l_inner = label_out[halo : -halo, halo : -halo]
                out_labels[h_window : h_window + model_shape[0], w_window : w_window + model_shape[1]] = l_inner
                # when the batch is ready 
                # TODO: set a flag when the batch is full or when the end is reached to pass the model
                # TODO: then get back the result. Make the for loop inside a model depending on the batch size
                # ! create the in_batch dynamically. and include the FUCKING k == n_tot -1 in the condition
                # ! turn into a dictionary mode?
                # if (k and (l == 0)):
                    # labels_batch = model_pass_reduced(model, in_batch, augmentations, device)
                    # for m in range(batch_size):
                    #     #! these are not calculated right
                    #     n = k - m
                    #     ii = n // n_w
                    #     jj = n % n_w
                    #     h_insert = ii * stride[0]
                    #     w_insert = jj * stride[1]
                    #     # calculate the indices bit by bit
                    #     # inner ones
                    #     in_st_h = halo
                    #     in_end_h = halo + model_shape[0]
                    #     in_st_w = halo
                    #     in_end_w = halo+model_shape[1]
                    #     inner_labels = labels_batch[m, in_st_h : in_end_h, in_st_w : in_end_w]    # ! careful with indexing here, could be halo+model_shape switched around
                    #     # outer indices
                    #     out_st_h = h_insert
                    #     out_end_h = h_insert + model_shape[0]
                    #     out_st_w = w_insert
                    #     out_end_w = w_insert + model_shape[1]
                    #     out_labels[out_st_h : out_end_h, out_st_w : out_end_w] = inner_labels
                    #     # out_labels = np.zeros(out_img_shape, dtype=np.uint8)        


            # TODO: what if a batch is not full?


            # for j in range(n_y):
            #     y_window = j * stride[0]
            #     y = halo + j * stride[0]
    
            #     for i in range(n_x):
            #         x_window = i * stride[1]
            #         x = halo + i * stride[1]

            #         # get the **window** out and insert it into the batch
            #         window = padded[x_window : x_window+window_shape[1], y_window : y_window+window_shape[0]]
            #         in_batch[ctr%batch_size,...] = window
            #         out_loc[ctr%batch_size] = np.asarray((y,x))
            #         ctr +=1
            #         if (ctr % batch_size == 0):
            #             labels_batch = model_pass_reduced(model, in_batch, augmentations, device)
            #             for k in range(batch_size):
            #                 x = out_loc[k][1]
            #                 y = out_loc[k][0]
            #                 out_labels[x:x+stride, y : y+stride] = labels_batch[k,...]
                        
            #             # TODO: use 1-D indexing to calculate the index
            #             # should be something around (ctr %n_x) * stride[0] = x_loc, (ctr//n_x) * stride[1] = y_loc
            
            labels = get_final_image(out_labels, img_shape)
        else:
            labels = model_pass(model, x, augmentations, device)

        # turn the labels into a binary image
        bin_img = np.copy(labels).astype(np.uint8)
        # bin_img -= 1 # turn to maximum value - alternative: bin_img -= 1 -> should turn 0 into 255 and 1 into 0
        cnts = get_polygons_from_binary(bin_img, param=px_threshold)
        # cnts = get_cnts(cnts)
        
        if out:
            img_id, img_dict = get_image_dict(fpath, img_f + f_ext, cnts)

            insert_into_dict(json_dict, img_id, img_dict)

        
        else:
            # plot
            drawing = img.copy().astype(np.uint8)
            col = (255, 0, 0)
            for i in range(len(cnts)):
                cv2.drawContours(drawing, cnts, -1, col, 3) #, cv2.LINE_8, hierarchy, 0)
            mask = cdec(labels)
            mask = overlay_images(img, mask, alpha=0.5)
            plot_images(mask, drawing, "", "")

    if out:
        write_dict(json_dict, out)

        # Step 1: identify Polygons
        # out_img = get_polygons_from_labels(img, labels)
        # plot_overlaid(mask)

        # Step 2: write the polygons into the json dictionary

    # Step 3: write the json file out again
    # with open(args["output"], "w") as json_f:
    #     json.dump(json_dict)