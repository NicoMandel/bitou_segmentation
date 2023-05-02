""" The basic semantic segmentation as outlined in the pytorch flash documentation [here](https://lightning-flash.readthedocs.io/en/latest/reference/semantic_segmentation.html)
    TODO: put into batching
    for j in range(n_y):
        in_batch[ctr%batch_size,...] = window
        out_loc[ctr%batch_size] = np.asarray((y,x))
        ctr +=1
        if (ctr % batch_size == 0 and j):
            labels_batch = model_pass_reduced(model, in_batch, augmentations, device)
            for k in range(batch_size):
                x = out_loc[k][1]
                y = out_loc[k][0]
                out_labels[x:x+stride, y : y+stride] = labels_batch[k,...]
    # TODO: set a flag when the batch is full or when the end is reached to pass the model
                # TODO: then get back the result. Make the for loop inside a model depending on the batch size
                # ! create the in_batch dynamically. and include the FUCKING k == n_tot -1 in the condition
                # ! turn into a dictionary mode?
            # for m in range(batch_size):
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
"""

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

from csupl.utils import load_image, get_colour_decoder, get_image_list, overlay_images, plot_images
from csupl.propose_utils import get_tile_numbers, get_window_dims, get_stride, too_large, get_padding, get_out_shape, get_final_image, get_polygons_from_binary, get_image_dict, insert_into_dict, write_dict, pad_image
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

# def rescale_image(img : torch.Tensor, msg: str) -> torch.Tensor:
#     """
#         function to pad the image if necessary by the architecture
#     """
#     nshape = extract_new_size(msg)
#     nimg = pad_image(img, nshape)
#     return nimg

# def model_pass(model : Model, img : np.ndarray, augmentations : A.Compose, device : torch.device) -> np.ndarray:
#     """
#         ! model size 100% breaks the GPU memory on my computer ( > 6 GB) -> rescaling image necessary.
#         TODO: could break into a batch of 4 (25%) -> run as one batch, then restitch
#         Binary search by hand resulted in **44 %** being the largest possible size. 
#         For images of size (5460, 8192) that results in input image of size (2402, 3604) -> padded to (2432, 3616)
#     """

#     if len(img.shape) == 3:
#         x = augmentations(image=img)['image'].to(device)
#         x = x.unsqueeze(dim=0)
#     else:
#         # x = torch.Tensor()
#         for i in range(img.shape[0]):
#             img_i = img[i,...]
#             x_i = augmentations(image=img_i)['image'].to(device)
#             if i == 0:
#                 x_shape = tuple([img.shape[0]] + list(x_i.shape))
#                 x = torch.empty((x_shape), device=device)
#             x[i,...] = x_i
#     # x.to(device)
#     with torch.no_grad():
#         # y_hat = model(x)
#         try:
#             y_hat = model(x)
#         except RuntimeError as e:
#             nx = rescale_image(x, e)
#             # nx.to(device)
#             y_hat = model(nx)
#     labels = model.get_labels(y_hat)

#     return labels.cpu().numpy().astype(np.int8)

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

# def to_quadrants(img : np.ndarray) -> np.ndarray:
#     """
#         Function to turn an image into a batch from images
#         top left is 0, top right is 1, bottom left is 2, bottom right is 3
#     """
#     half_v = img.shape[0] // 2
#     half_h = img.shape[1] // 2
#     l_t = img[: half_v, :half_h]
#     l_b = img[half_v :, :half_h]
#     r_t = img[: half_v, half_h :]
#     r_b = img[half_v:, half_h : ]
#     nimg = np.zeros((4, half_v, half_h, img.shape[2])).astype(np.uint8)
#     nimg[0,...] = l_t
#     nimg[1, ...] = r_t
#     nimg[2, ...] = l_b
#     nimg[3, ...] = r_b
#     return nimg

# def from_quadrants(img : np.ndarray) -> np.ndarray:
#     """
#         Function to return a batch of 4 images into a single image again.
#         top left is 0, top right is 1, bottom left is 2, bottom right is 3
#     """
#     img_h = img.shape[1]
#     img_v = img.shape[2]
#     nimg_shape = [img_h *2, img_v * 2]
#     nimg_sh = nimg_shape if len(img.shape) == 3 else nimg_shape + [img.shape[3]] 
#     nimg = np.zeros(nimg_sh).astype(np.uint8)
#     nimg[: img_h, : img_v] = img[0,...]
#     nimg[: img_h, img_v :] = img[1, ...]
#     nimg[img_h:, : img_v] = img[2,...]
#     nimg[img_h:, img_v :] = img[3, ...]
#     return nimg


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
            n_tot, n_h  = get_tile_numbers(img_shape, model_shape)
            out_img_shape = get_out_shape(n_tot, n_h, model_shape)
            out_labels = np.zeros(out_img_shape, dtype=np.uint8)        
            padded = pad_image(img, pad_top, pad_left, pad_right, pad_bottom)
            # in_batch_size = [batch_size] + list(window_shape) + [3]
            # in_batch = np.zeros(tuple(in_batch_size), dtype=np.uint8)
            for k in range(n_tot):
                j = k // n_h
                i = k % n_h

                # l is the index in the batch
                # l = k % batch_size
                
                # window locations
                h_window = i * stride[0]
                w_window = j * stride[1]

                # take the window
                window = padded[h_window : h_window + window_shape[0], w_window : w_window + window_shape[1]]
                #! window is the wrong shape now - why?
                # insert it into the batch
                # in_batch[l, ...] = window

                label_out = model_pass_reduced(model, window, augmentations, device)
                l_inner = label_out[halo : -halo, halo : -halo]
                out_labels[h_window : h_window + model_shape[0], w_window : w_window + model_shape[1]] = l_inner
            
            labels = get_final_image(out_labels, img_shape)
        else:
            labels = model_pass_reduced(model, x, augmentations, device)

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
            for _ in range(len(cnts)):
                cv2.drawContours(drawing, cnts, -1, col, 3) #, cv2.LINE_8, hierarchy, 0)
            mask = cdec(labels)
            mask = overlay_images(img, mask, alpha=0.5)
            plot_images(mask, drawing, "", "")

    if out:
        write_dict(json_dict, out)