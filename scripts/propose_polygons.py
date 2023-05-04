
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

from tqdm import tqdm

def parse_args():
    parser = ArgumentParser(description="Proposing polygons for Semantic Segmentation model")
    # Model settings
    parser.add_argument("-m", "--model", type=str, help="Which model to load. If None given, will look for <best> in <results> folder", default=None)
    parser.add_argument("-i", "--input", help="Input. If directory, will look for .png and .JPG files within this directory. If file, will attempt to load file.", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default=None, help="Output name to be used for the file in the input directory. If not given, will plot image with polygons")    
    parser.add_argument("-p", "--param", type=int, default=20, help="Number of pixels under which to remove polygons. default is 20")
    parser.add_argument("--shape", type=int, default=256, help="model shape to be used during inference. Should be larger (a multiple of) training. Defaults to 512")
    parser.add_argument("--halo", type=int, default=128, help="halo to be used by the model during inference. Defaults to 256, but could be as small as 128")
    parser.add_argument("--batch", type=int, default=12, help="Batch Size for GPU inference. Defaults to 12")
    args = parser.parse_args()
    return vars(args)

def model_pass(model : Model, batch: torch.Tensor, device : torch.device) -> np.ndarray:
    batch = batch.to(device)
    with torch.no_grad():
        y_hat = model(batch)
    labels = model.get_labels(y_hat)
    return labels.cpu().numpy().astype(np.uint8)

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
    for img_f in tqdm(img_list, desc="Image"):
        fpath = os.path.join(img_dir, ".".join([img_f, f_ext]))
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

            # Torch format: b c h w
            in_batch = None 
            # torch.zeros(tuple([batch_size, 3, window_shape[0], window_shape[1]]), dtype=torch.uint8)
            ctr = 0
            for k in tqdm(range(n_tot), leave=False, desc="Window"):
                j = k // n_h
                i = k % n_h

                # l is the index in the batch
                # l = k % batch_size
                
                # window locations
                h_window = i * stride[0]
                w_window = j * stride[1]

                # take the window and turn it into the batch format
                window = padded[h_window : h_window + window_shape[0], w_window : w_window + window_shape[1]]
                x = augmentations(image=window)['image']

                # create the in_batch
                if in_batch is None:
                    in_batch = [x]
                else:
                    in_batch.append(x)
                ctr += 1
                # x = x.unsqueeze(dim=0)

                # if the batch is full - or if it is the last element
                if ((ctr % batch_size) == 0 or (ctr == n_tot)):
                    batch_len = len(in_batch)
                    in_batch = torch.stack(tuple(in_batch), dim=0)
                    batch_out = model_pass(model, in_batch.float(), device)
                    for m in range(batch_len-1, -1, -1):
                        # which index in the batch - inverse order
                        b_id = batch_len -m -1                        
                        # how many images before are we using now to calculate?
                        n = k - m #-1
                        jj = n // n_h
                        ii = n % n_h
                        # where do we want to insert it in the new image?
                        h_insert = ii * stride[0]
                        w_insert = jj * stride[1]
                        
                        # get the image out
                        out_im = batch_out[b_id, halo : -halo, halo : -halo]

                        # insert it into the right place in the output image
                        out_labels[h_insert : h_insert + model_shape[0], w_insert : w_insert + model_shape[1]] = out_im
                        in_batch = None

                # label_out = model_pass(model, x, device)
                # l_inner = label_out[halo : -halo, halo : -halo]
                # out_labels[h_window : h_window + model_shape[0], w_window : w_window + model_shape[1]] = l_inner
            
            labels = get_final_image(out_labels, img_shape)
        else:
            labels = model_pass(model, x, augmentations, device)

        # turn the labels into a binary image
        bin_img = np.copy(labels).astype(np.uint8)
        # bin_img -= 1 # turn to maximum value - alternative: bin_img -= 1 -> should turn 0 into 255 and 1 into 0
        cnts = get_polygons_from_binary(bin_img, param=px_threshold)
        # cnts = get_cnts(cnts)
        
        if out:
            img_id, img_dict = get_image_dict(fpath, ".".join([img_f, f_ext]), cnts)
            insert_into_dict(json_dict, img_id, img_dict)
        
        else:
            # plot
            drawing = img.copy().astype(np.uint8)
            col = (255, 0, 0)
            for _ in tqdm(range(len(cnts)), desc="Drawing", leave=False):
                cv2.drawContours(drawing, cnts, -1, col, 3) #, cv2.LINE_8, hierarchy, 0)
            mask = cdec(labels)
            mask = overlay_images(img, mask, alpha=0.5)
            plot_images(mask, drawing, "", "")

    if out:
        write_dict(json_dict, out)