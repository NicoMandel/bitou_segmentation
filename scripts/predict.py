# The basic semantic segmentation as outlined in the pytorch flash documentation [here](https://lightning-flash.readthedocs.io/en/latest/reference/semantic_segmentation.html)


import torch
import numpy as np
import os.path
from argparse import ArgumentParser

import pytorch_lightning as pl
# from flash.core.data.utils import download_data
from csupl.model import Model
from csupl.dataloader import BitouDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

from csupl.utils import load_image, load_label, get_colour_decoder, InverseNormalization, get_image_list, overlay_images, plot_overlaid, write_image

def parse_args():
    parser = ArgumentParser(description="Training Loop for Semantic Segmentation model")
    # Model settings
    parser.add_argument("-m", "--model", type=str, help="Which model to load. If None given, will look for <best> in <results> folder", default=None)
    parser.add_argument("-i", "--input", help="Input. If directory, will look for .png and .JPG files within this directory. If file, will attempt to load file.", type=str)
    parser.add_argument("-o", "--output", type=str, help="Output directory. Will use input file name. If none given, will plot result. If already exists, will also plot.", default=None)
    parser.add_argument("--alpha", help="alpha factor for overlay. If not given, will plot input and output side by side.", default=None)

    # Model size settings
    # parser.add_argument("--width", help="Width to be used for training", default=512, type=int)
    # parser.add_argument("--height", help="Height to be used during training", default=512, type=int)
    
    args = parser.parse_args()
    return vars(args)

def model_pass(model : Model, img : np.ndarray, augmentations : A.Compose, device : torch.device) -> np.ndarray:
    x = augmentations(img)
    x.to(device)
    with torch.no_grad():
        y_hat = model(x)
    # binary case:
    if model.classes == 1:
        y_hat.sigmoid().detach().cpu().numpy()
    else:       # multiclass case
        y_hat = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
    return y_hat


if __name__=="__main__":
    #Setup
    args = parse_args()
    fdir = os.path.abspath(os.path.dirname(__file__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cdec = get_colour_decoder()

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
        # A.Resize(height, width, p=1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True)
    ])

    # check output format.
    if args["output"] is not None and not os.path.isdir(args["output"]):
        raise OSError("Output is not a directory. Breaking")
    
    # check input format
    if os.path.isdir(args["input"]):
        # get file list
        print("{} is a directory. Reading all .png and .JPG files".format(args["input"]))
        img_list_jpg = get_image_list(args["input"], ".JPG")
        img_list_png = get_image_list(args["input"], f_ext=".png")
        print("Found {} .JPG and {} .png images. Going through them individually".format(
            len(img_list_jpg), len(img_list_png)
        ))
        img_dir = args["input"]
        img_list = img_list_png + img_list_jpg
    else:
        print("{} is an image. Reading image".format(args["input"]))
        img_dir = os.path.dirname(args["input"])
        img_list = [os.path.basename(args["input"])]

    # go through the image list
    for img_f in img_list:
        fpath = os.path.join(img_dir, img_f)
        img = load_image(fpath)
        x = img.copy()
        # model pass
        y_hat = model_pass(model, x, augmentations, device)
        mask = cdec(y_hat)
        if args["alpha"] is not None:
            mask = overlay_images(img, mask, alpha=args["alpha"])
        
        if args["output"] is None:
            plot_overlaid(mask, title=img_f)
        else:
            write_image(args["output"], img_f, mask)
            