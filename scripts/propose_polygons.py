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

from csupl.utils import load_image, get_colour_decoder, get_image_list, overlay_images, plot_overlaid, write_image, extract_new_size, pad_image, resize_img
from csupl.generate_masks import get_polygons_from_labels

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


def model_pass(model : Model, img : np.ndarray, augmentations : A.Compose, device : torch.device) -> np.ndarray:
    """
        ! model size 100% breaks the GPU memory on my computer ( > 6 GB) -> rescaling image necessary.
        Binary search by hand resulted in **44 %** being the largest possible size. 
        For images of size (5460, 8192) that results in input image of size (2402, 3604) -> padded to (2432, 3616)
    """
    x = augmentations(image=img)['image'].to(device).unsqueeze(dim=0)
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
        # A.Resize(height, width, p=1),
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

    # go through the image list
    for img_f in img_list:
        fpath = os.path.join(img_dir, (img_f +"."+ f_ext))
        img = load_image(fpath)
        assert img is not None, "Not an image File, None object. Ensure {} exists".format(fpath)
        x = img.copy()
        # model pass
        # ! with resizing the polygon corners cannot be guaranteed to be in the same places 
        # make sure that we return the actual RUN inmage and perform inverse normalization
        labels = model_pass(model, x, augmentations, device)

        # Step 1: identify Polygons
        out_img = get_polygons_from_labels(img, labels)
        plot_overlaid(out_img)

        # Step 2: write the polygons into the json dictionary

    # Step 3: write the json file out again
    # with open(args["output"], "w") as json_f:
    #     json.dump(json_dict)