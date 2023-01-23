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

from csupl.utils import load_image, get_colour_decoder, get_image_list, overlay_images, plot_overlaid, write_image, extract_new_size, pad_image, resize_img

def parse_args():
    parser = ArgumentParser(description="Prediction for Semantic Segmentation model")
    # Model settings
    parser.add_argument("-m", "--model", type=str, help="Which model to load. If None given, will look for <best> in <results> folder", default=None)
    parser.add_argument("-i", "--input", help="Input. If directory, will look for .png and .JPG files within this directory. If file, will attempt to load file.", type=str)
    parser.add_argument("-o", "--output", type=str, help="Output directory. Will use input file name. If none given, will plot result. If already exists, will also plot.", default=None)
    parser.add_argument("--alpha", help="alpha factor for overlay. If not given, will plot mask.", default=None, type=float)
    parser.add_argument("--f_ext", help="File extension to use on folder. Defaults to .JPG", type=str, default=".JPG")
    parser.add_argument("--gpu", action="store_true", help="If set to true, will look for GPU to run inference. Uses CPU otherwise")
    parser.add_argument("--scale", type=int, help="Scale percentage for the image. If none given will attempt to run the full image size", default=None)
    # Model size settings
    # parser.add_argument("--width", help="Width to be used for training", default=512, type=int)
    # parser.add_argument("--height", help="Height to be used during training", default=512, type=int)
    
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
    if args["gpu"]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu') 
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
        print("{} is a directory. Reading all {} files".format(args["input"], args["f_ext"]))
        img_list = get_image_list(args["input"], args["f_ext"])
        print("Found {} {} images. Going through them individually".format(
            len(img_list), args["f_ext"]
        ))
        img_dir = args["input"]
        f_ext = args["f_ext"]
    else:
        print("{} is an image. Reading image".format(args["input"]))
        img_dir = os.path.dirname(args["input"])
        img_list = [str(Path(args["input"]).stem)]
        _, f_ext = os.path.splitext(args["input"])

    # go through the image list
    for img_f in img_list:
        fpath = os.path.join(img_dir, (img_f + f_ext))
        img = load_image(fpath)
        assert img is not None, "Not an image File, None object. Ensure {} exists".format(fpath)
        if args['scale'] is not None:
            img = resize_img(img, args["scale"])
        x = img.copy()
        # model pass
        labels = model_pass(model, x, augmentations, device)
        mask = cdec(labels)
        if args["alpha"] is not None:
            mask = overlay_images(img, mask, alpha=args["alpha"])
        
        if args["output"] is None:
            plot_overlaid(mask, title=img_f)
        else:
            write_image(args["output"], img_f, mask)
            