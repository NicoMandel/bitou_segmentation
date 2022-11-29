
"""
Script to generate binary masks from json labels as exported by VGG Image Annotator
    TODO: continue here:
        1. overlay for checks
        2. check against existence of files - with imglist
        3. check against 
     !  4. via has a filelist inside ->  "_via_image_id_list": 
        5. do the file writing as tqdm -> for progress bar
"""

import json
import os.path
import argparse
from pathlib import Path
import string
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
from csupl.generate_masks import get_polygon_coordinates, generate_labels, get_config, crop_from_polygon
from csupl.utils import get_image_list

def parse_args():
    """
        Argument parser
    """
    fdir = os.path.abspath(os.path.dirname(__file__))
    def_input = os.path.join(fdir, "..", "data", "bitou_test")
    def_output = os.path.join(os.path.dirname(def_input), "bitou_test_masks") 
    # def_output = os.path.join(fdir, "..", "data", "bitou_test")
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Directory to generate masks from", type=str, default=def_input)
    parser.add_argument("-o", "--output", help="Directory where **Masks** should be output to", type=str, default=def_output)
    parser.add_argument("-c", "--config", help="Location of the config file. If not specified, will look for a .json in the input directory", default=None)
    parser.add_argument("--file-extension", help="Image file extension, without dot. Defaults to JPG", default="JPG")
    parser.add_argument("--crop", help="Directory where cropped **Images** should be output to", default=None)
    parser.add_argument("--whiteout", action="store_true", help="If set, will whiteout all the masks in the image")
    args = vars(parser.parse_args())
    return args

def crop_images(polygon_coord: dict, input_dir : Path, mask_dir : Path, crop_dir : Path, f_ext : string, whiteout : bool):
    """
        function to load the images and write the mask
    """

    for k, poly_coord_list in tqdm(polygon_coord.items()):
        orig_path = input_dir / ".".join([k,f_ext])
        orig_img = Image.open(orig_path)
        imsize = orig_img.size
        for i, poly_coord in enumerate(poly_coord_list):
            mask_im = Image.new("RGB", imsize)
            mask_im = generate_mask_image(mask_im, poly_coord, whiteout)

            new_img, mask_im = crop_image(poly_coord, orig_img, mask_im)
            if len(poly_coord_list) > 1:
                # new filename here
                fname = k + str(i)
            else:
                fname = k
            mask_path = mask_dir / ".".join([fname, f_ext])
            mask_im.save(mask_path, "png")
            new_img_path = crop_dir / ".".join([fname, f_ext])
            new_img.save(new_img_path, "png")
            # TODO: mask balance?

if __name__=="__main__":
    args  = parse_args()

    img_directory = Path(args["input"])
    mask_directory = Path(args["output"])
    f_ext = args["file_extension"]
    whiteout = args["whiteout"]

    crop_dir = args["crop"]
    if crop_dir is None:
        raise ValueError("No crop directory specified")
    else:
        crop_dir = Path(crop_dir)

    img_list = get_image_list(img_directory)
    config_file = args["config"]

    polygon_dict = get_config(config_file, img_directory)
    crop_images(polygon_dict, img_directory, mask_directory, crop_dir, f_ext, whiteout)
