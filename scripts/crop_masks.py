
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
    parser.add_argument("-b", "--balance", help="Ratio to balance the images by. Background Ratio. Defaults to None", default=None, type=float)
    parser.add_argument("--epsilon", help="Default range to accept balance ratio by. defaults to 0.02", default=0.02, type=float)
    args = vars(parser.parse_args())
    return args

def get_polygon_coordinates(json_dict : dict) -> dict:
    """
        Gets the xy coordinates of the specific image defined by img_fname from the json dict and returns them in the format required by 
        PIL ImageDraw.Polygon : [(x, y), (x,y), ..., ...]
    """
    poly_dict = {}
    for _, v in json_dict.items():
        if len(v['regions']) > 1:
            print(v['filename'])
        if v['regions']:
            xy_list = []
            for reg in v['regions']:
                x = reg['shape_attributes']['all_points_x']
                y = reg['shape_attributes']['all_points_y']
                xy = list(zip(x, y))
                xy_list.append(xy)
        else:
            poly_dict[v['filename'].split('.')[0]] = []
        
        # if len(xy_list) > 1:
        #     print(xy_list)
        poly_dict[v['filename'].split('.')[0]] = xy_list
    
    return poly_dict

def generate_mask_image(mask_img : Image.Image, polygon_coord : list, whiteout : bool = False) -> Image.Image:
    """
        Function to generate a single polygon for a single image and return the image
    """
    # Image modes from Pillow: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    d = ImageDraw.Draw(mask_img)
    d.polygon(polygon_coord, fill=(255, 255, 255) if whiteout else (1,0,0))
    return mask_img


def crop_image(poly_coord : list, orig_img : Image.Image, mask_im : Image.Image,  epsilon : float, balance_ratio : float = None) -> tuple:
    """
        Function to crop the image around the polygon
    """
    xy_arr = np.asarray(poly_coord)
    x_max, y_max = xy_arr.max(axis=0)
    x_min, y_min = xy_arr.min(axis=0)
    coords = (x_min, y_min, x_max, y_max)
    # balance the coordinates
    if balance_ratio is not None:
        coords = balance_image(mask_im, balance_ratio, coords, epsilon)
    crop_img = orig_img.crop(coords)
    crop_mask = mask_im.crop(coords)
    return crop_img, crop_mask

def _is_between(lower, upper, testval) -> bool:
    return min(lower, upper) < testval < max(lower, upper)

def _dilate_by(coords : tuple, boundaries : tuple, px_ct : int = 1) -> tuple:
    x_min, y_min, x_max, y_max = coords
    x_min -= px_ct if x_min > 0 else 0
    y_min -= px_ct if y_min > 0 else 0
    x_max += px_ct if x_max < (boundaries[1] - 1) else (boundaries[1] - 1)
    y_max += px_ct if y_max < (boundaries[0] - 1) else (boundaries[0] - 1)
    return (x_min, y_min, x_max, y_max)

def _get_ratio(crop : np.array, backgrd_idx : int = 0) -> float:
    m_ct = np.count_nonzero(crop > backgrd_idx)
    background_ct = np.count_nonzero(crop == backgrd_idx)
    ratio = background_ct / (m_ct + background_ct)
    return ratio

def balance_image(mask_img : Image.Image, balance_ratio : float, coords : tuple, epsilon : float = 0.05) -> tuple:
    """
        Function to balance out the image pixels.
        balance_ratio is the targeted amount of **Background** pixels
        Note: only for binary case currently
    """
    m_arr = np.asarray(mask_img)[... ,0]      # only concered about red channel
    boundaries = m_arr.shape
    x_min, y_min, x_max, y_max = coords
    init_crop = m_arr[y_min:y_max+1, x_min:x_max+1]
    ratio = _get_ratio(init_crop, 0)
    while not _is_between(balance_ratio-epsilon, balance_ratio+epsilon, ratio):   
        # crop image around the coordinates
        x_min, y_min, x_max, y_max = coords
        crop_arr = m_arr[y_min:y_max+1,x_min:x_max+1]
        
        # get the pixel ratio
        ratio = _get_ratio(crop_arr)
        # tqdm.write(f"{ratio}")
        # expand the pixel ratio
        coords = _dilate_by(coords, boundaries)
    return coords


def write_masks(polygon_coord: dict, input_dir : Path, mask_dir : Path, crop_dir : Path, f_ext : string, whiteout : bool, epsilon : float, balance_ratio : float = None):
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

            new_img, mask_im = crop_image(poly_coord, orig_img, mask_im, epsilon, balance_ratio)
            if len(poly_coord_list) > 1:
                # new filename here
                fname = k + str(i)
            else:
                fname = k
            mask_path = mask_dir / ".".join([fname, f_ext])
            mask_im.save(mask_path, "png")
            new_img_path = crop_dir / ".".join([fname, f_ext])
            new_img.save(new_img_path, "png")

if __name__=="__main__":
    args  = parse_args()

    img_directory = Path(args["input"])
    mask_directory = Path(args["output"])
    f_ext = args["file_extension"]
    whiteout = args["whiteout"]
    balance_ratio = args["balance"]
    epsilon = args["epsilon"]

    crop_dir = args["crop"]
    if crop_dir is None:
        raise ValueError("No crop directory specified")
    else:
        crop_dir = Path(crop_dir)

    img_list = list([x.stem for x in img_directory.glob("*."+f_ext)])
    if args["config"] is None:
        config_file = next(img_directory.glob("*.json"))
    else:
        config_file = args["config"]

    json_f = open(config_file, "r")
    json_dict = json.load(json_f)

    json_metadata = json_dict["_via_img_metadata"]
    polygon_dict = get_polygon_coordinates(json_metadata)
    write_masks(polygon_dict, img_directory, mask_directory, crop_dir, f_ext, whiteout, epsilon, balance_ratio)
