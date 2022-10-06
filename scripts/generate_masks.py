
"""
Script to generate binary masks from json labels as exported by VGG Image Annotator

"""

import json
import os.path
import argparse
from pathlib import Path
import string
from PIL import Image, ImageDraw


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
    parser.add_argument("-o", "--output", help="Directory where masks should be output to", type=str, default=def_output)
    parser.add_argument("-c", "--config", help="Location of the config file. If not specified, will look for a .json in the input directory", default=None)
    parser.add_argument("--file-extension", help="Image file extension, without dot. Defaults to JPG", default="JPG")
    args = vars(parser.parse_args())
    return args

def get_polygon_coordinates(json_dict : dict) -> dict:
    """
        Gets the xy coordinates of the specific image defined by img_fname from the json dict and returns them in the format required by 
        PIL ImageDraw.Polygon : [(x, y), (x,y), ..., ...]
    """
    poly_dict = {}
    for _, v in json_dict.items():
        if v['regions']:
            x = v['regions'][0]['shape_attributes']['all_points_x']
            y = v['regions'][0]['shape_attributes']['all_points_y']
            xy = list(zip(x, y))
        else:
            poly_dict[v['filename'].split('.')[0]] = []
        poly_dict[v['filename'].split('.')[0]] = xy
    
    return poly_dict

def generate_mask_image(polygon_coord : list, orig_img : Image.Image) -> Image.Image:
    """
        Function to generate a polygon for a single image and return the image
    """
    imsize = orig_img.size
    mask_img = Image.new("RGB", imsize)              # Image modes from Pillow: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    d = ImageDraw.Draw(mask_img)
    d.polygon(polygon_coord, fill=(1,0,0))
    return mask_img


def write_masks(polygon_coord: dict, img_list : list, input_dir : Path, mask_dir : Path, f_ext : string):
    """
        function to load the images and write the mask
    """

    for k, poly_coord in polygon_coord.items():
        orig_path = input_dir / ".".join([k,f_ext])
        orig_img = Image.open(orig_path)
        mask_im = generate_mask_image(poly_coord, orig_img)
        mask_path = mask_dir / ".".join([k, f_ext])
        mask_im.save(mask_path, "JPEG")

if __name__=="__main__":
    args  = parse_args()

    img_directory = Path(args["input"])
    mask_directory = Path(args["output"])
    f_ext = args["file_extension"]

    img_list = list([x.stem for x in img_directory.glob("*."+f_ext)])
    if args["config"] is None:
        config_file = next(img_directory.glob("*.json"))
    else:
        config_file = args["config"]

    json_f = open(config_file, "r")
    json_dict = json.load(json_f)

    json_metadata = json_dict["_via_img_metadata"]
    polygon_dict = get_polygon_coordinates(json_metadata)
    write_masks(polygon_dict, img_list, img_directory, mask_directory, f_ext)
    print("Test Done")

    """
    TODO: continue here:
        1. overlay for checks
        2. check against existence of files - with imglist
        3. check against 
     !  4. via has a filelist inside ->  "_via_image_id_list": 
        5. do the file writing as tqdm -> for progress bar
    """
