
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
from csupl.generate_masks import *


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
    parser.add_argument("--whiteout", action="store_true", help="If set, will whiteout the mask")
    args = vars(parser.parse_args())
    return args


if __name__=="__main__":
    args  = parse_args()

    img_directory = Path(args["input"])
    mask_directory = Path(args["output"])
    f_ext = args["file_extension"]
    whiteout = args["whiteout"]

    img_list = list([x.stem for x in img_directory.glob("*."+f_ext)])
    if args["config"] is None:
        config_file = next(img_directory.glob("*.json"))
    else:
        config_file = args["config"]

    json_f = open(config_file, "r")
    json_dict = json.load(json_f)

    json_metadata = json_dict["_via_img_metadata"]
    polygon_dict = get_polygon_coordinates(json_metadata)
    write_masks(polygon_dict, img_directory, mask_directory, f_ext, whiteout)
