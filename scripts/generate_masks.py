"""
    Script to generate masks using the watershed algorithm for sand labelling
"""

from os import path, mkdir
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import albumentations as A

from csupl.watershed import Watershed
from csupl.utils import ColourDecoder, get_image_list, plot_overlaid, overlay_images, load_image, get_colour_decoder, write_image
from csupl.generate_masks import merge_classes, generate_labels, get_config, crop_pair_from_polygon, crop_from_polygon, split_polyon_dict

def parse_args():
    """
        Argument parser
    """
    fdir = path.abspath(path.dirname(__file__))
    
    parser = ArgumentParser(description="File for generating masks")
    parser.add_argument("-i", "--input", help="Directory to generate masks from", type=str, required=True)
    parser.add_argument("-o", "--output", help="Directory where label files should be output to. If none, will cycle through images and plot side by side", type=str, default=None)
    parser.add_argument("--config", help="Location of the config file. If not specified, will look for a .json in the input directory", default=None)
    parser.add_argument("--colour", type=str, help="Which colour code to use. If none, will look for file colour_code.json in default config dir", default=None)
    parser.add_argument("--file-extension", help="Image file extension to read in, with dot. If None, will search through the directory for most pictures", default=None)
    # parser.add_argument("--tolerance", type=float, help="Tolerance to be used for the kmeans algorithm nearest neighbor-distance!", default=0.5)
    # parser.add_argument("-w", "--watershed", action="store_true", help="If set to true, will use watershed alorithm to pre-label sand class")
    parser.add_argument("--overlay", type=float, help="Whether the mask written out will be an overlay or only the colour code. Alpha value", default=None)
    parser.add_argument("-c", "--class", type=int, help="Class index for the polygon to be assigned. Defaults to 1 for binary classification", default=1)
    args = vars(parser.parse_args())
    return args

def write_img_to_dir(dir, fname, img):
    try:
        write_image(dir, fname, img)
    except OSError: raise

def setup_dir(basedir, name):
    try:
        ndir = path.join(basedir, name)
        mkdir(ndir)
        print(f"Created new directory: {ndir}")
        return ndir
    except OSError: raise

if __name__=="__main__":
    args = parse_args()
    
    # Files
    img_directory = Path(args["input"])
    label_dir = args["output"]
    in_fext = args["file_extension"]
    conf_f = args["config"]

    # Getting files and algorithm set up
    img_list, in_fext = get_image_list(img_directory, in_fext)
    poly_dict = get_config(conf_f, img_directory)
    overlay = args["overlay"]
    class_idx = args["class"]

    colour_path = args["colour"]
    colour_decoder = get_colour_decoder(colour_path)

    # Logging
    print("Reading from directory: {}\t{} files".format(
        img_directory, len(img_list)
    ))

    # Setting up output
    print("Output: {}".format(
        "plotting" if label_dir is None else label_dir
    ))

    if label_dir is not None:
        ldir = setup_dir(label_dir, "labels")
        print(f"Labels for loading in: {ldir}.")
        mdir = setup_dir(label_dir, "masks")
        print(f"Masks for visualisation in: {mdir}.")
        print(f"Images in: {img_directory}.")
    
    for img_name in tqdm(img_list, desc="Samples", leave=True):
        img_f = img_directory / (img_name + "." + in_fext)
        img = load_image(img_f)
        labels = np.zeros(img.shape[:-1], dtype=np.uint8)
        polylist = poly_dict[img_name]
        for poly in polylist:
            generate_labels(labels, poly, class_idx)
        mask = colour_decoder(labels)
        if overlay:
            mask = overlay_images(img, mask, alpha=overlay)
        else:
            mask = labels
        if label_dir:
            write_img_to_dir(mdir, img_name, mask)
            write_img_to_dir(ldir, img_name, labels)
        else:
            plot_overlaid(mask, title=img_name)