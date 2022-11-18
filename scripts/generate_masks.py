"""
    Script to generate masks using the watershed algorithm for sand labelling
"""

from os import path, mkdir
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

from csupl.watershed import Watershed
from csupl.utils import get_image_list, plot_overlaid, overlay_images, decode_colormap_labels, read_image
from csupl.generate_masks import merge_classes, generate_labels, get_config, write_image

def parse_args():
    """
        Argument parser
    """
    fdir = path.abspath(path.dirname(__file__))
    def_input = path.join(fdir, "..", "bitou_test")
    
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="Directory to generate masks from", type=str, default=def_input)
    parser.add_argument("-o", "--output", help="Directory where label files should be output to. If none, will cycle through images and plot side by side", type=str, default=None)
    parser.add_argument("-c", "--config", help="Location of the config file. If not specified, will look for a .json in the input directory", default=None)
    parser.add_argument("--file-extension", help="Image file extension to read in, with dot. Defaults to .JPG", default=".JPG")
    parser.add_argument("--tolerance", type=float, help="Tolerance to be used for the kmeans algorithm nearest neighbor-distance!", default=0.5)
    parser.add_argument("-w", "--watershed", action="store_true", help="If set to true, will use watershed alorithm to pre-label sand class")
    args = vars(parser.parse_args())
    return args

if __name__=="__main__":
    args = parse_args()
    
    # Files
    img_directory = Path(args["input"])
    label_dir = args["output"]
    in_fext = args["file_extension"]
    conf_f = args["config"]

    # Getting files and algorithm set up
    tolerance = args["tolerance"]
    img_list = get_image_list(img_directory, in_fext)
    poly_dict = get_config(conf_f, img_directory)
    watershed = args["watershed"]
    if watershed:
        ws = Watershed(tolerance=tolerance)

    # Logging
    print("Reading from directory: {}\t{} files".format(
        img_directory, len(img_list)
    ))

    # Setting up output
    if label_dir is not None:
        try:
            ldir = path.join(label_dir, "labels")
            mdir = path.join(label_dir, "masks")
            mkdir(ldir)
            mkdir(mdir)
            print(f"Writing labels (for loading) to directory: {ldir}")
            print(f"Writing masks (for visualisation) to directory: {mdir}")
        except OSError: raise
    print("Output: {}".format(
        "plotting" if label_dir is None else label_dir
    ))
    
    for im_f in tqdm(img_list):
        img_f = img_directory / (im_f + ".JPG")
        
        # pre-labelling
        img = read_image(img_f)

        if watershed:
            labels = ws(img)
            labels = merge_classes(labels, 1)
        else:
            labels = np.zeros(img.shape[:-1], dtype=np.uint8)
        # polygon drawing
        poly_list = poly_dict[im_f]
        poly_idx = labels.max() + 1
        for poly in poly_list:
            labels = generate_labels(labels, poly, poly_idx)

        mask = decode_colormap_labels(labels)
        mask = overlay_images(img, mask, alpha=0.5)
        if label_dir:
            try:
                write_image(ldir, im_f, labels)
                write_image(mdir, im_f, mask)
            except OSError: raise
        else:
            plot_overlaid(mask, title=im_f)
            # plot_images(mask, labels, im_f, 0)


