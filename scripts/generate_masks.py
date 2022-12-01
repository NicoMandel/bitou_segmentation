"""
    Script to generate masks using the watershed algorithm for sand labelling
"""

from os import path, mkdir
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

from csupl.watershed import Watershed
from csupl.utils import ColourDecoder, get_image_list, plot_overlaid, overlay_images, load_image, get_colour_decoder
from csupl.generate_masks import merge_classes, generate_labels, get_config, write_image, crop_pair_from_polygon, crop_from_polygon

def parse_args():
    """
        Argument parser
    """
    fdir = path.abspath(path.dirname(__file__))
    def_input = path.join(fdir, "..", "bitou_test")
    
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="Directory to generate masks from", type=str, default=def_input)
    parser.add_argument("-o", "--output", help="Directory where label files should be output to. If none, will cycle through images and plot side by side", type=str, default=None)
    parser.add_argument("--config", help="Location of the config file. If not specified, will look for a .json in the input directory", default=None)
    parser.add_argument("--colour", type=str, help="Which colour code to use. If none, will look for file colour_code.json in default config dir", default=None)
    parser.add_argument("--file-extension", help="Image file extension to read in, with dot. Defaults to .JPG", default=".JPG")
    parser.add_argument("--tolerance", type=float, help="Tolerance to be used for the kmeans algorithm nearest neighbor-distance!", default=0.5)
    parser.add_argument("-w", "--watershed", action="store_true", help="If set to true, will use watershed alorithm to pre-label sand class")
    parser.add_argument("--overlay", action="store_true", help="Whether the mask written out will be an overlay or only the colour code")
    parser.add_argument("-c", "--class", type=int, help="Class index for the polygon to be assigned. Defaults to 2", default=2)
    parser.add_argument("--crop", action="store_true", help="If set, will crop the bounding box surrounding the polygon from the image and use as new image")
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
    tolerance = args["tolerance"]
    img_list = get_image_list(img_directory, in_fext)
    poly_dict = get_config(conf_f, img_directory)
    watershed = args["watershed"]
    overlay = args["overlay"]
    class_idx = args["class"]
    crop=args["crop"]

    if watershed:
        print("Using watershed algorithm")
        ws = Watershed(tolerance=tolerance)

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
    if crop: print("Cropping around Polygon")
    print()
    if label_dir is not None:
        ldir = setup_dir(label_dir, "labels")
        print(f"Labels for loading in: {ldir}")
        mdir = setup_dir(label_dir, "masks")
        print(f"Masks for visualisation in: {mdir}")
        if crop:
            cdir = setup_dir(label_dir, "crops")
            print(f"Crops in: {cdir}")
        
    for im_f in tqdm(img_list):
        img_f = img_directory / (im_f + ".JPG")
        
        # pre-labelling
        img = load_image(img_f)

        if watershed:
            #! check if this is destructive - will change the original image
            labels = ws(img)
            labels = merge_classes(labels, 1)
        else:
            labels = np.zeros(img.shape[:-1], dtype=np.uint8)

        # polygon drawing
        poly_list = poly_dict[im_f]
        # this is where things are being split between cropping and not cropping
        for poly in poly_list:
            generate_labels(labels, poly, class_idx)
        mask = colour_decoder(labels)

        # this is where cropping should happen and things split
        if overlay:
            mask = overlay_images(img, mask, alpha=0.5)
        if crop:
            for i, poly in enumerate(poly_list):
                cr_img, cr_labels = crop_pair_from_polygon(img, labels, poly)
                cr_mask = crop_from_polygon(mask, poly)
                if len(poly_list) > 1:
                    fname = "_".join([im_f, str(i)])
                else:
                    fname = im_f
                
                if label_dir:
                    write_img_to_dir(ldir, fname, cr_labels)
                    write_img_to_dir(mdir, fname, cr_mask)
                    write_img_to_dir(cdir, fname, cr_img)
                else:
                    plot_overlaid(cr_mask, title=fname)
        # if things are not being cropped
        else:
            fname = im_f
            if label_dir:
                write_img_to_dir(mdir, fname, mask)
                write_img_to_dir(ldir, fname, labels)
            else:
                plot_overlaid(mask, title=fname)

            # if overlay:
            #     mask = overlay_images(img, mask, alpha=0.5)
            # if label_dir:
            #     try:
            #         write_image(ldir, im_f, labels)
            #         write_image(mdir, im_f, mask)
            #     except OSError: raise
            # else:
            #     plot_overlaid(mask, title=im_f)
            #         # plot_images(mask, labels, im_f, 0)


