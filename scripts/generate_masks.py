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
    parser.add_argument("--file-extension", help="Image file extension to read in, with dot. Defaults to .JPG", default=".JPG")
    # parser.add_argument("--tolerance", type=float, help="Tolerance to be used for the kmeans algorithm nearest neighbor-distance!", default=0.5)
    # parser.add_argument("-w", "--watershed", action="store_true", help="If set to true, will use watershed alorithm to pre-label sand class")
    parser.add_argument("--overlay", type=float, help="Whether the mask written out will be an overlay or only the colour code. Alpha value", default=None)
    parser.add_argument("-c", "--class", type=int, help="Class index for the polygon to be assigned. Defaults to 1 for binary classification", default=1)
    parser.add_argument("--crop", type=int, help="If set, will generate a crop of the specified resolution", default=None)
    parser.add_argument("--limit", type=int, help="Number of samples to generate for each positive and negative. Defaults to 300", default=300)
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
    crop = args["crop"]

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
    if crop: print("Cropping")

    if label_dir is not None:
        ldir = setup_dir(label_dir, "labels")
        ldir_pos = setup_dir(ldir, "positive")
        ldir_neg = setup_dir(ldir, "negative")
        print(f"Labels for loading in: {ldir}. Positive samples, including Bitou in: {ldir_pos}, Negative samples in: {ldir_neg}")
        mdir = setup_dir(label_dir, "masks")
        mdir_pos = setup_dir(mdir, "positive")
        mdir_neg = setup_dir(mdir, "negative")
        print(f"Masks for visualisation in: {mdir}. Positive samples, including Bitou in: {mdir_pos}, Negative samples in: {mdir_neg}")
        img_dir = setup_dir(label_dir, "crops" if crop else "images")
        img_dir_pos = setup_dir(img_dir, "positive")
        img_dir_neg = setup_dir(img_dir, "negative")
        print(f"Images in: {img_dir}. Positive samples in: {img_dir_pos}, Negative samples in: {img_dir_neg}")
        
    pos_dict, neg_list = split_polyon_dict(poly_dict)

    if crop:
        RC = A.RandomCrop(args["crop"], args["crop"])
    
    px_ct_total = 0
    px_ct_class = 0

    pos_samples = np.random.choice(list(pos_dict.keys()), args["limit"], replace=True)
    pos_sample_checker = dict.fromkeys(pos_samples, 0)
    for img_name in tqdm(pos_samples, desc="Positive Samples", leave=True):
        img_f = img_directory / (img_name + ".JPG")
        img = load_image(img_f)
        labels = np.zeros(img.shape[:-1], dtype=np.uint8)
        polyList = pos_dict[img_name]
        for poly in polyList:
            generate_labels(labels, poly, class_idx)
        if crop:
            # generate random crop in while not loop - which checks the coordinates
            out = RC(image=img, mask=labels)
            lab = out['mask']
            im = out['image']
            while not np.any(lab == class_idx):
                out = RC(image=img, mask=labels)
                lab = out['mask']
                im = out['image']
            labels = lab
            img = im
        mask = colour_decoder(labels)
        px_ct_total += labels.size
        px_ct_class += np.count_nonzero(labels == class_idx)
        if overlay:
            mask = overlay_images(img, mask, alpha=args["overlay"])
        if label_dir:
            fname = "_".join([img_name, str(pos_sample_checker[img_name])])
            pos_sample_checker[img_name] += 1 
            write_img_to_dir(mdir_pos, fname, mask)
            write_img_to_dir(ldir_pos, fname, labels)
            write_img_to_dir(img_dir_pos, fname, img)
        else:
            plot_overlaid(mask, title=img_name)
        

    print("Percentage of pixels being class in the positive dataset: {:.2f}%%".format((px_ct_class / px_ct_total) * 100))

    neg_samples = np.random.choice(neg_list, args["limit"], replace=True)
    neg_samples_checker = dict.fromkeys(neg_samples, 0)
    for img_name in tqdm(neg_samples, desc="Negative samples", leave=True):
        img_f = img_directory / (img_name + ".JPG")
        img = load_image(img_f)
        labels = np.zeros(img.shape[:-1], dtype=np.uint8)
        if crop:
            out = RC(image=img, mask=labels)
            img = out['image']
            labels = out['mask']
        mask = colour_decoder(labels)
        if overlay:
            mask = overlay_images(img, mask, alpha=args["overlay"])
        if label_dir:
            fname = "_".join([img_name, str(neg_samples_checker[img_name])])
            neg_samples_checker[img_name] += 1 
            write_img_to_dir(mdir_neg, fname, mask)
            write_img_to_dir(ldir_neg, fname, labels)
            write_img_to_dir(img_dir_neg, fname, img)

        else:
            plot_overlaid(mask, title=img_name)

