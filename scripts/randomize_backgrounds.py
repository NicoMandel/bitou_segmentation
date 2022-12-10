"""
    File for offline augmentation - for background randomization
    Cannot be performed online due to albumentations call syntax
"""

from os import path
from argparse import ArgumentParser
import numpy as np
import cv2

from csupl.utils import to_Path, get_image_list, load_image, load_label, replace_image_values, plot_overlaid, write_image

def parse_args():
    """
        Argument parser
    """
    fdir = path.abspath(path.dirname(__file__))
    def_input = path.join(fdir, "..", "bitou_test")
    
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="Directory where original images are stored", type=str, default=def_input)
    parser.add_argument("-r", "--random", help="Directory where random images come from", type=str)
    parser.add_argument("-l", "--labels", help="Directory where the labels are loaded from", type=str)
    parser.add_argument("-o", "--output", help="Directory where newly generated files should be output to. Both new masks and new images.\
                        If none, will cycle through images and plot side by side", type=str, default=None)
    parser.add_argument("--file-extension", help="Image file extension to read in, with dot. Defaults to .png", default=".png")
    parser.add_argument("-c", "--class", type=int, help="Class index for the background which will be overlaid. Defaults to 0", default=0)
    # TODO: make this generic - can it work with a list - see [here](https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse)
    parser.add_argument("--limit", help="What to use as a limit. Number of images or run full directories against one another [0]. May take a long time. Defaults to 100",
                        default=100)
    args = vars(parser.parse_args())
    return args

if __name__=="__main__":
    args = parse_args()

    label_dir = to_Path(args["labels"])
    random_dir = to_Path(args["random"])
    input_dir = to_Path(args["input"])

    f_ext = args["file_extension"]

    img_list = get_image_list(input_dir, f_ext=".JPG")
    assert len(img_list) > 0
    exchange_class = args["class"]

    # random generation of new image
    random_list = get_image_list(random_dir, f_ext=".JPG")
    assert len(random_list) > 0
    rng = np.random.default_rng()

    for img_fname in img_list:
        im_path = input_dir / (img_fname + ".JPG")
        img = load_image(im_path)
        assert img is not None
        label_path = label_dir / (img_fname + f_ext)
        label = load_label(label_path)
        assert label is not None

        r_img_fname = rng.choice(random_list)
        r_img_f = random_dir / (r_img_fname + ".JPG")
        r_img = load_image(r_img_f)
        assert r_img is not None

        out_img = replace_image_values(img, r_img, label, exchange_class)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        if args["output"] is None:
            plot_overlaid(out_img, "+".join([img_fname, r_img_fname]))
        else:
            # TODO: update the names with random integers
            write_image(args["output"], img_fname, out_img)
            # TODO: also write a mask out
