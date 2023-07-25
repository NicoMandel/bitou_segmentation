"""
    python script to convert semantic segmentation masks into a bounding box format
"""

import cv2
from argparse import ArgumentParser
from tqdm import tqdm
import os.path
from csupl.utils import get_image_list, load_image

def parse_args():
    """
        Argument parser
    """
       
    parser = ArgumentParser(description="File for turning a directory of images with masks into bounding boxes in yolo format")
    parser.add_argument("-i", "--input", help="Directory to generate bboxes from", type=str, required=True)
    parser.add_argument("-o", "--output", help="Directory where bbox txt files will be saved. If none, will cycle through images and output to console", type=str, default=None)
    parser.add_argument("-p", "--plot", help="Not implemented yet", type=str, default=None)

    args = vars(parser.parse_args())
    return args

def write_to_f(mask, outfile : str):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours: print(outfile)
    with open(outfile, 'w') as f:
        for contour in contours:
            # todo - add support to get the class index here
            x, y, w, h = cv2.boundingRect(contour)
            f.write('0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                (x+w/2) / mask.shape[1], (y+h/2) / mask.shape[0], w/mask.shape[1], h/mask.shape[0]
            ))
    return None


if __name__=="__main__":
    fdir = os.path.abspath(os.path.dirname(__file__))
    args = parse_args()
    flist, fext = get_image_list(args["input"])
    for imgname in tqdm(flist):
        imgfname = os.path.join(args["input"], ".".join([imgname, fext]))
        img = load_image(imgfname)
        outfname = os.path.join(args["output"], ".".join([imgname, "txt"]))
        write_to_f(img, outfname)
        