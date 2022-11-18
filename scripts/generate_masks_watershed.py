"""
    Script to generate masks using the watershed algorithm for sand labelling
"""

import os.path
from pathlib import Path
from PIL import Image
import numpy as np

from csupl.watershed import Watershed
from csupl.utils import get_image_list, plot_images, plot_overlaid, overlay_images, decode_colormap_labels
from csupl.generate_masks import get_polygon_dict, get_polygon_coordinates, convert_classes, generate_mask

def get_default_files():
    fdir = os.path.abspath(os.path.dirname(__file__))
    def_input = os.path.join(fdir, "..", "data", "bitou_test")
    config_f = os.path.join(fdir, "..", "config", "via_bitou_test1_220928.json")
    img_files = get_image_list(def_input)
    img_dir = Path(def_input)
    f_ext = ".JPG"

    return img_dir, img_files, config_f

if __name__=="__main__":
    img_dir, img_files, config_f = get_default_files()
    tolerance = 1.0
    ws = Watershed(tolerance=tolerance)
    
    poly_f = get_polygon_dict(config_f)
    poly_dict = get_polygon_coordinates(poly_f)
    for im_f in img_files:
        img_f = img_dir / (im_f + ".JPG")
        
        # pre-labelling
        img = Image.open(img_f)
        labels = ws(img)
        labels = convert_classes(labels, 1)
        
        # polygon drawing
        mask = np.copy(labels)
        poly_list = poly_dict[im_f]
        for poly in poly_list:
            mask = generate_mask(mask, poly, 2)

        # plot the mask
        # plot_images(img, mask, im_f, ws.classif.K)
        lab_decoded = decode_colormap_labels(mask, 3)
        overlaid = overlay_images(np.array(img), lab_decoded, alpha=0.7)
        plot_overlaid(overlaid)