from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import cv2

from csupl.utils import check_path, to_Image, to_Path

def get_config(conf_f : str, img_dir : str) -> dict:
    if conf_f is None:
        img_dir = to_Path(img_dir)
        conf_f = next(img_dir.glob("*.json"))

    json_dict = get_polygon_dict(conf_f)
    poly_dict = get_polygon_coordinates(json_dict)
    return poly_dict

def get_polygon_dict(fpath : str) -> dict:
    fpath = check_path(fpath)
    json_f = open(fpath, "r")
    json_dict = json.load(json_f)

    return json_dict["_via_img_metadata"]

def get_polygon_coordinates(json_dict : dict) -> dict:
    """
        Gets the xy coordinates of the specific image defined by img_fname from the json dict and returns them in the format required by 
        PIL ImageDraw.Polygon : [(x, y), (x,y), ..., ...]
    """
    poly_dict = {}
    for _, v in json_dict.items():
        # if len(v['regions']) > 1:
        #     print(v['filename'])
        if v['regions']:
            xy_list = []
            for reg in v['regions']:
                x = reg['shape_attributes']['all_points_x']
                y = reg['shape_attributes']['all_points_y']
                xy = list(zip(x, y))
                xy_list.append(xy)
        else:
            poly_dict[v['filename'].split('.')[0]] = []
        poly_dict[v['filename'].split('.')[0]] = xy_list
    
    return poly_dict

def generate_mask_image(mask_img : Image.Image, polygon_coord : list, class_idx : int = 1, whiteout : bool = False) -> Image.Image:
    """
        Function to generate a single polygon for a single RGB image and return the image
        Uses Pillow
    """
    # Image modes from Pillow: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    mask_img = to_Image(mask_img)
    d = ImageDraw.Draw(mask_img)
    d.polygon(polygon_coord, fill=(255, 255, 255) if whiteout else (class_idx,0,0))
    return mask_img

def generate_labels(label_img : np.ndarray, poly_coord : list, label_idx : int):
    """
        Function to generate a polygon mask on a single-channel image
        uses OpenCV
    """
    cv2.fillPoly(label_img, pts=np.array([poly_coord], dtype=np.int32), color=int(label_idx))
    return label_img


def write_masks(polygon_coord: dict, input_dir : Path, mask_dir : Path, f_ext : str, whiteout : bool):
    """
        function to load the images and write the mask
    """

    for k, poly_coord_list in tqdm(polygon_coord.items()):
        orig_path = input_dir / ".".join([k,f_ext])
        orig_img = Image.open(orig_path)
        imsize = orig_img.size
        mask_im = Image.new("RGB", imsize)
        for poly_coord in poly_coord_list:
            mask_im = generate_mask_image(mask_im, poly_coord, whiteout)
        mask_path = mask_dir / ".".join([k, f_ext])
        mask_im.save(mask_path, "png")

def write_image(mask_dir : Path, im_fname : str, im_f : np.ndarray, f_ext : str = ".png") -> None:
    """
        OpenCVs saving image function:
        Can be loaded as class indices with cv2.IMREAD_UNCHANGED in the load function
    """
    mask_dir = to_Path(mask_dir)
    m_path = mask_dir / (im_fname + f_ext)
    cv2.imwrite(str(m_path), cv2.cvtColor(im_f, cv2.COLOR_BGR2RGB))

def merge_classes(labels : np.ndarray, keep_class : int) -> np.ndarray:
    labels[labels != keep_class] = 0
    return labels