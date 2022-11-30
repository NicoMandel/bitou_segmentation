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
        uses OpenCV.
        Is destructive on the image
    """
    cv2.fillPoly(label_img, pts=np.array([poly_coord], dtype=np.int32), color=int(label_idx))

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
    cv2.imwrite(str(m_path), im_f)

def merge_classes(labels : np.ndarray, keep_class : int) -> np.ndarray:
    labels[labels != keep_class] = 0
    return labels

def crop_image(img : np.ndarray, start_y : int, stop_y : int, start_x : int, stop_x : int) -> np.ndarray:
    return img[start_y:stop_y, start_x:stop_x]

def crop_from_polygon(img : np.ndarray, poly_list : list) -> np.ndarray:
    xy_arr = np.asarray(poly_list)
    x_max, y_max = xy_arr.max(axis=0)
    x_min, y_min = xy_arr.min(axis=0)
    crop = crop_image(img, y_min, y_max, x_min, x_max)
    return crop

def crop_pair_from_polygon(img : np.ndarray, mask : np.ndarray, poly_coords : list) -> tuple[np.ndarray, np.ndarray]:
    nimg = crop_from_polygon(img, poly_coords)
    nmask = crop_from_polygon(mask, poly_coords)
    return nimg, nmask