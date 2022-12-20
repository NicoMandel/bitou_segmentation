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
        xy_list = []
        if v['regions']:
            for reg in v['regions']:
                x = reg['shape_attributes']['all_points_x']
                y = reg['shape_attributes']['all_points_y']
                xy = list(zip(x, y))
                xy_list.append(xy)
        poly_dict[v['filename'].split('.')[0]] = xy_list
    
    return poly_dict

def split_polyon_dict(poly_dict : dict) -> tuple:
    """
        Function to split a polygon dictionary into a positive dictionary and a negative list
    """
    pos_dict = {}
    neg_list = []
    for k, v in poly_dict.items():
        if v:
            pos_dict[k] = poly_dict[k]
        else:
            neg_list.append(k)

    return pos_dict, neg_list


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


################################################
# Section on Balancing out pixel classes - currently unused
################################################
def _is_between(lower, upper, testval) -> bool:
    return min(lower, upper) < testval < max(lower, upper)

def _dilate_by(coords : tuple, boundaries : tuple, px_ct : int = 1) -> tuple:
    x_min, y_min, x_max, y_max = coords
    x_min -= px_ct if x_min > 0 else 0
    y_min -= px_ct if y_min > 0 else 0
    x_max += px_ct if x_max < (boundaries[1] - 1) else (boundaries[1] - 1)
    y_max += px_ct if y_max < (boundaries[0] - 1) else (boundaries[0] - 1)
    return (x_min, y_min, x_max, y_max)

def _get_ratio(crop : np.array, backgrd_idx : int = 0) -> float:
    m_ct = np.count_nonzero(crop > backgrd_idx)
    background_ct = np.count_nonzero(crop == backgrd_idx)
    ratio = background_ct / (m_ct + background_ct)
    return ratio

def balance_image(mask_img : Image.Image, balance_ratio : float, coords : tuple, epsilon : float = 0.05) -> tuple:
    """
        Function to balance out the image pixels.
        balance_ratio is the targeted amount of **Background** pixels
        Note: only for binary case currently
    """
    m_arr = np.asarray(mask_img)[... ,0]      # only concered about red channel
    boundaries = m_arr.shape
    x_min, y_min, x_max, y_max = coords
    init_crop = m_arr[y_min:y_max+1, x_min:x_max+1]
    ratio = _get_ratio(init_crop, 0)
    while not _is_between(balance_ratio-epsilon, balance_ratio+epsilon, ratio):   
        # crop image around the coordinates
        x_min, y_min, x_max, y_max = coords
        crop_arr = m_arr[y_min:y_max+1,x_min:x_max+1]
        
        # get the pixel ratio
        ratio = _get_ratio(crop_arr)
        # tqdm.write(f"{ratio}")
        # expand the pixel ratio
        coords = _dilate_by(coords, boundaries)
    return coords