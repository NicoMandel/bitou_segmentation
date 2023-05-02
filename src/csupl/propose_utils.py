"""
    Utility file for proposing polygons
"""
import json
import numpy as np
import cv2

###################
# Functions for working with the json file
##################
def write_dict(json_dict : dict, out_file : str): 
    """
        writing dictionary into file
    """
    with open(out_file, 'w') as fp:
        json.dump(json_dict, fp)
    print("Written to: {}".format(out_file))

def insert_into_dict(json_dict : dict, img_id : str, img_dict : dict) -> None:
    # add the image to the number of images 
    imglist = json_dict["_via_image_id_list"]
    imglist.append(img_id)
    json_dict["_via_image_id_list"] = imglist
    curr_imgs = json_dict["_via_img_metadata"]
    curr_imgs[img_id] = img_dict
    json_dict["_via_img_metadata"] = curr_imgs      # in-place addition

def get_regions(cnts : np.ndarray) -> list:

    rglist = []
    for rg in cnts:
        # rg = rg.squeeze()
        # if len(rg.shape) >= 2:
        rg_x, rg_y = _get_region(rg)
        region = {
            "name" : "polygon",
            "all_points_x" : rg_x,
            "all_points_y" : rg_y,
        }
        rgdict = {
            "shape_attributes" : region,
            "region_attributes": {"plant" : "bitou_bush"}
            }
        rglist.append(rgdict)
    return rglist

def _get_region(rg : np.ndarray) -> tuple:
    rg_x = rg[:,0].tolist()
    rg_y = rg[:,1].tolist()
    assert len(rg_x) == len(rg_y), "Not same size, cannot be right indexing"
    return rg_x, rg_y

def get_image_dict(img_f, img_fname, cnts : np.ndarray ) -> tuple:
    import os
    sz = os.stat(img_f).st_size
    img_id = img_fname + str(sz)
    regs = get_regions(cnts)
    img_dict = {
        "filename" : img_fname,
        "size" : sz,
        "regions" : regs,
        "file_attributes" : {}
    }

    return img_id, img_dict 

########################
# Image functions
#######################
# Polygons
def get_polygons_from_binary(bin_img: np.ndarray, param : tuple) -> np.ndarray:
    """
        function to get polygons from the binary image.
        Playing around with settings to show polygons
        TODO: consider using RETR_CCOMP as a setting - to get polygons inside polygons: https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    """
    # cleaning - use Gaussian filter

    # use polygon approximation? 

    # base function
    cnts, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_TREE
    
    n_cts = [cnt for cnt in cnts if not _is_too_small(cnt, param)]

    return n_cts

def _is_too_small(cnt : np.ndarray, param : tuple) -> bool:
    """
        function to check if a polygon is too small by checking the extent along x and y
    """
    rg = cnt.squeeze()
    if len(rg.shape) >= 2:
        d_x = rg[:,0].max() - rg[:,0].min()
        d_y = rg[:,1].max() - rg[:,1].min()
    else:
        return True
    return True if (d_x < param[0] and d_y < param[1]) else False 

def get_cnts(cnts):
    """
        utility function based on the fucking pyimagesearch thing because OpenCV has changed their shitty interface. Mofos
        https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py#L154
    """
    if len(cnts) == 2:
        cnts = cnts[0]
    elif len(cnts) == 3:
        cnts = cnts[1]
    else: raise ValueError("Some shit happened to OpenCV. Good luck")
    return cnts

# Tiling functions
def get_tile_numbers(img_shape : tuple, model_shape : tuple) -> tuple:
    """
        Function to get the number of tiles that should be used.
        model_width and height are required
    """
    model_h = model_shape[0]
    model_w = model_shape[1]
    
    h = img_shape[0]
    w = img_shape[1]

    leftover_w = w % model_w
    leftover_h = h % model_h

    # n times the image for the width
    n_w = w // model_w if leftover_w == 0 else (w // model_w) +1
    n_h = h // model_h if leftover_h == 0 else (h // model_h) +1

    return n_w * n_h, n_h

def get_padding(img_shape : tuple, model_shape : tuple, halo : int = 256) -> tuple:
    """
        function to get the padding in the format:
        pad_left, pad_right, pad_top, pad_bottom
        the img_shape and model_shape indices hav to coincide
    """
    model_h = model_shape[0]
    model_w = model_shape[1]

    h = img_shape[0]
    w = img_shape[1]

    extra_w = model_w - (w % model_w)
    extra_h = model_h - (h % model_h)
    # extra_x = model_w - leftover_w
    # extra_y = model_h - leftover_h
    
    pad_left = int(halo)
    pad_right = int(extra_w + halo)
    pad_top = int(halo)
    pad_bottom = int(extra_h + halo)

    return pad_left, pad_right, pad_top, pad_bottom

def get_window_dims(model_shape : tuple, halo : int = 256) -> tuple:
    """
        Function to get the dimensions of the window
    """
    return model_shape[0] + 2*halo, model_shape[1] + 2*halo 

def get_stride(model_shape : tuple) -> tuple:
    """
        Function to get the stride of each dimension
    """
    return model_shape

def pad_image(img : np.ndarray, pad_top : int, pad_left : int, pad_right : int, pad_bottom : int) -> np.ndarray:
    padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT) # BORDER_REPLICATE, BORDER_REFLECT_101, BORDER_WRAP
    return padded

def get_out_shape(n_tot : int, n_h : int, model_shape : tuple) -> tuple:
    """
        get the shape of the image output - depends on n_x and n_y
    """
    n_w = int(n_tot / n_h)
    return (n_h * model_shape[0], n_w * model_shape[1])

def get_final_image(out_img : np.ndarray, img_shape : tuple) -> np.ndarray:
    """
        Get the final image from the oversized overhanging image
    """
    return out_img[:img_shape[0], :img_shape[1]]

# General
def too_large(img : np.ndarray) -> bool:
    return True if (img.shape[0] > 1024 or img.shape[1] > 1024)  else False