import numpy as np
import os.path
from pathlib import Path
import rasterio
from rasterio.plot import reshape_as_image, reshape_as_raster

ALLOWED_EXTENSIONS = set(['tif', 'tiff', 'geotif', 'geotiff'])
PROCESSED_PREFIX = "DNN"
JOIN_CHAR = "_"

def is_not_empty(window : np.ndarray, nodata_val : list = [255, 255, 255]) -> bool:
    img = reshape_as_image(window)
    wy = np.equal(img, nodata_val).all(2)
    img[wy] = [0, 0, 0]
    return np.any(img)

def convert_idx(w_width, w_height, cols, i):
    start_x = w_width * (i // cols)
    start_y = w_height * (i % cols)
    return start_x, start_y

def get_tiff_files(path : str) -> list:
    """
        Function to get all .tiff file extensions in a folder
        and exclude the ones with the DNN prefix
    """
    plPath = Path(path)
    img_list = ([[x for x in plPath.glob(".".join(["*",fext]))] for fext in ALLOWED_EXTENSIONS])
    img_l = [val for sublist in img_list for val in sublist if val.parts[-1].split(JOIN_CHAR)[0] != PROCESSED_PREFIX]
    return img_l