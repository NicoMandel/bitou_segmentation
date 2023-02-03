import numpy as np
import rasterio
from rasterio.plot import reshape_as_image, reshape_as_raster


def is_not_empty(window : np.ndarray, nodata_val : list = [255, 255, 255]) -> bool:
    img = reshape_as_image(window)
    wy = np.equal(img, nodata_val).all(2)
    img[wy] = [0, 0, 0]
    return np.any(img)

def convert_idx(w_width, w_height, cols, i):
    start_x = w_width * (i // cols)
    start_y = w_height * (i % cols)
    return start_x, start_y

