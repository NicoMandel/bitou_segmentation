""""
    utility functions for working with hyperspectral data (cubes). Mainly conversion functions to RGB
    using the code from:
        * [Tutorial](https://personalpages.manchester.ac.uk/staff/d.h.foster/Tutorial_HSI2RGB/Tutorial_HSI2RGB.html#ref6)
        * [Paper](https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-36-4-606&id=407438)
    TODO:
        * MatLab simple approaches (summing over the appropriate channels), [in the Matlab Documentation](https://de.mathworks.com/help/images/ref/hypercube.colorize.html)
            * Three most informative bands
            * sum  of appropriate channels
            * Most informative Bands [Matlab Documentation](https://de.mathworks.com/help/images/ref/hypercube.selectbands.html), [Paper](https://ieeexplore.ieee.org/document/4656481)
        * From a GitHub repo looking at colour matching functions [repo](https://github.com/JakobSig/HSI2RGB)
        * Paper is available [here](https://ieeexplore.ieee.org/document/9323397)
"""

import numpy as np
import scipy.io
from csupl.utils import check_path 
import os.path

fpath = os.path.abspath(os.path.dirname(__file__))

# Convenience Function
def load_from_mat(mat_loc) -> dict:
    pth = check_path(mat_loc)
    data = scipy.io.loadmat(pth)
    return data

# Loading luminance data
def load_luminance(path : str) -> dict:
    if path in ["illum_6500", "illum_4000", "illum_25000"]:
        confdir = os.path.join(fpath, '..', 'config', 'hyperspectral')
        path = os.path.join(confdir, path + ".mat")
    return load_from_mat(path)

# loading xyzbar
def _load_xyzbar() -> np.ndarray:
    xyzbar_path = os.path.join(fpath, '..', 'config', 'hyperspectral', 'xyzbar.mat')
    xyz = load_from_mat(xyzbar_path)['xyzbar']
    return xyz

# Checking the image
def _check_hs_img(hs_img : np.ndarray):
    assert hs_img.shape[2] == 33, "Hypercube dimension not 33, Should be format H x W x 33, is: {}".format(hs_img.shape)

# Clipping values to [0, 1]
def _clip_arr(arr : np.ndarray) -> np.ndarray:
    arr[arr < 0] = 0
    arr = arr / arr.max()
    return arr

# actual conversion
def xyzsrgb(arr : np.ndarray) -> np.ndarray:
    """
        Actual implementation of the conversion function
    """
    m = np.asarray(
        [[3.2406, -1.5372, -0.4986], 
        [-0.9689,  1.8758, 0.0414], 
        [0.0557, -0.2040, 1.0570]]
    )
    d = arr.shape
    r = d[0] * d[1]
    w = d[-1]
    narr = np.reshape(arr, (r, w))

    sRGB = m @ narr.T
    sRGB = sRGB.T.reshape(d)
    return sRGB

# Gamma correction
def _scale_arr(arr : np.ndarray) -> np.ndarray:
    return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')

def gamma_correction(img : np.ndarray, gamma : float = 2.0) -> np.ndarray:
    nimg = _scale_arr(img)
    invGamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return lut[nimg]


def convert_to_rgb(hs_img : np.ndarray, illum_format : str = "illum_6500", gamma : float = 1.0) -> np.ndarray:
    """
        Function to convert a hyperspectral image of dimensions H x W x 33 to RGB
    """
    _check_hs_img(hs_img)
    # Radiance calculation
    illum = load_luminance(illum_format)[illum_format]
    rad = np.copy(hs_img)
    for i in range(illum.size):
        rad[...,i] = hs_img[...,i] * illum[i]
    
    # xyzbar estimation
    xyzbar = _load_xyzbar()
    rad_2D = rad.reshape((rad.shape[0] * rad.shape[1], rad.shape[2]))
    xyz = (xyzbar.T @ rad_2D.T).T

    # xyz clipping
    xyz = xyz.reshape((hs_img.shape[0], hs_img.shape[1], 3))
    xyz = _clip_arr(xyz)

    # xyzsrgb conversion
    sRGB = xyzsrgb(xyz)
    # gamma correction if required
    if gamma != 1.0:
        sRGB = gamma_correction(sRGB)
    return sRGB