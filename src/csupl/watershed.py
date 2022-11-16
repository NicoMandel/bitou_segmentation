"""
    File for the watershed algorithm
    See [OpenCV Tutorial] (https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)
    for Segmentation
"""

import cv2
import numpy as np


class Watershed:

    def __init__(self) -> None:
        pass

    def __call__(self, img : np.ndarray, markers : np.ndarray) -> np.ndarray:
        """
            Function to watershed an image.
            Will first preprocess, then call, then postprocess
            params:
                * input image to apply the watershed algorithm to
                * markers: a pre-segmentation with proposals
            Notes:
                * sure classes should be labelled with a number
                * unknown should be marked with 0 
        """
        markers = self.__preprocess(img)
        out_img = cv2.watershed(img, markers)

    def __preprocess(self, img : np.ndarray) -> np.ndarray:
        """
            Use preprocess to generate marker proposals
        """
        pass

    def __postprocess(self):
        """
            Use postprocess to return colour codes and other utilities
        """
        pass