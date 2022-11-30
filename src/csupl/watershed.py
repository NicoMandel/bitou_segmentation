"""
    File for the watershed algorithm
    See [OpenCV Tutorial] (https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)
    for Segmentation
"""

import cv2
import numpy as np
from csupl.k_means import km_algo
from csupl.utils import to_numpy


class Watershed:

    def __init__(self, tolerance : float = 1.0) -> None:
        fpath = "results/kmeans/classifiers/kmeans_K-3_scale-20_hsv_full.pkl"
        self.classif = km_algo.load_classifier(fpath)
        self.tolerance = tolerance


    def __call__(self, img : np.ndarray) -> np.ndarray:
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
        nimg = to_numpy(img)
        markers = self.__preprocess(nimg)
        out_img = cv2.watershed(img, markers.astype(np.int32))
        out_img = self.__postprocess(out_img)
        return out_img

    def __preprocess(self, img : np.ndarray) -> np.ndarray:
        """
            Use preprocess to generate marker proposals
        """
        _, label = self.classif.calculate_distance(img, tol = self.tolerance)
        return label

    def __postprocess(self, out_img : np.ndarray) -> np.ndarray:
        """
            Use postprocess to return colour codes and other util.
            Watershed output:
                * 0 = unknown
                * -1 = border
                * rest are classes
        """
        kernel = np.ones((5,5), np.uint8)
        n_img = cv2.dilate(out_img.astype(np.float32), kernel, 1)
        return n_img.astype(np.uint8)
