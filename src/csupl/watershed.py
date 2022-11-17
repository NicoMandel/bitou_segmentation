"""
    File for the watershed algorithm
    See [OpenCV Tutorial] (https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)
    for Segmentation
"""

import cv2
import numpy as np
from csupl.k_means import km_algo


class Watershed:

    def __init__(self) -> None:
        # TODO: initialise the best k_means classifier here
        # TODO afterwards - use for pre-labelling
        # TODO afterwards - generate masks with pre-labelled data & overlay from hand-generated mask
        fpath = "results/kmeans/classifiers/kmeans_K-3_scale-20_hsv_full.pkl"
        self.classif = km_algo.load_classifier(fpath)
        self.sand_label = 1
        self.tolerance = 1.0


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
        n_img = cv2.dilate(out_img.astype(np.int32), kernel, 1)
        return n_img