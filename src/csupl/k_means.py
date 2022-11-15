"""
    Following [this blog](https://cierra-andaur.medium.com/using-k-means-clustering-for-image-segmentation-fe86c3b39bf4)
    using help from [OpenCV docs](https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html)
    and [here](https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python)
"""

from typing import Any
import numpy as np
import cv2
# plt.rcParams['figure.dpi'] = 300
from sklearn.cluster import KMeans
import pickle

from itertools import combinations

from src.csupl.utils import decode_colormap, resize_img


def cluster_img(img : np.array, K : int = 4, attempts : int = 10, iterations : int = 100, epsilon : float = 0.2):
    """
        Function to perform the actual clustering
            @params:
            K = number of clusters
            attempts = attempts to run on the image
            iterations = maximu number of iterations for the algorithm.
            epsilon = cluster stopping criteria

            default criteria are:
                iter = 100
                eps = 0.2
            OR:
                10
                1.0
    """
    # Converting image into m x 2 matrix
    vectorized = img.reshape((-1,3))
    vect = np.float32(vectorized)
    # Setting criteria
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)    # Alternative: criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # _, label, center = cv2.kmeans(vect, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)       # alternative: cv2.KMEANS_RANDOM_CENTERS - random initialized centers
    
    # using the random state so that we can see if we get the same clusters
    km = KMeans(K, max_iter=iterations,tol=epsilon, random_state=42).fit(vect)
    label = km.labels_
    center = km.cluster_centers_

    # using a lookup table to convert clusters
    idx = np.argsort(center.sum(axis=1))
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(K)

    # converting the image back
    label = lut[label.flatten()]
    center = np.uint8(center)
    res = center[label]
    res = res.reshape((img.shape))
    return res, label

class km_algo:

    def __init__(self, K : int = 4, attempts : int = 10, iterations : int =100, epsilon : float = 0.2,
                scale : int = None, hsv : bool = False, 
                # overlay : bool = False
                ) -> None:
        self.km = KMeans(K, max_iter=iterations, tol=epsilon, random_state=42)
        # self.labels = None
        self.centers = None
        self.K = K

        self.hsv = hsv
        self.scale = scale
        # self.overlay = overlay

    def fit(self, img : np.ndarray):
        """
            Fitting on a single image
        """
        # img = self.preprocess_img(img, "fit")
        vect = self._preprocess(img)
        self.km.fit(vect)
        # self.labels = self.km.labels_
        self.centers = np.uint8(self.km.cluster_centers_)

    def predict(self, img : np.ndarray):
        """
            predicting on a single image
        """
        vect = self._preprocess(img)
        label = self.km.predict(vect)
        # label = label.flatten()
        res = self.centers[label]
        res = res.reshape((img.shape))
        return res, label


    def __call__(self, inp : np.ndarray, overlay : bool) -> np.ndarray:
        inp = self.preprocess_img(inp, "predict")
        m, l = self.predict(inp)
        m = self._postprocess_mask(m, l, overlay)
        return m
    
    def _lookup(self, idx : np.ndarray):
        raise NotImplementedError

    def _preprocess(self, img : np.ndarray) -> np.ndarray:
        vect = img.reshape((-1,3))
        return np.float32(vect)
    
    def preprocess_img(self, img: np.ndarray, mode : str = "predict") -> np.ndarray:
        """
            Function to preprocess a single image
        """
        if self.scale and mode == "fit":
            img = resize_img(img, self.scale)
        if self.hsv:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img        

    def _postprocess_img(self, img: np.ndarray) -> np.ndarray:
        """
            Function to postprocess a single image
        """
        if self.hsv:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img
    
    def _postprocess_mask(self, mask : np.ndarray, labels : np.ndarray, overlay : bool, classes : int = None) -> np.ndarray:
        """
            Function to postprocess a mask
        """
        mask = self._postprocess_img(mask)
        if overlay:
            #
            mask = decode_colormap(mask, labels, self.K if classes is None else classes)
        return mask

    def calculate_distance(self, img : np.ndarray, tol : float = 1.):
        """
            Function to calculate the distance to the centers with a tolerance
            TODO: watershed algorithm prediction
            In HSV:
                * hue: 0 - 179 (angle)
                * saturation: 0 - 255
                * value: 0 - 255
            In RGB: 0 - 255
        """
        # tol should be a list of tolerance per dimension. Depending on the range of the axis
        img = self.preprocess_img(img)
        inp = self._preprocess(img)
        outp = np.zeros(inp.shape[0]).astype(np.int)

        mindist = self._min_dist()
        for i in range(self.K):
            tolvec = tol * mindist[i,:]
            cvec = self.centers[i,:]
            darr = np.abs(inp - cvec)
            dtol_arr = (darr - tolvec)
            # dtol_arr = dtol_arr[dtol_arr < 0.
            # .sum(axis=1)
            # dtol_arr = dtol_arr * -1.
            outp[(dtol_arr < 0.).sum(axis=1) == inp.shape[1]] = i+1 
            # b_arr = np.isclose(inp, rvec, rtol=tol)    # todo: figure out rtol and atol
            # outp[b_arr] = i+1

        outp = outp.flatten()
        # to make this equal to the predict function, it should output an array with the set colors already as the first part of the tuple
        ncenters = np.insert(self.centers, 0, np.array([179., 255., 255.]), 0)
        # and in the second part of the tuple a long vector with the labels - that can be used by "postprocess"
        res = ncenters[outp]
        res = res.reshape((img.shape))
        # but for "postprocess_mask" this is irrelevant - the mask is only used to create the image size. Outp is important
        return res, outp
        

    def _min_dist(self) -> np.ndarray:
        """
            Function to calculate the minimum distance of the center array
        """
        md = np.ones_like(self.centers) * 255.
        pairs = list(combinations(range(0, self.K), 2))
        d_3 = np.ones((self.K, self.K, self.centers.shape[1])) * 255.
        for i,j in pairs:
            d = np.abs(self.centers[i,:].astype(np.int) - self.centers[j,:].astype(np.int))
            d_3[i,j] = d_3[j,i] = d
        return d_3.min(axis=0)
        d1 = d_3.min(axis=0)
        d2 = d_3.min(axis=1)
        d3 = d_3.min(axis=2)


        for k in range(0, self.K):
            colvec = self.centers[k,:].astype(np.int)
            pairs = list(combinations(range(0, self.K), 2))
            dmat = np.ones((self.K, self.K, len(colvec))) * 255.
            for i,j in pairs:
                d = np.abs(colvec[i] - colvec[j])
                dmat[i,j,:] = dmat[j,i,:] = d
            md[:,k] = dmat.min(axis=0)
        return md



    # Saving and loading functions
    def save_classifier(self, fname : str, f_ext : str=".pkl"):
        fn = fname + f_ext
        with open(fn, 'wb') as f:
            pickle.dump(self, f)
        print("Saved to: {}".format(fn))

    @classmethod
    def load_classifier(cls, fname : str):
        print("Loading from: {}".format(fname))
        with open(fname, 'rb') as f:
            classif = pickle.load(f)
        return classif
        
