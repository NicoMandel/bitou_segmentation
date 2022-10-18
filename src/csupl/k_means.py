"""
    Following [this blog](https://cierra-andaur.medium.com/using-k-means-clustering-for-image-segmentation-fe86c3b39bf4)
    using help from [OpenCV docs](https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html)
    and [here](https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi'] = 300

def disable_cluster(img : np.array, cluster : int, labels, color : list = [255,255,255]) -> np.array:
    """
        Function to disable a cluster from visualisation and return the image
    """
    masked = np.copy(img)
    masked = masked.reshape((-1,3))
    masked[labels == cluster] = color     # turning the specified cluster white
    masked = masked.reshape(img.shape)
    return masked

def resize_img(img : np.array, scale_perc : int = 50) -> np.array:
    """
        function to resize the image to whatever is specified by the scale
    """
    width = int(img.shape[1] * scale_perc / 100)
    height = int(img.shape[0] * scale_perc / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    return resized

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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)    # Alternative: criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, label, center = cv2.kmeans(vect, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)       # alternative: cv2.KMEANS_RANDOM_CENTERS - random initialized centers
    
    # converting the image back
    label = label.flatten()
    center = np.uint8(center)
    res = center[label]
    res = res.reshape((img.shape))
    return res, label

def get_image_list(dir : str, f_ext : str = ".JPG") -> list:
    img_list = list([x.stem for x in dir.glob("*"+f_ext)])
    return img_list
    
def read_image(img_path : str) -> np.array:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def plot_images(img : np.ndarray, mask : np.ndarray, img_name : str, K : int) -> None:
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title('Original')

    axs[1].imshow(mask)
    axs[1].axis('off')
    axs[1].set_title('Clusters')
    fig.suptitle(f"Image: {img_name}, Clusters: {K}")
    plt.show()

def save_image(outfig : str, mask : np.ndarray) -> None:
    cv2.imwrite(outfig, mask)

