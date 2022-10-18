"""
    Following [this blog](https://cierra-andaur.medium.com/using-k-means-clustering-for-image-segmentation-fe86c3b39bf4)
    using help from [OpenCV docs](https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html)
    and [here](https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python)
    TODO: run the clusters on ALL images at once - reshape
    ! k means is unsupervised, KNN is supervised
"""

import numpy as np
import argparse
# from sklearn.cluster import KMeans
import cv2
import os.path
from os import mkdir
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
# plt.rcParams['figure.dpi'] = 300

def parse_args():
    """
        Argument parser
    """
    # default directory setup
    fdir = os.path.abspath(os.path.dirname(__file__))
    def_input = os.path.join(fdir, "..", "data", "bitou_test")
    data_dir = os.path.join(os.path.dirname(def_input), "..", "results") 

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Directory to generate masks from", type=str, default=def_input)
    parser.add_argument("-o", "--output", help="Directory where masks should be output to.\
                        will create a subfolder with a string concatenated by the K-means settings.\
                        if argument plot is given, will not create output folder", type=str, default=data_dir)
    parser.add_argument("-K", help="Number of clusters K to use. Default is 4",type=int, default=4)
    parser.add_argument("-s", "--scale", help="scale for resizing the images in percentage. If None is given, will not resize", type=int, default=0)
    parser.add_argument("-e", "--epsilon", help="epsilon stopping criteria for the KMeans clustering algorithm. Defaults to 0.2", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, help="Iterations to run the algorithm. Defaults to 100", default=100)
    parser.add_argument("--overlay", help="Class index to whiteout. If not given, will not overlay. 1-indexed", default=0, type=int)
    parser.add_argument("--file-extension", help="Image file extension, with dot. Defaults to JPG", default=".JPG")
    parser.add_argument("-p", "--plot", help="Index to plot. If given, will not write directory out. 1-indexed!", default=0, type=int)
    args = vars(parser.parse_args())
    return args

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

if __name__=="__main__":
    args = parse_args()

    img_dir = Path(args["input"])
    f_ext = args["file_extension"]
    img_list = get_image_list(img_dir, f_ext=f_ext)

    # Clustering settings
    K = args["K"]
    epsilon = args["epsilon"]
    iterations = args["iterations"]
    overlay_class = args["overlay"]
    scale = args["scale"]

    # Reading image
    plot_idx = args["plot"]
    if plot_idx:
        img_name = img_list[plot_idx-1]
        fname = img_dir / (img_name+f_ext)
        img = read_image(str(fname))
        if scale:
            img = resize_img(img, int(scale))

        # doing the clustering
        mask, labels = cluster_img(img, K=K, iterations=iterations, epsilon=epsilon)

        # whiteout the xths cluster
        if overlay_class:
            mask = disable_cluster(mask, overlay_class-1, labels)

        # Plot the images
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(img)
        axs[0].axis('off')
        axs[0].set_title('Original')

        axs[1].imshow(mask)
        axs[1].axis('off')
        axs[1].set_title('Clusters')
        fig.suptitle(f"Image: {img_name}, Clusters: {K}")

        plt.show()
        print("Test line for debugging")
    
    
    else:
        print("Reading entire directory {}, {} images".format(img_dir, len(img_list)))
        outdir_name = "K-{}_scale-{}_Overlay-{}".format(K, scale, overlay_class-1)
        output_parentdir = args["output"]
        outdir = os.path.join(output_parentdir, outdir_name)

        try:
            mkdir(outdir)
            # Section to read the entire directory
            for img_name in tqdm(img_list):
                fname = img_dir / (img_name+f_ext)
                img = read_image(str(fname))
                if scale:
                    img = resize_img(img, scale)

                # doing the clustering
                mask, labels = cluster_img(img, K=K, iterations=iterations, epsilon=epsilon)

                # whiteout the xths cluster
                if overlay_class:
                    mask = disable_cluster(mask, overlay_class-1, labels)

                # Save the image
                outfig = os.path.join(outdir, img_name + ".jpg")
                tqdm.write("Saving to: {}".format(outfig))
                cv2.imwrite(outfig, mask)
                # # Write the images
                # fig, axs = plt.subplots(1,2)
                # axs[0].imshow(img)
                # axs[0].axis('off')
                # axs[0].set_title('Original')

                # axs[1].imshow(mask)
                # axs[1].axis('off')
                # axs[1].set_title('Clusters')
                # fig.suptitle(f"Image: {img_name}, Clusters: {K}")
                # outfig = os.path.join(outdir, img_name + ".eps")
                # tqdm.write("Saving to: {}".format(outfig))
                # plt.savefig(outfig, dpi=300)
                
        except OSError: raise # FileExistsErros is subclass of OSError



