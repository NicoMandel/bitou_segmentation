"""
    Following [this blog](https://cierra-andaur.medium.com/using-k-means-clustering-for-image-segmentation-fe86c3b39bf4)
    using help from [OpenCV docs](https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html)
    and [here](https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python)
    TODO: run the clusters on ALL images at once - reshape
    ! k means is unsupervised, KNN is supervised
    All data imported rom csupl
"""
from csupl.k_means import *
import argparse
# from sklearn.cluster import KMeans
import os.path
from os import mkdir
from pathlib import Path
from tqdm import tqdm

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
    parser.add_argument("-s", "--scale", help="scale for resizing the images in percentage. If None is given, will not resize", type=int, default=30)
    parser.add_argument("-e", "--epsilon", help="epsilon stopping criteria for the KMeans clustering algorithm. Defaults to 0.2", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, help="Iterations to run the algorithm. Defaults to 100", default=100)
    parser.add_argument("--overlay", help="Whether to decode colours. Defaults to false", default=False, action="store_true")
    parser.add_argument("--file-extension", help="Image file extension, with dot. Defaults to JPG", default=".JPG")
    parser.add_argument("-p", "--plot", help="Index to plot. If given, will not write directory out. 1-indexed!", default=0, type=int)
    parser.add_argument("--full", help="Taking the entire directory as a single data entry. Long Run duration with high K. \
                        Even longer with no Scale factor. Recommended: Scale 30, K 5. Default is False" , default=False, action="store_true")
    parser.add_argument("--hsv", default=False, action="store_true", help="If set, will convert to hsv colorspace before performing clustering")
    args = vars(parser.parse_args())
    return args

def run_full(img_list : str, img_dir : Path, scale : int, K : int, iterations : int, epsilon : float, plot_idx : int, overlay : bool, output_dir : str,
            f_ext : str, hsv : bool) -> None:
    """
        Function that will run the full directory
        ! add f_ext to the list of arguments
    """
    # get the image shape of the first image to create a big numpy array
    img_name = img_list[0]
    fname = img_dir / (img_name + f_ext)
    img = read_image(str(fname))
    if scale:
        img = resize_img(img, scale_perc=scale)       # resizes to 50% by default
    img_shape = img.shape
    img_shape = tuple([len(img_list)]) + img_shape
    batch_arr = np.zeros(img_shape)

    for i, img_name in tqdm(enumerate(img_list)):
        fname = img_dir / (img_name+f_ext)
        img = read_image(str(fname))
        if scale:
            img = resize_img(img, scale_perc=scale)
        if hsv:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # insert into big array
        batch_arr[i,...] = img

    # Actual clustering is happening here
    print("Starting Clustering. Lengthy Operation. Sit back and grab a coffee")
    mask, labels = cluster_img(batch_arr, K=K, iterations=iterations, epsilon=epsilon)
    labels = labels.reshape(batch_arr.shape[:-1])
    print("Finished Clustering. Have a look at the images being written next (with coffee in hand)")

    # Split whether plotting is necessary or not
    if plot_idx:
        idx = plot_idx-1

        img_name = img_list[idx]
        img = batch_arr[idx, ...]
        m = mask[idx, ...]
        label = labels[idx, ...]
        label = label.flatten()
        if overlay:
            if hsv: 
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
                mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            m = decode_colormap(m, label, K)
        plot_images(img, m, img_name, K)
        print("Test line for debugging")
    else:
        print("Writing entire directory {}, {} images".format(img_dir, len(img_list)))
        outdir_name = "K-{}_scale-{}_Overlay-{}-full".format(K, scale, overlay)
        output_parentdir = output_dir
        outdir = os.path.join(output_parentdir, outdir_name)
        print("Running Test case K: {}\tScale: {}\nSettings: Iterations {}\tEpsilon: {}\tOverlay Class: {}".format(
        K, scale, iterations, epsilon, overlay
            ))
        print("Writing to: {}".format(outdir))
        try:
            mkdir(outdir)
            for i, img_name in tqdm(enumerate(img_list)):
                img = batch_arr[i,...]
                m = mask[i,...]
                label = labels[i,...]
                label = label.flatten()
                if overlay:
                    if hsv: 
                        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
                        m = cv2.cvtColor(m, cv2.COLOR_HSV2BGR)
                    m = decode_colormap(m, label, K)
                
                outfig = os.path.join(outdir, img_name + ".jpg")
                tqdm.write("Saving to: {}".format(outfig))
                save_image(outfig, m)
                
        except OSError: raise # FileExistsErros is subclass of OSError

def run_single(img_list : str, img_dir : Path, scale : int, K : int, iterations : int, epsilon : float, plot_idx : int, overlay : bool, output_dir : str,
                f_ext : str, hsv : bool):
    """
        Function that will run a single file from the directory
    """
    if plot_idx:
        img_name = img_list[plot_idx-1]
        fname = img_dir / (img_name+f_ext)
        img = read_image(str(fname))
        if scale:
            img = resize_img(img, scale)
        
        if hsv:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # doing the clustering
        mask, labels = cluster_img(img, K=K, iterations=iterations, epsilon=epsilon)

        # whiteout the xths cluster
        if overlay:
            # mask = disable_cluster(mask, overlay, labels)
            if hsv:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
                mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            mask = decode_colormap(mask, labels, K)

        # Plot the images
        plot_images(img, mask, img_name, K)
        print("Test line for debugging")
    
    
    else:
        print("Reading entire directory {}, {} images".format(img_dir, len(img_list)))
        outdir_name = "K-{}_scale-{}_Overlay-{}".format(K, scale, overlay)
        output_parentdir = output_dir
        outdir = os.path.join(output_parentdir, outdir_name)
        print("Running Test case K: {}\tScale: {}\nSettings: Iterations {}\tEpsilon: {}\tOverlay Class: {}".format(
        K, scale, iterations, epsilon, overlay
            ))

        try:
            mkdir(outdir)
            # Section to read the entire directory
            for img_name in tqdm(img_list):
                fname = img_dir / (img_name+f_ext)
                img = read_image(str(fname))

                if scale:
                    img = resize_img(img, scale)

                if hsv:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # doing the clustering
                mask, labels = cluster_img(img, K=K, iterations=iterations, epsilon=epsilon)

                # whiteout the xths cluster
                if overlay:
                    # mask = disable_cluster(mask, overlay, labels)
                    if hsv: 
                        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
                        mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
                    mask = decode_colormap(mask, labels, K)

                # Save the image
                outfig = os.path.join(outdir, img_name + ".jpg")
                tqdm.write("Saving to: {}".format(outfig))
                save_image(outfig, mask)
                
        except OSError: raise # FileExistsErros is subclass of OSError


if __name__=="__main__":
    args = parse_args()

    # directory settings
    img_dir = Path(args["input"])
    f_ext = args["file_extension"]
    img_list = get_image_list(img_dir, f_ext=f_ext)
    outdir = args["output"]

    # Clustering settings
    K = args["K"]
    epsilon = args["epsilon"]
    iterations = args["iterations"]
    overlay = args["overlay"]
    scale = args["scale"]
    
    # Reading image
    plot_idx = args["plot"]
    
    # Whether to run full folder:
    full = args["full"]

    # Whether to convert into hsv colorspace before clustering
    hsv = args["hsv"]

    if full:
        run_full(img_list, img_dir, scale, K, iterations, epsilon, plot_idx, overlay, outdir, f_ext, hsv)
    else:
        run_single(img_list, img_dir, scale, K, iterations, epsilon, plot_idx, overlay, outdir, f_ext, hsv)

