"""
    Following [this blog](https://cierra-andaur.medium.com/using-k-means-clustering-for-image-segmentation-fe86c3b39bf4)
    using help from [OpenCV docs](https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html)
    and [here](https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python)
    TODO: run the clusters on ALL images at once - reshape
    ! k means is unsupervised, KNN is supervised
    All data imported rom csupl
"""
import argparse
# from sklearn.cluster import KMeans
import os.path
from os import mkdir
from pathlib import Path
from tqdm import tqdm

from csupl.k_means import *
from csupl.utils import read_image, plot_images, save_image, get_image_list

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
    parser.add_argument("--overlay", help="Whether to decode colours. Defaults to false", default=False, action="store_true")
    parser.add_argument("--file-extension", help="Image file extension, with dot. Defaults to JPG", default=".JPG")
    parser.add_argument("-p", "--plot", help="Index to plot. If given, will not write directory out. 1-indexed!", default=0, type=int)
    parser.add_argument("-c", "--config", help="Path to load the classifier from", default=None, type=str)
    parser.add_argument("-t", "--tolerance", help="Tolerance. Used to define a tolerance for the prediction - will calculate masks using distances of clusters \
                        defaults to None. If None given, labels will be as proposed. Try 1.0 first", type=float, default=None)
    args = vars(parser.parse_args())
    return args

def predict_files(img_list : str, img_dir : Path, plot_idx : int, overlay : bool, output_dir : str, 
                f_ext : str, classif_path : str, tolerance : float):
    """
        function to load a classifier and predict 
    """
    classif = km_algo.load_classifier(classif_path)

    if plot_idx:
        img_name = img_list[plot_idx-1]
        print("Predicting on image: {}".format(
            img_name
        ))
        fname = img_dir / (img_name+f_ext)
        img = read_image(str(fname))
        mask, _ = classif(img, overlay)
        if tolerance:
            nmask, nlabel = classif.calculate_distance(img, tol = tolerance)
            mask = classif._postprocess_mask(nmask, nlabel, overlay, classes = classif.K+1)
        plot_images(img, mask, img_name, classif.K)
    else:
        print("Predicting on entire directory: {}\t{} files\nUsing classifier: {}".format(
            img_dir, len(img_list), classif_path
        ))
        print("Classifier settings: {} classes\t{}%% scale,\tColorspace: {}\tOverlay: {}".format(
            classif.K, classif.scale, "hsv" if classif.hsv else "rgb", overlay
        ))

        output_dir += "_{}".format(tolerance) if tolerance is not None else ""
        outdir_name = os.path.basename(classif_path).split(".")[0]
        outdir = os.path.join(output_dir, outdir_name)
        print("Writing out to directory: {}".format(outdir))
        try:
            mkdir(outdir)
            # Section to read the entire directory
            for img_name in tqdm(img_list):
                # input processing
                fname = img_dir / (img_name+f_ext)
                img = read_image(str(fname))

                # prediction
                mask, _ = classif(img, overlay)
                if tolerance:
                    nmask, nlabel = classif.calculate_distance(img, tol = tolerance)
                    mask = classif._postprocess_mask(nmask, nlabel, overlay, classes = classif.K+1)

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
    img_list, f_ext = get_image_list(img_dir, f_ext=f_ext)
    outdir = args["output"]

    overlay = args["overlay"]
    
    # Reading image
    plot_idx = args["plot"]

    # Tolerance
    tolerance = args["tolerance"]

    # TODO: turn into Training and prediction part
    cpath = args["config"]
    predict_files(
        img_list=img_list,
        img_dir=img_dir,
        plot_idx=plot_idx,
        output_dir=outdir,
        f_ext=f_ext,
        overlay=overlay,
        classif_path=cpath,
        tolerance=tolerance
        )
