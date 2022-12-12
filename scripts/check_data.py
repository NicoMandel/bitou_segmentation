"""
    File to check the integrity of the dataset.
    TODO: add logging functionality to file - for persistent checking: https://docs.python.org/3/howto/logging.html
    https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
    https://stackoverflow.com/questions/40858658/python-logging-to-stdout-and-log-file
"""

import os
from argparse import ArgumentParser
from csupl.utils import get_image_list

def parse_args():
    parser = ArgumentParser(description="File for checking the integrity of a data folder.")
    parser.add_argument("-i", "--input", type=str, help="Location of the data folder. Will look for <images> and <masks> in folder", required=True)
    parser.add_argument("-c", "--classes", required=True, type=int, help="How many classes should be in the <labels> folder.")
    parser.add_argument("-e", "--extended", help="Whether to run extended check", action="store_true")
    args = parser.parse_args()
    return vars(args)

def get_file_extensions(dir : str) -> list:
    f_exts = [os.path.splitext(f)[1] for f in os.listdir(dir)]
    return set(f_exts)

def check_exist(path : str):
    assert os.path.exists(path), f"{path} does not exist."
    print(f"{path} exists")
    print(f"Full location: {os.path.realpath(path)}")

def check_overlap(set1 : set, set2: set):
    over = set1 & set2
    assert over, f"No overlap between {set1} and {set2}"
    print(f"Overlap between sets: {over}")
    return over

if __name__=="__main__":
    args = parse_args()
    fullpath = os.path.abspath(args["input"])
    label_folder_name = "masks"
    image_folder_name = "images"
    print("Checking folder: {}".format(fullpath))
    
    # ensure that there are folders <images> and <masks>
    label_folder = os.path.join(fullpath, label_folder_name)
    check_exist(label_folder)
    image_folder = os.path.join(fullpath, image_folder_name)
    check_exist(image_folder)

    # check file extensions
    valid_extensions = set([".jpg", ".png", ".JPG"])
    print(f"Valid file extensions are: {valid_extensions}")
    ext_imgs = get_file_extensions(image_folder)
    ext_labels = get_file_extensions(label_folder)
    ov_imgs = check_overlap(valid_extensions, ext_imgs)
    assert len(ov_imgs) == 1, f"{image_folder} has more than 1 image type extension: {ov_imgs}, check consistency"
    print(f"Overlap between image folder and valid extensions: {ov_imgs}")
    ov_labels = check_overlap(valid_extensions, ext_labels)
    assert len(ov_labels) == 1, f"{label_folder} has more than 1 image type extension: {ov_labels}, check consistency"
    print(f"Overlap between mask folder and valid extensions: {ov_labels}")

    label_ext = str(tuple(ov_labels)[0])
    img_ext = str(tuple(ov_imgs)[0])
    print(f"Proceeding with image extension: {img_ext}")
    print(f"Proceeding with mask extension: {label_ext}")
    # ensure that there are as many images in <images> as in <data>
    img_list = get_image_list(image_folder, img_ext)
    label_list = get_image_list(label_folder, label_ext)
    assert len(img_list) == len(label_list), f"{len(img_list)} images and {len(label_list)} masks. Does not match"
    print(f"{len(img_list)} images and {len(label_list)} masks found. Proceeding")
    # check that every image in <labels> has a name correspondence in <images>
    for img_f in img_list:
        lpath = os.path.join(label_folder, img_f+label_ext)
        assert os.path.exists(lpath), f"{lpath} does not exist"
    print("Every image has a corresponding mask. Proceeding")
    # check that every image in <images> has a correspondence in <labels>
    for label_f in label_list:
        ipath = os.path.join(image_folder, label_f+img_ext)
        assert os.path.exists(ipath), f"{ipath} does not exist"
    print("Every mask has a corresponding image. Proceeding")
    # Check for classes

    # extended functionality:
    if args["extended"]:
        pass
        # find out class statistics. Count over the entire dataset

        # find the maximum value in <labels>

        # find the minimum value in <labels>


        # find out approximate mean and std. deviation over the dataset
        # use batches for this

