"""
    File to check the augmentations on the pipeline
    uses the default dataset
    https://albumentations.ai/docs/examples/replay/
    Replays the augmentations as provided in the train_model.py file
"""
import os
import random
from argparse import ArgumentParser

from csupl.utils import plot_overlaid, plot_images
from train_model import get_training_transforms
from csupl.dataloader import BitouDataset
from torch.utils.data import DataLoader

import albumentations as A

def parse_args():
    parser = ArgumentParser(description="File for visually inspecting the transforms that are used during training")
    parser.add_argument("-i", "--input", required=True, type=str, help="Location of the data folder. Will look for <images> and <masks> in folder")
    args = parser.parse_args()
    return vars(args)


if __name__=="__main__":
    random.seed(42)
    args = parse_args()
    print(f"Checking <images> and <masks> in folder: {args['input']}")

    ds = BitouDataset(args["input"], img_folder="images", mask_folder="masks", img_ext=".JPG", mask_ext=".png")
    dl = DataLoader(ds, batch_size=1, num_workers=1, pin_memory=True)

    # !missing 4 arguments: height, width, mean, std.
    tfs = get_training_transforms()

    transform = A.ReplayCompose(
        tfs
    )

    for batch in next(iter(dl)):
        img, mask = batch
        data = transform(image = img, mask = mask)
        plot_images(data['image'], data['mask'], "", None)




