"""
    File to check the augmentations on the pipeline
    uses the default dataset
    https://albumentations.ai/docs/examples/replay/
    Replays the augmentations as provided in the train_model.py file
"""
import os
import random
import numpy as np
from argparse import ArgumentParser

from csupl.utils import plot_overlaid, plot_images, plot_three, get_colour_decoder
from train_model import get_training_transforms
from csupl.dataloader import BitouDataset
from torch.utils.data import DataLoader
from csupl.model import Model

import albumentations as A
from albumentations.pytorch import ToTensorV2


def parse_args():
    parser = ArgumentParser(description="File for visually inspecting the transforms that are used during training")
    parser.add_argument("-i", "--input", required=True, type=str, help="Location of the data folder. Will look for <images> and <masks> in folder")
    parser.add_argument("-m", "--model", action="store_true", help="Whether to include a model forward pass. Will always use the <best> model in the <results> directory")
    args = parser.parse_args()
    return vars(args)


if __name__=="__main__":
    # TODO: use the model for a forward pass here
    random.seed(42)
    args = parse_args()
    print(f"Checking <images> and <masks> in folder: {args['input']}")

    ds = BitouDataset(args["input"], img_folder="images", mask_folder="masks", img_ext=".JPG", mask_ext=".png")
    dl = DataLoader(ds, batch_size=1, num_workers=1, pin_memory=True, shuffle=True)

    # !missing 4 arguments: height, width, mean, std.
    fdir = os.path.abspath(os.path.dirname(__file__))
    resdir = os.path.join(fdir, '..', 'results')
    model_f = os.path.join(resdir, 'best')
    model = Model.load_from_checkpoint(model_f)
    preprocess_params = model.get_preprocessing_parameters()
    mean = tuple(preprocess_params['mean'])
    std = tuple(preprocess_params['std'])
    height = 512
    width = 512
    tr_tfs = get_training_transforms(height=height, width=width, mean = mean, std = std)
    tf_list = tr_tfs.transforms

    # Removing normalization and "To Tensor" -> these are not the visible ones
    norm_name = type(A.Normalize())
    totens_name = type(ToTensorV2())
    tfs = []
    excluded_tfs = [norm_name, totens_name]
    model_tfs = []
    for tf in tf_list:
        if type(tf) not in excluded_tfs:
            tfs.append(tf)
        else:
            model_tfs.append(tf)
    
    model_tf = A.Compose(model_tfs)
    transform = A.ReplayCompose(
        tfs
    )

    if args['model']:
        cdec = get_colour_decoder()

    for test_img, test_label in dl:
        # img, mask = batch
        data = transform(image = test_img.squeeze().numpy(), mask = test_label.squeeze().numpy())
        # img = data['image'].permute(1,2,0).numpy()
        # img = np.moveaxis(data['image'], 0, -1)
        img = data['image']
        mask = data['mask']
        if args["model"]:
            img_in = model_tf(image=img)['image']
            pred = model(img_in.unsqueeze(0))
            labels = model.get_labels(pred, detach=True)
            if model.classes != 2:
                labels_dec = cdec(labels)
            else:
                labels_dec = labels
            plot_three(img, mask, labels_dec, "random")
        else:
            plot_images(img, mask, "", None)




