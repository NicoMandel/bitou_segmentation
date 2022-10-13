"""
    File to test the dataloading structure
"""

import pytest
import os.path
import numpy as np
import cv2

from csupl import dataloader

fdir = os.path.abspath(os.path.dirname(__file__))
root = os.path.join(fdir, "..", "data")
num_classes = 2

def test_dataset_length():
    ds = dataloader.BitouDataset(root = root, img_folder = "bitou_test", mask_folder = "bitou_test_masks", f_ext = ".JPG")
    assert len(ds) > 0
    
def test_dataset_loading():
    ds = dataloader.BitouDataset(root = root, img_folder = "bitou_test", mask_folder = "bitou_test_masks", f_ext = ".JPG")
    # load first item
    first_img, first_mask = ds[0]
    # load last item
    last_img, last_mask = ds[-1]
    # Do type testing
    # assert isinstance(last_img)
    for img in [first_img, last_img, first_mask, last_mask]:
        assert isinstance(img, np.ndarray)

def test_datamodule():
    dm = dataloader.BitouDataModule(root = root)
    dm.prepare_data()
    assert dm
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0

def test_mask_length():
    ds = dataloader.BitouDataset(root = root)
    _, mask = ds[0]
    assert mask.max() < num_classes

def test_mask_validity():
    ds = dataloader.BitouDataset(root = root)
    fct = 0
    cl_ct = 0
    for i in range(len(ds)):
        _, mask = ds[i]

        # count red channel
        r_max = mask[..., 0].max()
        if r_max > 0:
            fct += 1
        
        # count if max is bigger than num_classes -1
        if r_max > (num_classes - 1):
            cl_ct += 1
    
    assert fct > 0
    assert cl_ct == 0