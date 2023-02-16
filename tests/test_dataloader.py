"""
    File to test the dataloading structure
"""

import pytest
import os.path
import numpy as np
from tqdm import tqdm

from csupl import dataloader
from csupl.utils import load_image

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

def test_mask_validity():
    ds = dataloader.BitouDataset(root = root)
    fct = 0
    cl_ct = 0
    for i in tqdm(range(len(ds))):
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

def test_image_loading():
    imgdir = os.path.join(fdir, "..", "tmp")
    
    def test_channel(img : np.ndarray, channel_id : int):
        assert np.all(img[..., channel_id]) >= 250 
    
    def test_channels(img: np.ndarray, channel_1 : int, channel_2 : int):
        assert img[..., channel_1].all() == 128
        assert img[..., channel_2].all() == 128 / 2

    r_jpg = load_image(os.path.join(imgdir, "r.jpg"))
    r_png = load_image(os.path.join(imgdir, "r.png"))
    test_channel(r_jpg, 0)
    test_channel(r_png, 0)

    g_jpg = load_image(os.path.join(imgdir, "g.jpg"))
    g_png = load_image(os.path.join(imgdir, "g.png"))
    test_channel(g_jpg, 1)
    test_channel(g_png, 1)

    b_jpg = load_image(os.path.join(imgdir, "b.jpg"))
    b_png = load_image(os.path.join(imgdir, "b.png"))
    test_channel(b_jpg, 2)
    test_channel(b_png, 2)

    mix1 = load_image(os.path.join(imgdir, "mix1.png"))
    mix2 = load_image(os.path.join(imgdir, "mix2.png"))
    mix3 = load_image(os.path.join(imgdir, "mix3.png"))
    test_channels(mix1, 0, 1)
    test_channels(mix2, 1, 2)
    test_channels(mix3, 2, 0)