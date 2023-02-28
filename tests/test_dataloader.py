"""
    File to test the dataloading structure
"""

import pytest
import os.path
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.plot import reshape_as_image

from csupl import dataloader
from csupl.utils import load_image


fdir = os.path.abspath(os.path.dirname(__file__))
root = os.path.join(fdir, "..", "data")
num_classes = 2

def check_channel(img : np.ndarray, channel_id : int, value : int = 255):
    for i in range(img.shape[2]):
        if i == channel_id:
            assert np.all(img[..., i] == value)
        else:
            assert np.all(img[..., i] == 0)
    
def check_channels(img: np.ndarray, channel_1 : int, channel_2 : int, value_1 : int = 128, value_2 : int = 128 /2):
    assert np.all(img[..., channel_1] == value_1)
    assert np.all(img[..., channel_2] == value_2)

def check_channel_mask(img : np.ndarray, mask : np.ndarray) -> bool:
    return np.all(img[..., int(mask.mean())] == 255)

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
    imgdir = os.path.join(fdir, "test_images")
    
    # r_jpg = load_image(os.path.join(imgdir, "r.jpg"))
    r_png = load_image(os.path.join(imgdir, "r.png"))
    # check_channel(r_jpg, 0)
    check_channel(r_png, 0)

    # g_jpg = load_image(os.path.join(imgdir, "g.jpg"))
    g_png = load_image(os.path.join(imgdir, "g.png"))
    # check_channel(g_jpg, 1)
    check_channel(g_png, 1)

    # b_jpg = load_image(os.path.join(imgdir, "b.jpg"))
    b_png = load_image(os.path.join(imgdir, "b.png"))
    # check_channel(b_jpg, 2)
    check_channel(b_png, 2)

    mix1 = load_image(os.path.join(imgdir, "mix1.png"))
    mix2 = load_image(os.path.join(imgdir, "mix2.png"))
    mix3 = load_image(os.path.join(imgdir, "mix3.png"))
    check_channels(mix1, 0, 1)
    check_channels(mix2, 1, 2)
    check_channels(mix3, 2, 0)

def test_tif_loading():
    imgdir = os.path.join(fdir, "test_images")

    rfile = os.path.join(imgdir, "r.tif")
    with rasterio.open(rfile, 'r') as src:
        img = src.read()
    check_channel(reshape_as_image(img), 0)

    gfile = os.path.join(imgdir, "g.tif")
    with rasterio.open(gfile, 'r') as src:
        img = src.read()
    check_channel(reshape_as_image(img), 1)
    
    bfile = os.path.join(imgdir, "b.tif")
    with rasterio.open(bfile, 'r') as src:
        img = src.read()
    check_channel(reshape_as_image(img), 2)

    # Mixes
    mix1file = os.path.join(imgdir, "mix1.tif")
    with rasterio.open(mix1file, 'r') as src:
        mix1 = src.read()
    check_channels(reshape_as_image(mix1), 0, 1)

    mix2file = os.path.join(imgdir, "mix2.tif")
    with rasterio.open(mix2file, 'r') as src:
        mix2 = src.read()
    check_channels(reshape_as_image(mix2), 1, 2)

    mix3file = os.path.join(imgdir, "mix3.tif")
    with rasterio.open(mix3file, 'r') as src:
        mix3 = src.read()
    check_channels(reshape_as_image(mix3), 2, 0)

def test_dataloader_channels():
    root = os.path.join(fdir, "test_images")
    ds = dataloader.BitouDataset(root = root, img_folder = "images", mask_folder = "labels", img_ext = ".png", mask_ext= ".png")
    for i, img_name in enumerate(ds.img_list):
        img, mask = ds[i]
        assert check_channel_mask(img, mask), "Image {} does not have 255 at specified channel. Channel via mask is: {}, Values are: {}".format(
            img_name, mask.mean(), img[..., int(mask.mean())]
        )

