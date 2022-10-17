"""
    File to test the dataloading structure
"""

import pytest
import os.path
import numpy as np
from tqdm import tqdm

from flash.image import SemanticSegmentationData

fdir = os.path.abspath(os.path.dirname(__file__))
root = os.path.join(fdir, "..", "data")
num_classes = 3
dm = SemanticSegmentationData.from_folders(
        train_folder= os.path.join(root, "bitou_test"),
        train_target_folder= os.path.join(root, "bitou_test_masks"),
        val_split=0.1,
        transform_kwargs=dict(image_size=(256, 256)),
        num_classes=2,
        batch_size=24,
        num_workers=12
    )
dm.prepare_data()

def test_dataset_length():
    assert len(dm.train_dataset) > 0
    
def test_dataset_loading():
    ds = dm.train_dataset
    # load first item
    first_img, first_mask = ds[0]
    # load last item
    last_img, last_mask = ds[-1]
    # Do type testing
    # assert isinstance(last_img)
    for img in [first_img, last_img, first_mask, last_mask]:
        assert isinstance(img, np.ndarray)

def test_datamodule():
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0

def test_mask_validity():
    ds = dm.train_dataset
    fct = 0
    cl_ct = 0
    for i in tqdm(range(len(ds))):
        mask = ds[i]['target']
        # count red channel
        r_max = mask.max()
        if r_max > 0:
            fct += 1
        
        # count if max is bigger than num_classes -1
        if r_max > (num_classes - 1):
            cl_ct += 1
    
    assert fct > 0
    assert cl_ct == 0