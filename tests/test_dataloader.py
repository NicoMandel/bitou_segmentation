"""
    File to test the dataloading structure
"""

import pytest
import os.path
from csupl import dataloader

fdir = os.path.abspath(os.path.dirname(__file__))
root = os.path.join(fdir, "..", "data")
num_classes = 2

def test_dataset():
    ds = dataloader.BitouDataset(root = root, num_classes=num_classes)
    assert len(ds) > 0
    assert ds[0]

def test_datamodule():
    dm = dataloader.BitouDataModule(root = root, test_dir = "test")
    assert dm
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0

def test_mask():
    ds = dataloader.BitouDataset(root = root, num_classes = num_classes)
    _, mask = ds[0]
    assert mask.max() < num_classes
