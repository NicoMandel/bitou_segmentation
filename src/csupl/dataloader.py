"""
    TODO: Check in the VisionDataset class - it somehow allows for using separate transform and target_transforms
    Maybe post on [this forum](https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/62) for clarification?
"""

import torch
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
from PIL import Image
from torch.utils.data import random_split, DataLoader
from typing import Any, Callable, Optional
# For albumentations
import cv2
import numpy as np

import pytorch_lightning as pl

class BitouDataset(VisionDataset):

    def __init__(self, root: str,
            # num_classes: int,
            transforms: Optional[Callable] = None,
                img_folder : str ="bitou_test", mask_folder: str ="bitou_test_masks", f_ext : str = ".JPG") -> None:
        # directories
        super().__init__(root, transforms)
        self.img_dir = Path(self.root) / img_folder
        self.mask_dir = Path(self.root) / mask_folder
        
        # lists
        self.img_list = list([x.stem for x in self.img_dir.glob("*"+f_ext)])

        # number of classes
        # self.num_classes = num_classes
        self.f_ext = f_ext

    def __len__(self) -> int:
        return len(self.img_list)
    
    def __getitem__(self, idx: int) -> Any:
        if torch.torch.is_tensor(idx):
            idx = idx.tolist()
        
        fname = self.img_list[idx] + self.f_ext
        img_name = self.img_dir / fname
        mask_name = self.mask_dir / fname

        img = cv2.imread(str(img_name), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_name), cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask[...,0]
        mask = mask[..., np.newaxis]

        # ! Albumentations specific transform syntax!
        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        
        return img, mask

class BitouDataModule(pl.LightningDataModule):

    def __init__(self, root : str,
                # test_dir : str,
                num_workers : int = 1, batch_size : int =4, val_percentage : float = 0.25,
                img_folder : str = "bitou_test", mask_folder : str = "bitou_test_masks",
                train_transforms: Optional[Callable] = None,
                # test_transforms: Optional[Callable] = None
                ) -> None:
        super().__init__()
        
        # folder structure
        self.root_dir = root
        # self.test_dir = test_dir
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        
        # Training and loading parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percentage = val_percentage
        
        # Transforms
        self.train_transforms = train_transforms
        # self.test_transforms = test_transforms

    # TODO: change this from assigning states (self.x) because it will only be run on one process - use setup.
    # see [here](https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data)
    def prepare_data(self):
            """
                Same as the SegDataModule loader, but this one uses albumentations
            """

            self.default_dataset = BitouDataset(self.root_dir, self.train_transforms, img_folder=self.img_folder, mask_folder=self.mask_folder)
            
            # Splitting the dataset
            dataset_len = len(self.default_dataset)
            train_part = int( (1-self.val_percentage) * dataset_len)
            val_part = dataset_len - train_part

            # Actual datasets
            self.train_dataset, self.val_dataset = random_split(self.default_dataset, [train_part, val_part])
            
            # test dataset
            # testpath = Path(self.root_dir) / self.test_dir
            # self.test_dataset = BitouDataset(testpath, self.num_classes, self.test_transforms, image_folder=self.img_folder, mask_folder=self.mask_folder)

    # Dataloaders:
    def train_dataloader(self):
        dl = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        # pin_memory=True
        )
        return dl
    
    def val_dataloader(self):
        dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        # pin_memory=True
        )
        return dl
    
    # def test_dataloader(self):
        # dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        # return dl
