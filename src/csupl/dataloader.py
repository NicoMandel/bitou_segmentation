"""
    TODO: Check in the VisionDataset class - it somehow allows for using separate transform and target_transforms
    Maybe post on [this forum](https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/62) for clarification?
"""

import torch
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
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
