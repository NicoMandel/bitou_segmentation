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

# from segmentation_models_pytorch.datasets import OxfordPetDataset
from PIL import Image
import os.path

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

# Adopted from the OG OxfordPetDataset
class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "orig")
        self.masks_directory = os.path.join(self.root, "mask")

        perc = 0.25
        self.filenames = self._read_split(perc)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".JPG")
        mask_path = os.path.join(self.masks_directory, filename + ".JPG")

        image = np.array(Image.open(image_path).convert("RGB"))

        # have to reduce the channel here from 3 to 1
        trimap = np.array(Image.open(mask_path))[... , 0]
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self, perc):
        inv_perc = int(1. / perc)
        imdir = Path(self.images_directory)
        filenames = list([x.stem for x in imdir.glob("*.JPG")])
        # filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # % for train
            filenames = [x for i, x in enumerate(filenames) if i % inv_perc != 0]
        elif self.mode == "valid":  # % for validation
            filenames = [x for i, x in enumerate(filenames) if i % inv_perc == 0]
        return filenames

# adopted from the Simple OxfordPetDataset
class SimpleBitouPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.LINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample