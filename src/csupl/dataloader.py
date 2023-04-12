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
import numpy as np
import os.path

import pytorch_lightning as pl

from csupl.utils import load_image, load_label

class BitouDataset(VisionDataset):

    def __init__(self, root: str,
            # num_classes: int,
            transforms: Optional[Callable] = None,
                img_folder : str ="bitou_test", mask_folder: str ="bitou_test_masks", img_ext : str = ".JPG", mask_ext : str = ".png") -> None:
        # directories
        super().__init__(root, transforms)
        self.img_dir = Path(self.root) / img_folder
        self.mask_dir = Path(self.root) / mask_folder
        
        # lists
        self.img_list = list([x.stem for x in self.img_dir.glob("*"+img_ext)])

        # number of classes
        # self.num_classes = num_classes
        self.img_ext = img_ext
        self.mask_ext = mask_ext

    def __len__(self) -> int:
        return len(self.img_list)
    
    def __getitem__(self, idx: int) -> Any:
        if torch.torch.is_tensor(idx):
            idx = idx.tolist()
        
        fname = self.img_list[idx]
        img_name = self.img_dir / (fname + self.img_ext)
        mask_name = self.mask_dir / (fname + self.mask_ext)

        img = load_image(img_name)

        mask = load_label(mask_name)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        
        return img, mask

class BitouDataModule(pl.LightningDataModule):

    def __init__(self, root : str,
                test_dir : str,
                num_workers : int = 1, batch_size : int = 4, val_percentage : float = 0.25,
                img_folder : str = "bitou_test", mask_folder : str = "bitou_test_masks",
                train_transforms: Optional[Callable] = None,
                img_ext : str = ".JPG", mask_ext : str = ".png",
                test_transforms: Optional[Callable] = None
                ) -> None:
        super().__init__()
        
        # folder structure
        self.root_dir = root
        self.test_dir = test_dir
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        
        # Training and loading parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percentage = val_percentage
        
        # Transforms
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        # File extensions
        self.img_ext = img_ext
        self.mask_ext = mask_ext

    # TODO: change this from assigning states (self.x) because it will only be run on one process - use setup.
    # see [here](https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data)
    def prepare_data(self):
        """
            Same as the SegDataModule loader, but this one uses albumentations
        """

        self.default_dataset = BitouDataset(self.root_dir, self.train_transforms, img_folder=self.img_folder, mask_folder=self.mask_folder,
                                            img_ext=self.img_ext, mask_ext=self.mask_ext)
        
        # Splitting the dataset
        dataset_len = len(self.default_dataset)
        train_part = int( (1-self.val_percentage) * dataset_len)
        val_part = dataset_len - train_part

        # Actual datasets
        self.train_dataset, self.val_dataset = random_split(self.default_dataset, [train_part, val_part])
    
        # test dataset
        testpath = Path(self.root_dir) / self.test_dir
        self.test_dataset = BitouDataset(testpath, self.test_transforms, img_folder=self.img_folder, mask_folder=self.mask_folder,
                                    img_ext=".jpg", mask_ext=self.mask_ext)

    # Dataloaders:
    def train_dataloader(self):
        dl = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        pin_memory=True
        )
        return dl
    
    def val_dataloader(self):
        dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        pin_memory=True
        )
        return dl
    
    def test_dataloader(self):
        dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        pin_memory=True
        )
        return dl

#################################
# Secondary Datasets
#################################

# Adopted from the OG OxfordPetDataset
class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None, perc=0.25):

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "orig")
        self.masks_directory = os.path.join(self.root, "mask")

        self.perc = perc
        self.filenames = self._read_split(self.perc)

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