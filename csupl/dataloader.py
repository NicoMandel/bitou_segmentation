"""
    TODO: Check in the VisionDataset class - it somehow allows for using separate transform and target_transforms
    Maybe post on [this forum](https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/62) for clarification?
"""

import torch
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
from PIL import Image
from torch.utils.data import random_split, DataLoader
from typing import Any, Callable, List, Optional, Tuple
# For albumentations
import cv2
import numpy as np

import pytorch_lightning as pl

class SegmentationDataset(VisionDataset):

    def __init__(self, root: str, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                image_folder : str ="bitou_test", mask_folder: str ="_masks") -> None:
        super().__init__(root, transforms, transform, target_transform)


class SegDataset(VisionDataset):
    """
        Dataset class, inherits from pytorch's vision datasets
    """

    def __init__(self, root_dir,  transforms=None, image_folder="images", mask_folder="labels"):

        super(SegDataset, self).__init__(root_dir, transforms=transforms)

        # Directory stuff
        self.root_dir = root_dir
        self.img_directory = Path(root_dir) / image_folder
        self.mask_directory = Path(root_dir) / mask_folder

        # fixed indexing issues - because lists in python are mutable
        self.img_list = list([x.stem for x in self.img_directory.glob("*.jpg")])
        self.img_suffix = "jpg"
        self.mask_suffix = "label.png"

    # pytorch - required
    def __len__(self):
        return len(self.img_list)

    # pytorch - required - for retrieving an element
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Robustify - create the mask filename from filename of the image 
        img_name = self.img_directory / (".".join((self.img_list[idx], self.img_suffix)))
        mask_name = self.mask_directory / ("_".join((self.img_list[idx], self.mask_suffix)))

        # pytorch works with PIL images
        img = Image.open(img_name)
        mask = Image.open(mask_name).convert('L')

        # resizing to appropriate size - comes from the imdb.py dataset file
        
        # pytorch transforms
        if self.transforms:
            img, mask = self.transforms(img, mask)
        
        return img, mask

# Lightning - suggested way of making the Data Agnostic to use: https://pytorch-lightning.readthedocs.io/en/1.2.10/extensions/datamodules.html#what-is-a-datamodule
class SegDataModule(pl.LightningDataModule):
    """
        Suggested way of making the data loading part of the model agnostic - obsoletes the 'abstract-model.py' file. See Documentation by lightning for usage:
        [documentation](https://pytorch-lightning.readthedocs.io/en/1.2.10/extensions/datamodules.html#what-is-a-datamodule)
        Makes use of the SegDataSet class above that is used above!
    """

    def __init__(self, root_dir, test_dir, num_workers=1, batch_size=4, val_percentage=0.25, train_transforms = None, test_transforms = None, img_folder="images", mask_folder="labels"):
        super(SegDataModule, self).__init__()
        
        # keeping all the default values
        # Folder structure
        self.root_dir = root_dir
        self.test_dir = test_dir
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        
        # Training and loading parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percentage = val_percentage

        # transforms
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    # lightning - done on one GPU
    def prepare_data(self):
        # Validation and training dataset
        self.default_dataset = SegDataset(self.root_dir, self.train_transforms, image_folder=self.img_folder, mask_folder=self.mask_folder)
        
        # Splitting the dataset
        dataset_len = len(self.default_dataset)
        train_part = int( (1-self.val_percentage) * dataset_len)
        val_part = dataset_len - train_part

        # Actual datasets
        self.train_dataset, self.val_dataset = random_split(self.default_dataset, [train_part, val_part])
        
        # test dataset
        testpath = Path(self.root_dir) / self.test_dir
        self.test_dataset = SegDataset(testpath, self.test_transforms, image_folder=self.img_folder, mask_folder=self.mask_folder)


    # lightning - done on each DDP
    # def setup(self, stage=None):
    #     if stage == "fit" or stage is None:
            

    # Dataloaders:
    def train_dataloader(self):
        dl = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        return dl
    
    def val_dataloader(self):
        dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        return dl
    
    def test_dataloader(self):
        dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        return dl
        

# Albumentations change to work with lightning
class AlbumentationsDataset(SegDataset):
    """
        init and len are the same as the SegDataset
    """
    def __init__(self, root_dir, num_classes, transforms=None, image_folder="images", mask_folder="labels"):
        super().__init__(root_dir, transforms=transforms, image_folder=image_folder, mask_folder=mask_folder)
        self.num_classes = num_classes
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        # Robustify - create the mask filename from filename of the image 
        img_name = self.img_directory / (".".join((self.img_list[idx], self.img_suffix)))
        mask_name = self.mask_directory / ("_".join((self.img_list[idx], self.mask_suffix)))

        img = cv2.imread(str(img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # mask = cv2.imread(str(mask_name), cv2.IMREAD_GRAYSCALE)
        # mask = mask.astype('float32')
        mask = np.array(Image.open(mask_name), dtype=np.int64)
        mask = self.preprocess_mask(mask)
        # Transforms
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        
        return img, mask

    def preprocess_mask(self, mask):
        """
            Attempt to fix indexing errors encountered in CUDA. Thanks @ Albumentations, OpenCV and Numpy. I just lost 2 days because of this!
        """
        # if np.any(np.greater(mask, 1.0)) or np.any(np.less(mask, 0.0)):
        #     idx = np.unravel_index(np.argmax(mask, axis=None), mask.shape)
        #     print("Mask of image {} outside of range at {}. Value: {}".format(img_idx, idx, mask[idx] ))
        #     # raise TypeError("Mask larger than 1 at: {}".format(idx))
        mask[mask < 0] = 0
        mask[mask > self.num_classes] = 0
        return mask

class AlbumentationsDataModule(SegDataModule):
    """
        init and everything is the same as the SegDataModule, only the `prepare_data` gets replaced so that the right kind of dataset is being loaded
    """

    def __init__(self, root_dir, test_dir, num_classes, num_workers=1, batch_size=4, val_percentage=0.25, train_transforms=None, test_transforms = None, img_folder="images", mask_folder="labels"):
        super().__init__(
        root_dir,
        test_dir, 
        num_workers=num_workers, 
        batch_size=batch_size, 
        val_percentage=val_percentage, 
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        img_folder=img_folder,
        mask_folder=mask_folder
        )
        self.num_classes = num_classes



    def prepare_data(self):
        """
            Same as the SegDataModule loader, but this one uses albumentations
        """

        self.default_dataset = AlbumentationsDataset(self.root_dir, self.num_classes, self.train_transforms, image_folder=self.img_folder, mask_folder=self.mask_folder)
        
        # Splitting the dataset
        dataset_len = len(self.default_dataset)
        train_part = int( (1-self.val_percentage) * dataset_len)
        val_part = dataset_len - train_part

        # Actual datasets
        self.train_dataset, self.val_dataset = random_split(self.default_dataset, [train_part, val_part])
        
        # test dataset
        testpath = Path(self.root_dir) / self.test_dir
        self.test_dataset = AlbumentationsDataset(testpath, self.num_classes, self.test_transforms, image_folder=self.img_folder, mask_folder=self.mask_folder)