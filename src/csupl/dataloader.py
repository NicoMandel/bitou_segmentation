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
from tqdm import tqdm

import pytorch_lightning as pl

from csupl.utils import load_image, load_label, write_image, get_image_list
from csupl.propose_utils import get_window_dims, get_padding, get_padding_labels, get_tile_numbers, pad_image, get_stride

class BitouDataset(VisionDataset):

    def __init__(self, root: str,
            # num_classes: int,
            transforms: Optional[Callable] = None,
                img_folder : str ="images", mask_folder: str ="labels", img_ext : str = ".JPG", mask_ext : str = ".png") -> None:
        # directories
        super().__init__(root, transforms)
        self.img_dir = Path(self.root) / img_folder
        self.mask_dir = Path(self.root) / mask_folder
        
        # lists
        self.img_list = list([x.stem for x in self.img_dir.glob("*"+img_ext)])

        # number of classes
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
    """
        Datamodule to be used for training and validation.
        Because this works with small crops from the large images, it can run plenty of times.
        The test dataset will be evaulated on FULL images, which need to be cropped and windowed to make sense.
        This is defined in the Test Datamodule!
    """
    def __init__(self,
                root : str,
                num_workers : int = 1, batch_size : int = 4, val_percentage : float = 0.25,
                img_folder : str = "images", mask_folder : str = "labels",
                train_transforms: Optional[Callable] = None,
                img_ext : str = ".JPG", mask_ext : str = ".png",
                ) -> None:
        super().__init__()
        
        # folder structure
        self.root_dir = root
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        
        # Training and loading parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percentage = val_percentage
        
        # Transforms
        self.train_transforms = train_transforms

        # File extensions
        self.img_ext = img_ext
        self.mask_ext = mask_ext

    # change this from assigning states (self.x) because it will only be run on one process - use setup.
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

class TestDataModule(pl.LightningDataModule):

    def __init__(self,
                root : str,
                test_dir : str,
                model_shape : tuple,
                num_workers : int = 1, batch_size : int = 4,
                img_folder : str = "images", mask_folder : str = "labels",
                img_ext : str = ".JPG", mask_ext : str = ".png",
                test_transforms: Optional[Callable] = None,
                create_hidden : bool = False,
                halo : int =None,
                ) -> None:
        super().__init__()
        
        # folder structure
        self.root_dir = root
        self.test_dir = test_dir
        self.img_folder = img_folder
        self.mask_folder = mask_folder

        # Model parameters
        self.model_shape = model_shape
        self.stride = get_stride(model_shape)
        if halo is None:
            self.halo = int(model_shape[0] / 2)
        else:
            self.halo = halo
        self.window_shape = get_window_dims(model_shape, self.halo)
        
        # Training and loading parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Transforms
        self.test_transforms = test_transforms

        # File extensions
        self.img_ext = img_ext
        self.mask_ext = mask_ext

        # hidden creation
        self.create_hidden = create_hidden

    # change this from assigning states (self.x) because it will only be run on one process - use setup.
    # see [here](https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data)
    def prepare_data(self):
        """
            prepares the dataset by splitting it into smaller sets
        """
        if self.create_hidden:
            hidden_dir = self._create_hidden_dataset()
            testpath = Path(hidden_dir)
        else:
            testpath = Path(self.root_dir) / self.test_dir
        self.dataset = BitouDataset(testpath, self.test_transforms, img_folder=self.img_folder, mask_folder=self.mask_folder,
                                    img_ext=self.img_ext, mask_ext=self.mask_ext)

    def _create_hidden_dataset(self):
        """
            Function to create a hidden dataset of tiles.
        """
        # make a new test dataset, which is a hidden folder in the same directory
        import os
        hidden_dir = os.path.join(self.root_dir, "." + self.test_dir)
        if os.path.exists(hidden_dir):
            print("Path {} already exists! Check if the files have already been written".format(hidden_dir))
            return hidden_dir
        print("Creating new hidden dataset at {} \n This may take a while.".format(hidden_dir))
        os.mkdir(hidden_dir)
        # create the subfolders
        hidden_img_dir = os.path.join(hidden_dir, self.img_folder)
        hidden_mask_dir = os.path.join(hidden_dir, self.mask_folder)
        os.mkdir(hidden_img_dir)
        os.mkdir(hidden_mask_dir)

        # read in the old dataset
        test_dir = Path(self.root_dir) / self.test_dir
        imdir = Path(test_dir) / self.img_folder
        maskdir = Path(test_dir) / self.mask_folder
        imlist, _  = get_image_list(imdir)
        # go through all the images
        for imgname in tqdm(imlist, desc="Preparing Tiles"):
            # load image and mask
            img_f = os.path.join(str(imdir), imgname + self.img_ext)
            mask_f = os.path.join(str(maskdir), imgname + self.mask_ext)
            img = load_image(img_f)
            mask = load_label(mask_f)

            # pad accordingly
            imshape = img.shape[:-1]
            pad_left, pad_right, pad_top, pad_bottom = get_padding(imshape, self.model_shape, self.halo)
            pd_l_labels, pd_r_labels, pd_t_labels, pd_bt_labels = get_padding_labels(imshape, self.model_shape, self.halo)

            #! this is where the error is - pad the mask as well , otherwise it will be too small for the overhang. Then go from halo + i*stride
            n_tot, n_h = get_tile_numbers(imshape, self.model_shape)
            # out_imshape = get_out_shape(n_tot, n_h, self.model_shape)
            padded = pad_image(img, pad_top, pad_left, pad_right, pad_bottom)
            padded_labels = pad_image(mask, pd_t_labels, pd_l_labels, pd_r_labels, pd_bt_labels)

            # for the image, create image and subsequent labels to be evaluated against
            for k in tqdm(range(n_tot), desc="Window", leave=False):
                j = k // n_h
                i = k % n_h
                h_window = i * self.stride[0]
                w_window = j * self.stride[1]
                # from the image, take the large window
                window_im = padded[h_window : h_window + self.window_shape[0], w_window : w_window + self.window_shape[1]]
                # from the labels, take the inside window - the halo
                window_label = padded_labels[h_window : h_window + self.model_shape[0], w_window : w_window + self.model_shape[1]]
                # generate a new name
                out_name = "_".join([imgname,str(k)])
                write_image(hidden_img_dir, out_name, window_im, self.img_ext)
                write_image(hidden_mask_dir, out_name, window_label, self.mask_ext)

        return hidden_dir

    def test_dataloader(self):
        """
            Tests should be performed on large images, therefore the dataset needs to be rewritten.
        """
        dl = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        pin_memory=True
        )
        return dl

    # def teardown(self, stage: Optional[str] = None) -> None:
    #     """
    #         TODO: implement for end of "test" stage -> delete hidden dataset?
    #     """
    #     return super().teardown(stage)
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


        