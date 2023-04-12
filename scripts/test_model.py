"""
    Main File to test a model. Imports the model file, which itself imports:
        * the dataset from `dataloader.py`
     the transforms
    ! basic binary segmentation example using smp and albumentations: https://github.com/catalyst-team/catalyst/blob/v21.02rc0/examples/notebooks/segmentation-tutorial.ipynb
    ! tutorial from pytorch lightning themselves: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/04-inception-resnet-densenet.html?highlight=segmentation%20task
    

"""

import torch

# lightning
import pytorch_lightning as pl
from csupl.dataloader import BitouDataset, DataLoader
from csupl.model import Model

from train_model import get_test_transforms, get_shape

# Own imports
from argparse import ArgumentParser

# import transforms as tfs
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Logging
# from pytorch_lightning import loggers as pl_loggers

def parse_args():
    parser = ArgumentParser(description="Testing Loop for Semantic Segmentation model")
    # Model settings
    parser.add_argument("-m", "--model", type=str, help="Which model to choose. Specify path")
    # Model size settings
    parser.add_argument("--width", help="Width to be used for image preprocessing", default=512, type=int)
    parser.add_argument("--height", help="Height to be used for image preprocessing", default=512, type=int)
    
    # Dataloader settings
    parser.add_argument("-b", "--batch", type=int, default=None, help="batch size to be used. Should not exceed memory, depends on Network")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers to be used for dataloading. Default 4. Recommended: 4 * (num_gpus)")
    
    # Dataset Settings
    parser.add_argument("-i", "--input", help="Input Directory. Within this directory, will look for <images> and <masks> for training", type=str)    
    parser.add_argument("--mask-ext", type=str, help="Mask file extension to be read in the directory. Defaults to .png", default=".png")
    parser.add_argument("--image-ext", type=str, help="Image file extension to be read from the image directory. Defaults to .JPG", default=".JPG")
    args = parser.parse_args()
    return vars(args)

def default_args():
    argdict = {}
    argdict["classes"] = 3
    argdict["batch"] = 6
    argdict["workers"] = argdict["batch"] if argdict["batch"] < 12 else 12
    argdict["save"] = True
    argdict["dev_run"] = False
    argdict["model"] = "irrelevant"
    argdict["pretrained"] = "irrelevant"
    # argdict["limit"] = 1.0
    argdict["epochs"] = 30

    # height and width
    argdict["height"] = 512
    argdict["width"] = 512

    argdict["img_ext"] = ".JPG"
    argdict["mask_ext"] = ".png"
    return argdict

if __name__=="__main__":

    # model
    args = parse_args()
    model = Model.load_from_checkpoint(
        args["model"], 
        )

    # inherent size and other parameters
    preprocess_params = model.get_preprocessing_parameters()
    mean = tuple(preprocess_params['mean'])
    std = tuple(preprocess_params['std'])
    shape = get_shape(args["height"], args["width"])
    test_tfs = get_test_transforms(shape, mean, std)
    
    # Test directory - own dataloader, not datamodule
    ds = BitouDataset(args["input"], test_tfs, img_folder="images", mask_folder="labels",
                        img_ext=args["image_ext"], mask_ext=args["mask_ext"])
    dl = DataLoader(ds, batch_size=args["batch"], num_workers=args["workers"],pin_memory=True)
    
    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else None,
        devices=1,
    )
    
    # Actual test step
    trainer.test(model = model, dataloaders = dl)
