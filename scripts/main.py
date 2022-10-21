"""
    Main File. Imports the model file, which itself imports:
        * the dataset from `dataloader.py`
     the transforms
    ! basic binary segmentation example using smp and albumentations: https://github.com/catalyst-team/catalyst/blob/v21.02rc0/examples/notebooks/segmentation-tutorial.ipynb
"""

import torch
from csupl.model import Model
import segmentation_models_pytorch as sgm

# lightning
import pytorch_lightning as pl
from csupl.dataloader import BitouDataModule
from csupl.task import SegmentationTask

# loss fct
from torch.nn import CrossEntropyLoss

# Own imports
import os.path
# from csupl.utils import onnx_export
from csupl.train_utils import LogImages
from datetime import date, datetime
from argparse import ArgumentParser

# import transforms as tfs
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Logging
from pytorch_lightning import loggers as pl_loggers


def parse_args():
    parser = ArgumentParser(description="Training and Testing Loop for Semantic Segmentation model")
    
    parser.add_argument("-c", "--classes", default=1, type=int, help="Number of classes in the dataset (without background!). Default is 1")
    parser.add_argument("-b", "--batch", type=int, default=None, help="batch size to be used. Should not exceed memory, depends on Network")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of workers to be used for dataloading. Default 4. Recommended: 4 * (num_gpus)")
    parser.add_argument("-s", "--save", action="store_true", default=False, help="Whether the model should be exported. Default false.")
    parser.add_argument("-d", "--dev-run", action="store_true", default=False, help="If true, a fast development run is done with 1 batch for train, val and test")
    parser.add_argument("-m", "--model", default=1, type=int, help="Which model to choose. 1 for Deeplab, 2 for Unet")
    parser.add_argument("-p", "--pretrained", default=True, action="store_false", help="If set, model will NOT be pretrained. Note: only influences Deeplab")
    parser.add_argument("-l", "--limit", default=1.0, type=float, help="\% the training and validation batches to be used. Default is 1.0")
    parser.add_argument("-e", "--epochs", default=5, type=int, help="Maximum epochs, iterations of training")
    args = parser.parse_args()
    return vars(args)

if __name__=="__main__":

    args = parse_args()

    num_classes = args["classes"]

    # Training parameters  - depending on the hardware
    num_workers = args["workers"]
    batch_size = args["batch"]

    # For exporting
    export_model = args["save"]

    # Dataset parameters
    root_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..' , 'dataset')

    # lightning - getting the model has to be done before loading the Data, because of the size dictated by the model
    model_name = "FPN"
    encoder_name = "resnet34"
    encoder_weights = "imagenet"
    in_channels = 3     # RGB Data
    if num_classes == 2:
        classes = 1     # reset for binary case
        loss_mode = sgm.losses.BINARY_MODE
    else:
        classes = num_classes
        loss_mode = sgm.losses.MULTICLASS_MODE

    # ! losses: https://smp.readthedocs.io/en/latest/losses.html
    # ! logits version may be wrong here!
    loss = sgm.losses.SoftBCEWithLogitsLoss(smooth_factor=None) # consider replacing smooth factor with 0 or 1

    # Getting the actual model
    model = Model(model_name, in_channels, classes, encoder_name, encoder_weights)
    preprocess_params = model.get_preprocessing_parameters()

    # Task parameters - depending on the training settings
   
    
    lr = 1.0e-3
    weight_decay = 1.0e-4
    pretrained = args["pretrained"]

    # Transform probability
    p = 0.3
    # Pytorch transform parameters
    train_aug = A.Compose([
            # Spatial Transforms
        A.OneOf([
            A.RandomSizedCrop((height, 4*height), height, width, width/height, p=1),
            A.RandomCrop(height, width, p=1),
            A.Resize(height, width, p=1)
        ], p=1),
        A.OneOf([
            A.VerticalFlip(p=p),
            A.Rotate(limit=179, p=p),
            A.HorizontalFlip(p=p)
        ], p=p),
        # Color Transforms
        A.OneOf([ 
            # A.CLAHE(),
            A.RandomBrightnessContrast(p=p),
            A.RandomGamma(p=p),
            A.HueSaturationValue(p=p),
            A.GaussNoise(p=p)
        ], p=p),
        # Elastic Transforms
        A.OneOf([
            A.ElasticTransform(p=p),
            A.GridDistortion(p=p),
            A.OpticalDistortion(p=p),
        ], p=p),
        # Necessary Transforms        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_aug = A.Compose([
        A.RandomSizedCrop((height, 4*height), height, width, width/height, p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # lightning - updated way to load the data - with a datamodule. Much simpler
    datamodule = BitouDataModule(
        root_dir, "Test", num_classes,
        test_transforms=test_aug, train_transforms=train_aug,
        batch_size=batch_size, num_workers=num_workers
    )

    # Logger
    now = datetime.now()
    tim = "{}:{}:{}".format(now.hour, now.minute, now.second)
    modelfname = "{}-{}-{}".format(type(model).__name__, date.today(), tim)
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    tb_logger = pl_loggers.TensorBoardLogger(logdir, default_hp_metric=False, name="Albumentations-"+modelfname)
    print("Logging to directory: {}".format(logdir))

    # lightning - task
    task = SegmentationTask(model, loss, lr=lr, weight_decay=weight_decay, num_classes=classes)

    # Training
    # Alternative to limiting the training batches: https://pytorch-lightning.readthedocs.io/en/1.2.10/common/trainer.html#limit-train-batches
    trainer = pl.Trainer(
        progress_bar_refresh_rate=5,
        max_epochs=args["epochs"],
        gpus=1,
        logger=tb_logger,
        fast_dev_run=args["dev_run"],
        limit_train_batches=args["limit"],
        limit_val_batches=args["limit"],
        callbacks=[LogImages(10)]
        )
    trainer.fit(task, datamodule=datamodule)

    # Testing
    trainer.test(task, datamodule=datamodule)

    # exporting the model, importing it again and then running the test suite
    # Exporting the model
    if export_model:
        model.freeze()
        # onnx_export(model, modelfname, logdir, height=height, width=width)

    print("Done with the execution, Hooray! Please see Tensorboard in directory {} for more information".format(logdir))