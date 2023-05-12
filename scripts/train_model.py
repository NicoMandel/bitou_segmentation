"""
    Main File. Imports the model file, which itself imports:
        * the dataset from `dataloader.py`
     the transforms
    ! basic binary segmentation example using smp and albumentations: https://github.com/catalyst-team/catalyst/blob/v21.02rc0/examples/notebooks/segmentation-tutorial.ipynb
    ! tutorial from pytorch lightning themselves: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/04-inception-resnet-densenet.html?highlight=segmentation%20task
    ! pet model for binary segmentation: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb

"""

import torch
from torch.utils.data import DataLoader
from csupl.model import Model
import segmentation_models_pytorch as sgm

# lightning
import pytorch_lightning as pl
from csupl.dataloader import BitouDataModule, TestDataModule

# loss fct
# from torch.nn import CrossEntropyLoss

# Own imports
import os.path
# from csupl.utils import onnx_export
# from csupl.train_utils import LogImages
from datetime import date, datetime
from argparse import ArgumentParser

# import transforms as tfs
import albumentations as A
from albumentations.pytorch import ToTensorV2
# import cv2 # for border_mode.CONSTANT in PadIfNeeded

# Logging
# from pytorch_lightning import loggers as pl_loggers
import yaml

def parse_args():
    parser = ArgumentParser(description="Training Loop for Semantic Segmentation model")
    # Model settings
    parser.add_argument("-c", "--classes", default=2, type=int, help="Number of classes in the dataset. Default is 2 - Binary case")
    parser.add_argument("-m", "--model", type=str, help="Which model to choose. Uses segmentation models pytorch", required=True)
    parser.add_argument("--encoder", type=str, help="Encoder name to be used with decoder architecture. Default is resnet34", default="resnet34")
    parser.add_argument("--weights", type=str, help="Encoder Weight pretraining to be used. Default is imagenet", default="imagenet")

    # Model size settings
    parser.add_argument("--width", help="Width to be used for training", default=512, type=int)
    parser.add_argument("--height", help="Height to be used during training", default=None, type=int)
    parser.add_argument("--max-size", help="Maximum image size encountered (along the smaller dimension). \
                        Defaults to None. If None, will not use spatial Pyramid", default=None, type=int)
    parser.add_argument("--halo", default=128, type=int, help="Halo to be used for model testing. Defaults to 128")

    # Training settings
    parser.add_argument("-b", "--batch", type=int, default=12, help="batch size to be used. Should not exceed memory, depends on Network")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers to be used for dataloading. Default 4. Recommended: 4 * (num_gpus)")
    parser.add_argument("-d", "--dev-run", action="store_true", default=False, help="If true, a fast development run is done with 1 batch for train, val and test")
    parser.add_argument("-l", "--limit", default=1.0, type=float, help="%% the training and validation batches to be used. Default is 1.0")
    parser.add_argument("-e", "--epochs", default=25, type=int, help="Maximum epochs, iterations of training. Default is 25")
    parser.add_argument("--val", type=float, help="Validation Percentage of the training dataset. Default is 0.25", default=0.25)
    parser.add_argument("--freeze", action="store_true", help="If set, will freeze the encoder during training")

    # Dataset Settings
    parser.add_argument("-i", "--input", help="Input Directory. Within this directory, will look for <images> and <masks> for training", type=str, required=True)
    parser.add_argument("--mask-ext", type=str, help="Mask file extension to be read in the directory. Defaults to .png", default=".png")
    parser.add_argument("--image-ext", type=str, help="Image file extension to be read from the image directory. Defaults to .JPG", default=".JPG")
    
    parser.add_argument("-o", "--output", type=str, help="Output location for the model to be stored, default is None, will not be stored!", default=None)
    args = parser.parse_args()
    return vars(args)

def get_model_name(model) -> str:
    """
        Function to get a name to store the model
    """
    now = datetime.now()
    tim = "{}:{}:{}".format(now.hour, now.minute, now.second)
    out_str = f"{str(model)}_{date.today()}-{tim}"
    return out_str

def get_model_export_path(directory, model, f_ext = ".pt"):
    modelname = get_model_name(model)
    return os.path.join(directory, modelname + f_ext)

def get_shape(height : int = None, width : int = None) -> tuple:
    if not (height or width): 
        return None
    elif height and not width:
        return (height, height)
    elif width and not height:
        return (width, width)
    else:
        return (height, width)

def get_training_transforms(max_size : int, shape : tuple, mean : tuple, std : tuple, p : float=0.5) -> A.Compose:
    if max_size is None:
        size_tf = A.OneOrOther(
            # A.RandomSizedCrop((shape[0], 10*shape[0]), shape[0], shape[1], shape[1]/shape[0]),
            A.RandomCrop(shape[0], shape[1], p=1),
            A.Resize(shape[0], shape[1], p=1),
            p=1)
    else:
        size_tf =  A.OneOf([
            A.RandomSizedCrop((shape[0], max_size), shape[0], shape[1], shape[1]/shape[0], p=1),
            A.RandomCrop(shape[0], shape[1], p=1),
            A.Resize(shape[0], shape[1], p=1)
            ], p=1)
    tf_list = [
            # Spatial Transforms
        size_tf,
        # A.RandomCrop(height, width, p=1),
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
        # A.PadIfNeeded(min_height = None, min_width= None, pad_height_divisor=32, pad_width_divisor=32, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.) if shape else None,
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True)        # 
    ]
    tfs = A.Compose([tf for tf in tf_list if tf is not None])
    return tfs

def get_test_transforms(mean : tuple, std : tuple) -> A.Compose:
    """
        Test Tranforms. Only:
            * Normalization
            * ToTensor
    """
    test_tfs = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True)
    ])
    return test_tfs


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

def log_experiment(model_name, resdir, **kwargs):
    res_yml = os.path.join(resdir, model_name + ".yaml")
    with open(res_yml, 'w') as f:
        yaml.dump(kwargs, f)    # potentially: default_flow_style=False

if __name__=="__main__":

    args = parse_args()

    # lightning - getting the model has to be done before loading the Data, because of the size dictated by the model
    in_channels = 3     # RGB Data
    if args["classes"] == 2:
        classes = 1     # reset for binary case
        loss_mode = sgm.losses.BINARY_MODE
    else:
        classes = args["classes"]
        loss_mode = sgm.losses.MULTICLASS_MODE

    # ! losses: https://smp.readthedocs.io/en/latest/losses.html
    # See paper - focal loss focusses on hard examples - so that these become weighted higher during training
    # loss = sgm.losses.SoftBCEWithLogitsLoss(smooth_factor=None) # consider replacing smooth factor with 0 or 1
    loss = sgm.losses.FocalLoss(loss_mode)
    # loss = sgm.losses.TverskyLoss(alpha=0.5, beta=0.5, mode=loss_mode)
    # Task parameters - depending on the training settings    
    lr = 1.0e-3
    weight_decay = 1.0e-4
    model = Model(args["model"], args["encoder"], args["weights"], in_channels, classes,      # model parameters
                loss=loss, lr = lr, weight_decay = weight_decay                           # task parameters
                )   

    # Pytorch transform parameters
    preprocess_params = model.get_preprocessing_parameters()
    mean = tuple(preprocess_params['mean'])
    std = tuple(preprocess_params['std'])

    # Transform probability
    p = 0.5
    shape = get_shape(args["height"], args["width"])
    train_aug = get_training_transforms(args["max_size"], shape, mean, std, p=p)
    test_transforms = get_test_transforms(mean=mean, std=std)

    # lightning - updated way to load the data - with a datamodule. Much simpler
    datamodule = BitouDataModule(
        root = args["input"],
        img_folder="images",
        mask_folder="labels",
        train_transforms=train_aug,
        batch_size=args["batch"], 
        num_workers=args["workers"],
        val_percentage=args["val"],
        img_ext=args["image_ext"],
        mask_ext=args["mask_ext"]
    )

    # Logger
    # logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    # tb_logger = pl_loggers.TensorBoardLogger(logdir, default_hp_metric=False, name="Albumentations-"+modelfname)
    # print("Logging to directory: {}".format(logdir))

    # Training
    # Alternative to limiting the training batches: https://pytorch-lightning.readthedocs.io/en/1.2.10/common/trainer.html#limit-train-batches
    trainer = pl.Trainer(
        # progress_bar_refresh_rate=5,
        # log_every_n_steps=2,
        max_epochs=args["epochs"],
        accelerator='gpu' if torch.cuda.is_available() else None,
        devices=torch.cuda.device_count(),
        # logger=tb_logger,
        fast_dev_run=args["dev_run"],
        # limit_train_batches=args["limit"],
        # limit_val_batches=args["limit"],
        # limit_predict_batches=1,
        # callbacks=[LogImages(10)]
        )

    print(80*"=")
    print("Training Settings:")
    print("Validation percentage: {}%%\tBatch size: {}\tEpochs:{}\t\
        Classes: {}".format(
        int(datamodule.val_percentage *100), args["batch"], args["epochs"], args["classes"]
    ))
    print("Cropping to ({})".format(shape)) if shape is not None else print("Using full sized images")
    print(80*"=")
    # actual training step
    if args["freeze"]:
        model.freeze_encoder()
    trainer.fit(model, datamodule=datamodule)

    # Loading the test datamodule
    test_dm = TestDataModule(
        root = args["input"],
        test_dir = "test",
        img_folder="images",
        mask_folder="labels",
        test_transforms=test_transforms,
        batch_size=args["batch"], 
        num_workers=args["workers"],
        img_ext=args["image_ext"],
        mask_ext=args["mask_ext"],
        halo = args["halo"],
        model_shape=shape,      #! recommendation is to do testing and inference on larger model shape,
        )

    # Test step
    trainer.test(model,datamodule=test_dm)

    # Exporting the model
    if args["output"] is not None:
        model.freeze()
        export_fpath = get_model_export_path(args["output"], model)
        trainer.save_checkpoint(export_fpath)
        settingsdict = {
            "loss" : loss,
            "accuracy" : model.accuracy,
            "weight decay": weight_decay,
            "learning rate" : lr,
            "batch size" : args["batch"],
            "workers" : args["workers"],
            "validation percentage" : args["val"],
            "augmentations" : train_aug,
            "classes" : classes,
            "epochs" : args["epochs"],
            "freeze backbone" : True if args["freeze"] else False,
            "image shape" : shape,
            "Max image size" : args["max_size"],
            "Halo": args["halo"]

            }
        modeln = get_model_name(model)
        log_experiment(modeln, args["output"], **settingsdict)
        print("Saved model to: {}. Experiment Setting in corresponging .yaml file".format(export_fpath))
