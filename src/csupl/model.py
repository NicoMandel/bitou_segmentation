"""
    Definining the model according to a combination from:
        * Torchvision models, see [here](https://pytorch.org/vision/stable/models.html#semantic-segmentation)

    TODO:
        * UNet from pl lightning bolts, see [here](https://github.com/PyTorchLightning/lightning-bolts/blob/2ba3f62709b50604c29448f4ad620449796bce06/pl_bolts/models/vision/unet.py)
            * check this import: from pl_bolts.models.vision.unet import UNet
        * models from lightning flash - list is much longer in the registry   
"""

import pytorch_lightning as pl
from typing import Any

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

class Model(pl.LightningModule):

    def __init__(self, arch : str , in_channels : int, classes : int, encoder_name : str, encoder_weights : str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        try:
            self.model = smp.create_model(arch, encoder_name, encoder_weights, in_channels, classes, **kwargs)
            self.encoder_name = encoder_name
            self.encoder_weights = encoder_weights

        except KeyError: raise

    def forward(self, x):
        y = self.model.forward(x)
        return y

    def get_preprocessing_parameters(self):
        """
            Function to get the preprocessing inputs
        """
        return get_preprocessing_fn(self.encoder_name, self.encoder_weights)

    def get_available_models(self):
        raise NotImplementedError("Getting models from smp not yet implemented")

