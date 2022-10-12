"""
    Definining the model according to a combination from:
        * Torchvision models, see [here](https://pytorch.org/vision/stable/models.html#semantic-segmentation)
    TODO:
        * UNet from pl lightning bolts, see [here](https://github.com/PyTorchLightning/lightning-bolts/blob/2ba3f62709b50604c29448f4ad620449796bce06/pl_bolts/models/vision/unet.py)
            * check this import: from pl_bolts.models.vision.unet import UNet
        * models from lightning flash - list is much longer in the registry   
"""

from tkinter import Y
from turtle import forward
import torch
import torch.nn as nn
import torchvision
from torchvision.models import segmentation as seg

import pytorch_lightning as pl
from typing import Any, Callable, Optional

class Model(pl.LightningModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def get_available_models(cls):
        return ["FCN_ResNet_50", "FCN_ResNet_101", "DeepLabV3_Mobilenet", "DeepLabV3_ResNet101", "DeepLabV3_ResNet50", "LRASPP_MobileNet"]


# Default models
class FCN(pl.LightningModule):

    av_backbones = ["resnet50", "resnet101"]
    def __init__(self, backbone : str, num_classes : int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if backbone.lower() not in self.av_backbones:
            raise NotImplementedError(f"Unknown backbone, must be in: {self.av_backbones}")

        if backbone.lower() == "resnet50":
            weights = seg.FCN_ResNet50_Weights.DEFAULT
            self._set_properties(weights, backbone)
            model = seg.fcn_resnet50(weights=weights)  # TODO: can change the backbone to be imageNetV2 here - follow the trail
        elif backbone.lower() == "resnet101":
            weights = seg.FCN_ResNet101_Weights.DEFAULT
            self._set_properties(weights, backbone)
            model = seg.fcn_resnet101(weights=weights)
        

        # setting the model to our own class numbers
        curr_head = model.classifier[-1]
        curr_head.out_channels = num_classes
        # model.classifier[-1] = torch.nn.Conv2d(curr_head.in_channels, num_classes, kernel_size=curr_head.kernel_size, stride=curr_head.stride)

        # fixing the backbone for training
        for param in model.backbone.parameters():
            param.requires_grad = False

        self.model = model


    def forward(self, x):
        y = self.model(x)["out"]
        return y

    # helper functions
    def _set_properties(self, weights, backbone):
        # model properties
        self.backbone = backbone
        self.model_name = weights.__class__
        self.pretrained = weights.name
        self.preprocessing = weights.transforms()


class DeepLab(pl.LightningModule):
    """
        May need to insert this in the __init__:
        if pretrained:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        # Changing the head
        self.model.classifier[-1] = torch.nn.Conv2d(256, self.num_classes, kernel_size=(1,1), stride=(1,1))
    """

    def __init__(self, num_classes=1, pretrained=False):
        super(DeepLab, self).__init__()

        # Number of classes + 'background'
        self.num_classes = num_classes

        # The actual model
        self.model = seg.deeplabv3_resnet101(pretrained=pretrained)
        # Freezing the first layers - according to https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#set-model-parameters-requires-grad-attribute
        
    # Lightning - resource [here](https://pytorch-lightning.readthedocs.io/en/1.2.10/advanced/transfer_learning.html#example-imagenet-computer-vision)
    # Alternative - Pytorch resource [here](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
    # Pytorch - differences on gradient setting here: https://stackoverflow.com/questions/51748138/pytorch-how-to-set-requires-grad-false
    def forward(self, x):
        x = self.model(x)['out']
        return x

