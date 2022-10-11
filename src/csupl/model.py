"""
    Definining the model according to a combination from:
        * res18_skip.py from network
        * the default lightning required format
    Uses the dataset defined in the dataloader
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import models

import pytorch_lightning as pl

# UNet implementation from pytorch lightning bolts - does not have a pretrained version
# https://github.com/PyTorchLightning/lightning-bolts/blob/2ba3f62709b50604c29448f4ad620449796bce06/pl_bolts/models/vision/unet.py
from pl_bolts.models.vision.unet import UNet

# lightning - import changes to LightningModule
class Resnet18Skip(pl.LightningModule):

    # pytorch - default definition
    def __init__(self, num_classes=1, pretrained=False):

        super(Resnet18Skip, self).__init__()

        # Number of classes + 'background'
        self.num_classes = num_classes + 1

        # Model definition
        res18 = models.resnet18(pretrained=pretrained)

        self.res_18_backbone = nn.Sequential(*list(res18.children())[:-6])
        self.conv2_x = nn.Sequential(*list(res18.children())[-6:-5])
        self.conv3_x = nn.Sequential(*list(res18.children())[-5:-4])
        self.conv4_x = nn.Sequential(*list(res18.children())[-4:-3])
        self.conv5_x = nn.Sequential(*list(res18.children())[-3:-2])

        self.top_conv = nn.Sequential(
            nn.Conv2d(in_channels=512 , out_channels=128, kernel_size=1),
            nn.ReLU()
            )
        self.lateral_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU()
        )
        self.lateral_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
            nn.ReLU(),
        )
        self.lateral_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.ReLU(),
        )
        self.segmentation_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, kernel_size=1)
        )

        # Freezing the backbone if pretrained is true
        if pretrained:
            for param in self.res_18_backbone.parameters():
                param.requires_grad = False

    # pytorch - default definition
    def forward(self, img):
        c1 = self.res_18_backbone(img)
        c2 = self.conv2_x(c1)
        c3 = self.conv3_x(c2)
        c4 = self.conv4_x(c3)
        c5 = self.conv5_x(c4)
        
        # Decoding step
        p5 = self.top_conv(c5)
        p4 = nn.UpsamplingBilinear2d(scale_factor=2)(p5) + self.lateral_conv1(c4)
        p3 = nn.UpsamplingBilinear2d(scale_factor=2)(p4) + self.lateral_conv2(c3)
        p2 = nn.UpsamplingBilinear2d(scale_factor=2)(p3) + self.lateral_conv3(c2)
        p2_2x = nn.UpsamplingBilinear2d(scale_factor=2)(p2)
        out = self.segmentation_conv(p2_2x)
        return out


class DeepLab(pl.LightningModule):

    def __init__(self, num_classes=1, pretrained=False):
        super(DeepLab, self).__init__()

        # Number of classes + 'background'
        self.num_classes = num_classes + 1

        # The actual model
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        # Freezing the first layers - according to https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#set-model-parameters-requires-grad-attribute
        if pretrained:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        # Changing the head
        self.model.classifier[-1] = torch.nn.Conv2d(256, self.num_classes, kernel_size=(1,1), stride=(1,1))

    # Lightning - resource [here](https://pytorch-lightning.readthedocs.io/en/1.2.10/advanced/transfer_learning.html#example-imagenet-computer-vision)
    # Alternative - Pytorch resource [here](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
    # Pytorch - differences on gradient setting here: https://stackoverflow.com/questions/51748138/pytorch-how-to-set-requires-grad-false
    def forward(self, x):
        x = self.model(x)['out']
        return x


# Not double inheritance from pl.LightningModule because the MRO will also run all other inits
class Unet(UNet, pl.LightningModule):

    def __init__(self, num_classes=1, input_channels=3, num_layers=5, features_start=64, bilinear=False):
        super(Unet, self).__init__(num_classes=num_classes+1, input_channels=input_channels, num_layers=num_layers, features_start=features_start, bilinear=bilinear)

    # def freeze(self):
    #     for param in self.parameters():
    #         param.requires_grad = False

    #     self.eval()


def getSize(model):
    """
        Utility function to get the size depending on the model
        Returns:
            * width
            * height
            * reduction
    """
    if isinstance(model, Resnet18Skip):
        return 256, 192, 2
    elif isinstance(model, DeepLab):
        return 480, 256, None
    elif isinstance(model, UNet):
        return 480, 256, None
    else:
        raise NotImplementedError("Unknown Model Class. Cannot determine size")


def getBatchSize(batch_size, model):
    """
        Function to return the (empirical)
    """
    if batch_size is not None:
        return batch_size
    elif isinstance(model, Unet):
        return 8
    elif isinstance(model, DeepLab):
        return 32
    elif isinstance(model, Resnet18Skip):
        return 32
    else:
        raise NotImplementedError("Model unknown")
        