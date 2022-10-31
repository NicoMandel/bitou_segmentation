"""
    Definining the model according to qbvel's [smp package](https://smp.readthedocs.io/en/latest/index.html), [github](https://github.com/qubvel/segmentation_models.pytorch)

    TODO - both from lightning flash?
        * Saving a model
        * freezing the backbone
"""
import torch
import pytorch_lightning as pl
from typing import Any

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_params
import torchmetrics
from torch import optim

class Model(pl.LightningModule):

    def __init__(self, arch : str , encoder_name : str, encoder_weights : str,  in_channels : int, classes : int,       # model parameters
                loss , lr : float, weight_decay : float,                                                                # task parameters
                *args: Any, **kwargs: Any) -> None:
        super().__init__()

        # model-specific components
        try:
            self.model = smp.create_model(arch, encoder_name, encoder_weights, in_channels, classes, **kwargs)
            self.encoder_name = encoder_name
            self.encoder_weights = encoder_weights
            
        except KeyError: raise

        # Task-specific components - for the JaccardIndex to work
        self.classes = classes + 1 if classes == 1 else classes
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay

        # accuracy
        self.accuracy = torchmetrics.JaccardIndex(num_classes=self.classes)
        self.save_hyperparameters()
    
    # Model-specific steps
    def forward(self, x):
        y = self.model.forward(x)
        return y

    def get_preprocessing_parameters(self):
        """
            Function to get the preprocessing inputs
        """
        prepr = get_preprocessing_params(self.encoder_name)         # get_preprocessing_fn(self.encoder_name, self.encoder_weights)
        return prepr

    def get_available_models(self):
        raise NotImplementedError("Getting models from smp not yet implemented")


    # Lightning steps - replacing the task
    def configure_optimizers(self):
        optimiser = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimiser

    # Steps section
    def _shared_step(self, x):
        """
            A shared step, as defined [here](https://pytorch-lightning.readthedocs.io/en/1.2.10/common/lightning_module.html#inference) 
        """
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self._shared_step(x)
        J = self.loss(out, y.squeeze().long())  
        self.log_dict({'loss/train': J}, prog_bar=True, logger=True, on_step=True)
        return {'loss': J}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self._shared_step(x)
        # Accuracy
        acc = self.accuracy(pred, y)    

        self.log_dict({'acc/val': acc}, prog_bar=True, logger=True, on_epoch=True)
        return {'val_acc': acc, 'in': x, 'truth': y, 'out': pred}

    def test_step(self, batch, batch_idx):
        x, y = batch
        # similar to simple forward step
        pred = self._shared_step(x)
        # Accuracy
        acc = self.accuracy(pred, y)

        self.log_dict({'acc/test': acc}, prog_bar=True, logger=True)
        return {'test_acc': acc, 'out': pred, 'in': x, 'truth': y}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, _ = batch
        out = self._shared_step(x)
        return out

