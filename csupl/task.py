from typing import Iterable
import torch
from torch import optim, nn

import pytorch_lightning as pl
import torchmetrics

import transforms as tfs

class SegmentationTask(pl.LightningModule):
    """
        A task, which is running the same steps on a multitude of different models. See the lightning Documentation [here](https://pytorch-lightning.readthedocs.io/en/1.2.10/common/lightning_module.html#inference-in-production) for further details
    """

    def __init__(self, model, loss, lr, weight_decay, num_classes):
        super(SegmentationTask, self).__init__()
        self.model = model
        self.num_classes = num_classes + 1

        # Task parameters
        self.loss = loss
        self.softm = nn.Softmax(dim=1)  # ! check if dimension 1 is the right dimension
        self.accuracy = torchmetrics.IoU(self.num_classes)

        self.lr = lr
        self.weight_decay = weight_decay

    # Optimizer
    def configure_optimizers(self):
        """
            default optimizer: also Adam, see [here](https://github.com/PyTorchLightning/lightning-bolts/blob/c3b60de7dc30c5f7947256479d9be3a042b8c182/pl_bolts/models/vision/segmentation.py#L77)
        """
        # ! Consider AdamW as an alternative optimiser
        optimiser = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimiser

    # Steps section
    def shared_step(self, batch):
        """
            A shared step, as defined [here](https://pytorch-lightning.readthedocs.io/en/1.2.10/common/lightning_module.html#inference) 
        """
        x, y = batch
        out = self.model(x)
        J = self.loss(out, y.long())
        return out, J

    def training_step(self, batch, batch_idx):
        _, J = self.shared_step(batch)
        self.log_dict({'loss/train': J}, prog_bar=True, logger=True, on_step=True)
        return {'loss': J}
    
    def validation_step(self, batch, batch_idx):
        out, J = self.shared_step(batch)
        x, y = batch
        # Accuracy
        pred = self.softm(out)
        acc = self.accuracy(pred, y)

        self.log_dict({'loss/val': J, 'acc/val': acc}, prog_bar=True, logger=True, on_epoch=True)
        return {'val_loss': J, 'val_acc': acc, 'in': x, 'truth': y, 'out': out}

    def test_step(self, batch, batch_idx):
        x, y = batch

        # similar to simple forward step
        out, J = self.shared_step(batch)
        # Accuracy
        pred = self.softm(out)
        acc = self.accuracy(pred, y)

        test_dict = {'loss/test': J, 'acc/test': acc}

        self.log_dict(test_dict, prog_bar=True, logger=True)
        return {'test_loss': J, 'test_acc': acc, 'out': out, 'in': x, 'truth': y}



class TestTask(pl.LightningModule):
    """
        A class only doing **tests** on network and logging the specified metrics. Not to be used for training
    """

    def __init__(self, model, metrics, num_classes=1):

        super(TestTask, self).__init__()

        # Model parameters
        self.model = model
        self.num_classes = num_classes + 1
        self.softm = nn.Softmax(dim=1)  
        self.metrics = metrics
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        pred = self.softm(out)
        metrics = self.metrics(pred, y) 
        self.log_dict(metrics, prog_bar=True, logger=True )

        return {"metrics": metrics, "out": out, "in": x, 'truth': y}
