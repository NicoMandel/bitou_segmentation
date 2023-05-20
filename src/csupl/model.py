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
                loss , lr : float, weight_decay : float,                                                               # task parameters
                *args: Any,
                halo : int = None,      # for inference
                **kwargs: Any) -> None:
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

        # halo for testing inference
        self.halo = halo

        # accuracy
        self.accuracy = torchmetrics.JaccardIndex(num_classes=self.classes)
        self.save_hyperparameters()

    def freeze_encoder(self) -> None:
        """
            Function to freeze the backbone (encoder) weights during training.
            Sets trainable parameters to false
        """
        for child in self.model.encoder.children():
            for param in child.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """
            Function to unfreeze the backbone (encoder) for training.
            Sets trainable parameters to true
        """
        for child in self.model.encoder.children():
            for param in child.parameters():
                param.requires_grad = True
    
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

    # Decoding the prediction. Model function
    def get_labels(self, pred : torch.Tensor, detach : bool = False, threshold : float = 0.5) -> torch.Tensor:
        """
            function to get the labels from the forward pass
            Works for multiclass and binary
            Flag can be set to call detach().cpu().numpy()
        """
        if self.classes == 2:
            labels = self._get_label_binary(pred, threshold=threshold)
        else:
            labels = self._get_label_multiclass(pred)
        if detach:
            labels = labels.detach().cpu().numpy()
        return labels

    def _get_label_binary(self, pred : torch.Tensor, threshold : float = 0.5) -> torch.Tensor:
        """
            Function to return the binary label from a prediction
        """
        prob = pred.sigmoid()
        cl = (prob > threshold).float()
        return cl.squeeze()
    
    def _get_label_multiclass(self, pred : torch.Tensor) -> torch.Tensor:
        """
            function to return the multiclass label from a prediction
        """
        return torch.argmax(pred, dim=1).squeeze()

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
        # out = out.sigmoid()
        # J = self.loss(out, y.float())  
        J = self.loss(out, y.long())
        self.log_dict({'loss/train': J}, prog_bar=True, logger=True, on_step=True)
        return {'loss': J}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self._shared_step(x)
        # Accuracy
        pred_cl = self.get_labels(pred)
        acc = self.accuracy(pred_cl, y)    

        self.log_dict({'acc/val': acc}, prog_bar=True, logger=True, on_epoch=True)
        return {'val_acc': acc, 'in': x, 'truth': y, 'out': pred}

    def test_step(self, batch, batch_idx):
        x, y = batch
        # similar to simple forward step
        pred = self._shared_step(x)
        pred_cl = self.get_labels(pred)

        # Accuracy
        #! check if this can be rewritten
        if self.halo is not None:
            pred_cl_edit =  pred_cl[..., self.halo : -self.halo, self.halo: -self.halo]
            acc = self.accuracy(pred_cl_edit, y)
        else:
            acc = self.accuracy(pred_cl, y)

        self.log_dict({'acc/test': acc}, prog_bar=True, logger=True)
        return {'test_acc': acc, 'out': pred, 'in': x, 'truth': y}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, _ = batch
        out = self._shared_step(x)
        return out

    def __repr__(self):
        try:
            repr = f"{self.model.name}-{self.encoder_weights}_{self.classes}"
        except AttributeError:
            repr = f"{self.model._get_name()}-{self.encoder_weights}_{self.classes}"
        return repr
    
##############################################
# Secondary Model
##############################################
    
class PetModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs) -> None:
        super().__init__()

        self.model = smp.create_model(arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs)

        # preprocessing parameters
        params = smp.encoders.get_preprocessing_params(encoder_name)
        std = torch.tensor(params["std"])
        mean = torch.tensor(params["mean"])
        self.register_buffer("std", std.view(1,3,1,1))
        self.register_buffer("mean", mean.view(1, 3, 1, 1))

        # Loss
        # explanation [here](https://medium.com/ai-salon/understanding-dice-loss-for-crisp-boundary-detection-bb30c2e5f62b)
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        
        # for saving the model
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)        #! changed LR from the original value of 0.0001

    def forward(self, image):
        # image normalization
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask
    
    def shared_step(self, batch, stage):
        image = batch["image"]
        mask = batch["mask"]
        
        # perform controls
        self._check_image(image)
        self._check_mask(mask)

        # actual pass
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        # converting some metrics
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # compute metrics - see original post for details
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
    
    # epoch end - aggregate fucking things
    def shared_epoch_end(self, outputs, stage):
        # aggregate metrics
        tp = torch.cat([x["tp"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")      # TODO: see what reduction does

        # aggregated over everything
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou" : per_image_iou,
            f"{stage}_dataset_iou" : dataset_iou,
        }
        self.log_dict(metrics, prog_bar=True)

    # training
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    # validation
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    # testing
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    # # prediction
    def predict_step(self, batch, batch_idx, dataloader_idx : int =0):
        logits = self(batch['image'])
        return logits

    # helper functions
    def _check_image(self, image):
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

    def _check_mask(self, mask):
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0
