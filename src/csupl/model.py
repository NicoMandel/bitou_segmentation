import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl


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