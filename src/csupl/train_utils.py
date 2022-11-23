"""
    Utility Class for the training pass. Used for logging inside the trainer through ''Callbacks'' interface.
    Built-in callbacks [here](https://pytorch-lightning.readthedocs.io/en/1.2.10/extensions/callbacks.html#built-in-callbacks)
"""

import torch
import pytorch_lightning as pl
import torchvision
from utils import decode_colormap, image_overlay
from torchvision.transforms import functional as F
from PIL import Image
from transforms import InverseNormalization


class InputMonitor(pl.Callback):
    """
        Example coming from a maintainer [here](https://medium.com/@adrian.waelchli/3-simple-tricks-that-will-change-the-way-you-debug-pytorch-5c940aa68b03) 
    
        Is potentially invalid, see [here](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#automatic-logging)
    """


    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        
        if(batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            logger = trainer.logger
            logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
            logger.experiment.add_histogram("target", y, global_step=trainer.global_step)


class LogImages(pl.Callback):

    def __init__(self, log_freq=10):
        super(LogImages, self).__init__()
        self.log_freq = log_freq
        self.IN = InverseNormalization()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        
        if(batch_idx + 1) % self.log_freq == 0:

            # logging the image to tensorboard
            tensorboard = trainer.logger.experiment
            grid = self.log_images(outputs)
            tensorboard.add_image("Epoch{}/Val/Batch{}/Image 1. Accuracy: {:.3f}. Order: Input, Ground Truth, Predicted".format(
                    trainer.current_epoch,
                    batch_idx,
                    outputs['val_acc'],
                ),
                grid, 0)
            
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        
        if (batch_idx + 1) % self.log_freq == 0:

            grid = self.log_images(outputs)
            tensorboard = trainer.logger.experiment
            tensorboard.add_image('Epoch{}/Test/Batch{}/Image 1. Accuracy: {:.3f}. Order: Input, Ground Truth, Predicted'.format(
                    trainer.current_epoch,
                    batch_idx,
                    1.0
                    # outputs['test_acc'],
                ),
                grid, 0)

    def log_images(self, outputdict):
        """
            Helper function to log images to the tensorboard
        """
        img_in = outputdict['in'][0]
        img_truth = outputdict['truth'][0]
        img_pred = outputdict['out'][0]
        img_in, img_truth = self.IN(img_in, img_truth)
        img_in = F.resize(img_in, (img_pred.shape[-2], img_pred.shape[-1]), interpolation=Image.NEAREST)
        # img_truth = F.resize(img_truth, img_in.shape[1], interpolation=Image.NEAREST)
        # decoding the colormap
        labels = torch.argmax(img_pred.squeeze(), dim=0).detach().cpu().numpy()
        truth_labels = img_truth.detach().cpu().numpy()

        decoded_pred = decode_colormap(labels, 2)
        decoded_truth = decode_colormap(truth_labels, 2)

        overlay = image_overlay(img_in, decoded_pred)

        grid = torchvision.utils.make_grid([img_in.cpu(), decoded_truth, overlay])
        # plot_triplet((img_in.cpu(), decoded_truth, decoded_pred))

        return grid

    def on_test_start(self, trainer, pl_module):
        """
            To Log the computational graph with Albumentations. 
        """
        try:
            tfs = trainer.datamodule.test_transforms
            for tf in tfs:
                if "crop" in type(tf).__name__.lower():
                    height = tf.height
                    width = tf.width
                    break
            input_array = torch.randn((1, 3, height, width)) 
            trainer.logger.log_graph(pl_module.model, input_array=input_array)
        except:
            pass