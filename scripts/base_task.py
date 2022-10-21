# The basic semantic segmentation as outlined in the pytorch flash documentation [here](https://lightning-flash.readthedocs.io/en/latest/reference/semantic_segmentation.html)

import numpy as np
import torch

from pytorch_lightning import seed_everything

import flash
# from flash.core.data.utils import download_data
from flash.image import SemanticSegmentation, SemanticSegmentationData

from PIL import Image
from predict import alternative_decode_colormap
import matplotlib.pyplot as plt

# 1. Create the DataModule
# The data was generated with the  CARLA self-driving simulator as part of the Kaggle Lyft Udacity Challenge.
# More info here: https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge
# download_data(
#     "https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip",
#     "./data",
# )
if __name__=="__main__":
    # fixing the seed for reproducibility
    seed_everything(42)

    datamodule = SemanticSegmentationData.from_folders(
        train_folder="data/bitou_test",
        train_target_folder="data/bitou_test_masks",
        val_split=0.2,
        transform_kwargs=dict(image_size=(512, 512)),
        num_classes=3,
        batch_size=1,   # MEMORY
        num_workers=1,
        pin_memory=True
    )
    head_name = "deeplabv3"
    backbone_name = "mobilenetv3_large_100"
    # 2. Build the task
    model = SemanticSegmentation(
        backbone=backbone_name,       #   mobilenetv3_large_100
        head=head_name,
        num_classes=datamodule.num_classes,
    )

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=30, gpus=torch.cuda.device_count())
    # trainer.finetune(model, datamodule=datamodule, strategy="freeze")  # strategy="no-freeze"

    # 4. Segment a few images!
    predict_files = [
        "data/bitou_test/DJI_20220404140149_0009.JPG",
        "data/bitou_test/DJI_20220404140422_0012.JPG",
        "data/bitou_test/DJI_20220404140702_0019.JPG"
    ]
    datamodule = SemanticSegmentationData.from_files(
        predict_files=predict_files,
        batch_size=1,
    )
    predictions = trainer.predict(model, datamodule=datamodule)
    print(predictions)
    for pred in predictions:
        pr = pred[0]
        img_in = pr['input'].detach().numpy()
        img_in = np.moveaxis(img_in,0,-1)
        img_path = pr['metadata']['filepath']
        img_name = img_path.split('/')[-1]

        p = pr['preds']
        # Plotting the images
        label = torch.argmax(p.squeeze(), dim=0).detach().numpy()
        dec_label = alternative_decode_colormap(img_in.shape, label, num_classes=21)  #! get the shape in here

        fig, axs = plt.subplots(1,2)    

        axs[0].imshow(img_in)
        axs[0].set_title("")  #! set the title to be the image and model name here
        axs[0].axis('off')

        axs[1].imshow(dec_label)
        axs[1].set_title("Mask")
        axs[1].axis('off')
        plt.suptitle(img_name+": " + head_name + " " + backbone_name)
        # plt.show()
        print("Test Debug Line")
        outpath = "results/tmp/Bitou/Untrained/" + img_name
        print("saving to: {}".format(outpath))
        plt.savefig(outpath, dpi=300)


        # Displaying the OG mask
        # maskf = mask_files[i]
        # mask = Image.open(maskf)
        # mask_arr = np.asarray(mask)[...,0]      # Get the red channel - where the labels are stored
        # mask_dec = decode_colormap(mask_arr, num_classes=21)
        # ax.imshow(mask_dec)
        # ax.set_title("Mask")
        # ax.axis('off')

        # ax.imshow(dec_lab)
        # ax.set_title("No squeeze")
        # ax.axis('off')
        # plt.show()
        # 5. Save the model!
        # trainer.save_checkpoint("results/tmp/bitou_3class_512_deeplabv3_mnetv3large_overfit_freeze.pt")