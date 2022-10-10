# The basic semantic segmentation as outlined in the pytorch flash documentation [here](https://lightning-flash.readthedocs.io/en/latest/reference/semantic_segmentation.html)

import torch

from pytorch_lightning import seed_everything

import flash
# from flash.core.data.utils import download_data
from flash.image import SemanticSegmentation, SemanticSegmentationData

from PIL import Image

# 1. Create the DataModule
# The data was generated with the  CARLA self-driving simulator as part of the Kaggle Lyft Udacity Challenge.
# More info here: https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge
# download_data(
#     "https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip",
#     "./data",
# )

# fixing the seed for reproducibility
seed_everything(42)

datamodule = SemanticSegmentationData.from_folders(
    train_folder="data/bitou_test",
    train_target_folder="data/bitou_test_masks",
    val_split=0.3,
    transform_kwargs=dict(image_size=(256, 256)),
    num_classes=2,
    batch_size=12,
    num_workers=12
)

# 2. Build the task
model = SemanticSegmentation(
    backbone="mobilenetv3_large_100",
    head="fpn",
    num_classes=datamodule.num_classes,
)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=10, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Segment a few images!
predict_files = [
        "data/bitou_test/DJI_20220404135614_0001.JPG",
        "data/bitou_test/DJI_20220404140510_0015.JPG",
        "data/bitou_test/DJI_20220404140802_0022.JPG"
    ]
datamodule = SemanticSegmentationData.from_files(
    predict_files=predict_files,
    batch_size=3,
)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("results/tmp/segmentation_model_overfit.pt")