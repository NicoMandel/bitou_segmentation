# The basic semantic segmentation as outlined in the pytorch flash documentation [here](https://lightning-flash.readthedocs.io/en/latest/reference/semantic_segmentation.html)

import torch
import numpy as np

import flash
# from flash.core.data.utils import download_data
from flash.image import SemanticSegmentation, SemanticSegmentationData

from PIL import Image

colour_code = np.array([(220, 220, 220), (128, 0, 0), (0, 128, 0),  # class
                        (192, 0, 0), (64, 128, 0), (192, 128, 0)])  # background

def decode_colormap(labels, num_classes=2):
        """
            Function to decode the colormap. Receives a numpy array of the correct label
        """
        r = np.zeros_like(labels).astype(np.uint8)
        g = np.zeros_like(labels).astype(np.uint8)
        b = np.zeros_like(labels).astype(np.uint8)
        for class_idx in range(0, num_classes):
            idx = labels == class_idx
            r[idx] = colour_code[class_idx, 0]
            g[idx] = colour_code[class_idx, 1]
            b[idx] = colour_code[class_idx, 2]
        colour_map = np.stack([r, g, b], axis=2)
        # colour_map = colour_map.transpose(2,0,1)
        # colour_map = torch.tensor(colour_map)
        # image = image.to("cpu").numpy().transpose(1, 2, 0)
        return colour_map

gpus = "cuda:0"
map_location = {'cpu':'cuda:0'}
model_f = "results/tmp/segmentation_model_overfit.pt"
model = SemanticSegmentation.load_from_checkpoint(model_f,map_location=map_location)
# pretrained_model.eval()
# pretrained_model.freeze()
# y_hat = pretrained_model(x)

# SemanticSegmentation.available_outputs()

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())

# 4. Segment a few images!
predict_files = [
        "data/bitou_test/DJI_20220404135614_0001.JPG",
        # "data/bitou_test/DJI_20220404140510_0015.JPG",
        # "data/bitou_test/DJI_20220404140802_0022.JPG"
    ]
datamodule = SemanticSegmentationData.from_files(
    predict_files=predict_files,
    batch_size=1
)
predictions = trainer.predict(model, datamodule=datamodule, output='preds') # using 'labels' does not work - unknown why. class names?
print(predictions)
pred = predictions[0][0]
print(pred.shape)
label = torch.argmax(pred.squeeze(), dim=0).detach().numpy()
# label = (pred > 0.0).float()
print(label.shape)
decoded = decode_colormap(label.detach().numpy(), num_classes=2)

# 6. Show the images
imf = predict_files[0] 
im = Image.open(imf)
im.show()

# show the predictions
im_lab = Image.fromarray(decoded)
im_lab.show()

