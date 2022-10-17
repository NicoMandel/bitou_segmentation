# The basic semantic segmentation as outlined in the pytorch flash documentation [here](https://lightning-flash.readthedocs.io/en/latest/reference/semantic_segmentation.html)

from base64 import decode
import torch
import numpy as np

import flash
# from flash.core.data.utils import download_data
from flash.image import SemanticSegmentation, SemanticSegmentationData

from PIL import Image
import matplotlib.pyplot as plt

colour_code = np.array([(220, 220, 220), (128, 0, 0), (0, 128, 0),  # class
                        (192, 0, 0), (64, 128, 0), (192, 128, 0),   # background
        (70, 70, 70),      # Buildings
        (190, 153, 153),   # Fences
        (72, 0, 90),       # Other
        (220, 20, 60),     # Pedestrians
        (153, 153, 153),   # Poles
        (157, 234, 50),    # RoadLines
        (128, 64, 128),    # Roads
        (244, 35, 232),    # Sidewalks
        (107, 142, 35),    # Vegetation
        (0, 0, 255),      # Vehicles
        (102, 102, 156),  # Walls
        (220, 220, 0),
        (220, 220, 0),
        (220, 220, 0),(220, 220, 0),(220, 220, 0)
                        ])  # background

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
model_f = "results/tmp/carla_FPN_mnetv3_small_075_overfit_freeze.pt"
model = SemanticSegmentation.load_from_checkpoint(model_f,map_location=map_location)
# pretrained_model.eval()
# pretrained_model.freeze()
# y_hat = pretrained_model(x)

# SemanticSegmentation.available_outputs()

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())

# 4. Segment a few images!
predict_files = [
        "data/CameraRGB/F61-20.png",
        "data/CameraRGB/F64-3.png",
        "data/CameraRGB/F66-27.png"
    ]

mask_files = [
    "data/CameraSeg/F61-20.png",
    "data/CameraSeg/F64-3.png",
    "data/CameraSeg/F66-27.png",
] 
datamodule = SemanticSegmentationData.from_files(
    predict_files=predict_files,
    batch_size=1
)
predictions = trainer.predict(model, datamodule=datamodule, output='preds') # using 'labels' does not work - unknown why. class names? output='preds'
print(predictions)
for i, pred in enumerate(predictions):
    pr = pred[0]
    print(pr.shape)
    label = torch.argmax(pr.squeeze(), dim=0).detach().numpy()
    l = (pr > 0.0).float()
    lab = torch.argmax(pr, dim=0).detach().numpy()
    print(label.shape)
    dec_label = decode_colormap(label, num_classes=21)
    dec_l = decode_colormap(l, num_classes=21)
    dec_lab = decode_colormap(lab, num_classes=21)

    # 6. Show the images
    imf = predict_files[i] 
    im = Image.open(imf)
    im_arr = np.asarray(im)

    # show the predictions
    fig, axs = plt.subplots(2,2)    
    axs[0,0].imshow(im_arr)
    axs[0,0].set_title("Original")
    axs[0,0].axis('off')

    axs[0,1].imshow(dec_label)
    axs[0,1].set_title("Squeeze")
    axs[0,1].axis('off')

    # Displaying the OG mask
    maskf = mask_files[i]
    mask = Image.open(maskf)
    mask_arr = np.asarray(mask)[...,0]      # Get the red channel - where the labels are stored
    mask_dec = decode_colormap(mask_arr, num_classes=21)
    axs[1,0].imshow(mask_dec)
    axs[1,0].set_title("Mask")
    axs[1,0].axis('off')

    axs[1,1].imshow(dec_lab)
    axs[1,1].set_title("No squeeze")
    axs[1,1].axis('off')
    plt.show()


