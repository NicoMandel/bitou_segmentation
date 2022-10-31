# The basic semantic segmentation as outlined in the pytorch flash documentation [here](https://lightning-flash.readthedocs.io/en/latest/reference/semantic_segmentation.html)

import torch
import numpy as np
import os.path

import pytorch_lightning as pl
# from flash.core.data.utils import download_data
from csupl.model import Model
from csupl.dataloader import BitouDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image

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
                        (220, 0, 220),
                        (0, 220, 220),
                        (110, 110, 0),
                        (0, 110, 110)
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

def alternative_decode_colormap(label, num_classes=2):
    m = np.zeros_like(label)
    for idx in range(0, num_classes):
        m[label == idx] = colour_code[idx]
    return m

def load_image(path : str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_mask(path : str):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = mask[...,0]
    mask = mask[..., np.newaxis]
    return mask

def load_image_and_mask(img_dir : str, mask_dir : str, img_name : str) -> tuple:
    img_fname = os.path.join(img_dir, img_name )
    mask_fname = os.path.join(mask_dir, img_name)
    
    img = load_image(img_fname)
    mask = load_mask(mask_fname)
    return (img, mask)

def run_deterministic_images(model : Model, transforms : A.Compose, img_list : list, img_dir : str, mask_dir : str, im_shape):
    """
        function to run deterministic dataloading and images
        or use: next(iter(dl))
        or use ds[10]
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_shape = tuple([len(img_list), 3] + list(im_shape))
    im_batch = torch.zeros(batch_shape)
    mask_batch_shape = tuple([len(img_list), 1] + list(im_shape))
    m_batch = torch.zeros(mask_batch_shape)
    model.eval()
    model.to(device)
    for i, img in tqdm(enumerate(img_list)):
        im, mask = load_image_and_mask(img_dir, mask_dir, img)
        out = transforms(image = im, mask = mask)
        im_batch[i, ...] = out['image']
        m_batch[i, ...] = out['mask']

    # pass all of them through at the same time
    # x, y = transforms(image = im_batch, mask = m_batch)
    x = im_batch.to(device)
    with torch.no_grad():
        y_hat = model(x)
    y_hat = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
    return im_batch.detach().numpy(), y_hat, m_batch.detach().numpy()

def plot3x4(input, mask, pred_trained, pred_untrained, fnames, num_classes=21):
    fig, axs = plt.subplots(3,4)
    for i in range(3):
        axs[i,0].imshow(input[i, ...])
        axs[i,0].axis('off')
        axs[i,0].set_title('Original {}'.format(fnames[i]))

        m = mask[i, ...]
        m = decode_colormap(m.squeeze(), num_classes=num_classes)
        axs[i,1].imshow(m)
        axs[i,1].axis('off')
        axs[i,1].set_title('Mask')

        untr = pred_untrained[i, ...]
        untr = decode_colormap(untr, num_classes=num_classes)
        axs[i,2].imshow(untr)
        axs[i,2].axis('off')
        axs[i,2].set_title('Untrained')

        tr = pred_trained[i, ...]
        tr = decode_colormap(tr, num_classes=num_classes)
        axs[i,3].imshow(tr)
        axs[i,3].axis('off')
        axs[i,3].set_title('Trained')
    plt.show()


def run_lightning_trainer_images(datadir, augmentations, model):

    ds = BitouDataset(datadir, augmentations, img_folder="bitou_test", mask_folder="bitou_binary_masks")
    # get a dataloader of the dataset
    batch_size = 3
    num_workers = batch_size if batch_size < 12 else 12
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # or use: next(iter(dl))
    # or use ds[10]
    trainer = pl.Trainer(
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=torch.cuda.device_count(),
                fast_dev_run=8,         # run 8 batch
                deterministic=True
                )
    predictions = trainer.predict(model,dl)
    for pred in predictions:  
        label_argm = torch.argmax(pred, dim=1).detach().cpu().numpy()
        # for i in range(0,batch_size):
        #     pred = predictions[i, ...]
        #     label = torch.argmax(pred, dim=0).detach().cpu().numpy()
        
        # l = torch.argmax(predictions.squeeze(), dim=0).detach().cpu().numpy()
        # label = (pred > 0.0).float()
        # print(label_argm.shape)
        fig, axs = plt.subplots(batch_size)
        for i in range(batch_size):
            lab = label_argm[i, ...]
            dec = decode_colormap(lab)
            axs[i].imshow(dec)
            axs[i].axis('off')
        # decoded = decode_colormap(label_argm, num_classes=2)
        plt.show()
        print("Test Debug")

if __name__=="__main__":
    # fixing the seed
    pl.seed_everything(42)     
    # get the model
    untrained_model = "results/tmp/models/carla/FPNresnet34_untrained-2022-10-31-15:19:17.pt"
    trained_model = "results/tmp/models/carla/FPNresnet34_trained-2022-10-31-15:19:17.pt"
    # ckpt_pth = "lightning_logs/version_8/checkpoints/epoch=9-step=60.ckpt"
    # model_f = "results/tmp/FPNresnet34-2022-10-27-16:20:25.pt"
    # ckpt = torch.load(ckpt_pth)
    # hparams = ckpt['hyper_parameters']
    model = Model.load_from_checkpoint(
        untrained_model,
        # map_location=map_location
        )
    model.freeze()

    # 4. Segment a few images!
    # predict_files = [
    #         "DJI_20220404135614_0001.JPG",
    #         "DJI_20220404140510_0015.JPG",
    #         "DJI_20220404140802_0022.JPG"
    #     ]
    
    # Carla Prediction files
    predict_files = [
        "F61-14.png",
        "F69-25.png",
        "F65-32.png",
    ]

    # generate a dataset
    datadir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data')

    # get the basic transform - normalization
    preprocess_params = model.get_preprocessing_parameters()
    mean = tuple(preprocess_params['mean'])
    std = tuple(preprocess_params['std'])

    height = 512
    width = 512
    augmentations = A.Compose([
        A.RandomCrop(height, width, p=1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True)
    ])

    # point where we split both paths
    img_dir = os.path.join(datadir, 'CameraRGB')
    mask_dir = os.path.join(datadir, 'CameraSeg')
    
    # EITHER
    im_batch, y_hat_untrained, y = run_deterministic_images(model, augmentations, predict_files, img_dir, mask_dir, im_shape=(height, width))
    model = Model.load_from_checkpoint(
        trained_model,
        # map_location=map_location
        )
    model.freeze()
    _, y_hat_trained, _ = run_deterministic_images(model, augmentations, predict_files, img_dir, mask_dir, im_shape=(height, width))
    # # invert axes
    im_batch = np.moveaxis(im_batch, 1, -1)
    # y_hat_trained = np.moveaxis(y_hat_trained, 1, -1)
    # y_hat_untrained = np.moveaxis(y_hat_trained, 1, -1)
    y = np.moveaxis(y, 1, -1)
    plot3x4(im_batch, y, y_hat_trained, y_hat_untrained, predict_files)

    # OR
    # run_lightning_trainer_images(datadir, augmentations, model)