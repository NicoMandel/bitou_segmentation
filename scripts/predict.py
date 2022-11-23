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

from tqdm import tqdm
import matplotlib.pyplot as plt

from csupl.utils import load_image, load_label, get_colour_decoder, InverseNormalization

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
        im_f = os.path.join(img_dir, img + ".JPG")
        im = load_image(im_f)
        mask_f = os.path.join(mask_dir, img + ".png")
        mask = load_label(mask_f)
        out = transforms(image = im, mask = mask)
        im_batch[i, ...] = out['image']
        m_batch[i, ...] = out['mask']

    # pass all of them through at the same time
    # x, y = transforms(image = im_batch, mask = m_batch)
    x = im_batch.to(device)
    with torch.no_grad():
        y_hat = model(x)
    y_hat = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
    # y_hat = y_hat.sigmoid()
    return im_batch.detach().numpy(), y_hat, m_batch.detach().squeeze().numpy()

def plot3x3(input, mask, output, _, fnames, inv_norm ,colour_decoder):
    fig, axs = plt.subplots(3,3)
    mask = mask.astype(np.int)
    for i in range(3):
        m = mask[i, ...]
        # in_img, m = inv_norm(input[i, ...], m)
        inp = input[i, ...]

        axs[i,0].imshow(inp)
        axs[i,0].axis('off')
        axs[i,0].set_title(f"Original: {fnames[i]}")

        m = colour_decoder(m)
        axs[i,1].imshow(m)
        axs[i,1].axis('off')
        axs[i,1].set_title('Mask')

        out = output[i, ...]
        out = colour_decoder(out)
        axs[i,2].imshow(out)
        axs[i,2].axis('off')
        axs[i,2].set_title('Output')
    plt.show()
    print("Test debug line")


def plot3x4(input, mask, y_tr, y_untr, fnames, inv_norm, colour_decoder):
    fig, axs = plt.subplots(3,4)
    mask = mask.astype(np.int)
    for i in range(3):
        m = mask[i, ...]
        # inp, m = inv_norm(input[i, ...], m)
        inp = input[i, ...]
        axs[i,0].imshow(inp)
        axs[i,0].axis('off')
        axs[i,0].set_title(f"Original: {fnames[i]}")

        
        m = colour_decoder(m)
        axs[i,1].imshow(m)
        axs[i,1].axis('off')
        axs[i,1].set_title('Mask')

        untr = y_untr[i, ...]
        untr = colour_decoder(untr)
        # untr = untr.squeeze().cpu().numpy()
        axs[i,2].imshow(untr)
        axs[i,2].axis('off')
        axs[i,2].set_title('Untrained')

        tr = y_tr[i, ...]
        tr = colour_decoder(tr)
        # tr = tr.squeeze().cpu().numpy()
        axs[i,3].imshow(tr)
        axs[i,3].axis('off')
        axs[i,3].set_title('Trained')
    plt.show()
    print("Test Debug Line")


def run_lightning_trainer_images(datadir, augmentations, model, colour_decoder):

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
            dec = colour_decoder(lab)
            axs[i].imshow(dec)
            axs[i].axis('off')
        # decoded = decode_colormap(label_argm, num_classes=2)
        plt.show()
        print("Test Debug")

def get_bitou_multiclass_case():
    predict_files = [
        "DJI_20220404135614_0001",
        "DJI_20220404140510_0015",
        "DJI_20220404140802_0022"
    ]
    model_f = "results/tmp/models/multiclass/FPNresnet34_trained-2022-11-22-19:1:23.pt"
    model_f2 = "results/tmp/models/multiclass/FPNresnet34_untrained-2022-11-22-19:1:23.pt"
    datadir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data')
    img_dir = os.path.join(datadir, 'bitou_test')
    mask_dir = os.path.join(datadir, 'labels_multiclass')
    num_classes = 3
    plot_fn = plot3x4
    return predict_files, model_f, model_f2, datadir, img_dir, mask_dir, num_classes, plot_fn

def get_bitou_standard_case():
    predict_files = [
        "DJI_20220404135614_0001.JPG",
        "DJI_20220404140510_0015.JPG",
        "DJI_20220404140802_0022.JPG"
    ]
    model_f = "results/tmp/models/bitou/FPNresnet34-2022-10-27-16:20:25.pt"
    model_f2 = None
    datadir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data')
    img_dir = os.path.join(datadir, 'bitou_test')
    mask_dir = os.path.join(datadir, 'bitou_binary_masks')
    num_classes = 2
    plot_fn = plot3x3
    return predict_files, model_f, model_f2, datadir, img_dir, mask_dir, num_classes, plot_fn


def get_bitou_crop_case():
    predict_files = [
        "DJI_20220404135614_0001.JPG",
        "DJI_20220404140510_0015.JPG",
        "DJI_20220404140802_0022.JPG"
    ]
    model_f = "results/tmp/models/bitou/FPNresnet34_trained_crop-2022-11-01-11:12:2.pt"
    model_f2 = "results/tmp/models/bitou/FPNresnet34_untrained_crop-2022-11-01-11:12:2.pt"
    datadir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', 'bitou_crop')
    img_dir = os.path.join(datadir, 'orig')
    mask_dir = os.path.join(datadir, 'mask')
    num_classes = 2
    plot_fn = plot3x4
    return predict_files, model_f, model_f2, datadir, img_dir, mask_dir, num_classes, plot_fn

def get_bitou_balance_case():
    predict_files = [
        "DJI_20220404135614_0001.JPG",
        "DJI_20220404140510_0015.JPG",
        "DJI_20220404140802_0022.JPG"
    ]
    # model_f = "results/tmp/models/bitou/FPNresnet34_trained_balance-2022-11-01-11:19:19.pt"
    # model_f2 = "results/tmp/models/bitou/FPNresnet34_untrained_balance-2022-11-01-11:19:19.pt"
    # model_f = "results/tmp/models/bitou/FPNresnet34_trained_balance-2022-11-02-9:40:55.pt"
    # model_f2 = "results/tmp/models/bitou/FPNresnet34_untrained_balance-2022-11-02-9:40:55.pt"
    model_f = "results/tmp/models/bitou/FPNresnet34_trained_balance-2022-11-02-10:17:13.pt"
    model_f2 = "results/tmp/models/bitou/FPNresnet34_untrained_balance-2022-11-02-10:17:13.pt"
    datadir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', 'bitou_balance')
    img_dir = os.path.join(datadir, 'orig')
    mask_dir = os.path.join(datadir, 'mask')
    num_classes = 2
    plot_fn = plot3x4
    return predict_files, model_f, model_f2, datadir, img_dir, mask_dir, num_classes, plot_fn

def get_carla_case():
    predict_files = [
        "F61-14.png",
        "F69-25.png",
        "F65-32.png"
    ]
    model_f = "results/tmp/models/carla/FPNresnet34_trained-2022-10-31-15:19:17.pt"
    model_f2 = "results/tmp/models/carla/FPNresnet34_untrained-2022-10-31-15:19:17.pt"
    datadir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data')
    img_dir = os.path.join(datadir, 'CameraRGB')
    mask_dir = os.path.join(datadir, 'CameraSeg')
    num_classes = 2
    plot_fn = plot3x4
    return predict_files, model_f, model_f2, datadir, img_dir, mask_dir, num_classes, plot_fn



if __name__=="__main__":
    # fixing the seed
    pl.seed_everything(8)       # seed 8 shows one image

    predict_files, model_f, model_f2, datadir, img_dir, mask_dir, num_classes, plot_fn = get_bitou_multiclass_case()

    model = Model.load_from_checkpoint(
        model_f, 
        # map_location=map_location
        )

    # get the basic transform - normalization
    preprocess_params = model.get_preprocessing_parameters()
    mean = tuple(preprocess_params['mean'])
    std = tuple(preprocess_params['std'])

    height = 512
    width = 512
    augmentations = A.Compose([
        A.Resize(height, width, p=1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True)
    ])
    
    # EITHER
    im_batch, y_tr, y = run_deterministic_images(model, augmentations, predict_files, img_dir, mask_dir, im_shape=(height, width))
    if model_f2 is not None:
        model = Model.load_from_checkpoint(model_f2)
        _, y_untr, _ = run_deterministic_images(model, augmentations, predict_files, img_dir, mask_dir, (height, width))
    else:
        y_untr = None
    # # invert axes
    im_batch = np.moveaxis(im_batch, 1, -1)
    # y_hat = np.moveaxis(y_hat, 1, -1)
    # y = np.moveaxis(y, 1, -1)
    coldec = get_colour_decoder()
    inv_norm = InverseNormalization(mean = mean, std = std)
    plot_fn(im_batch, y, y_tr, y_untr, predict_files, inv_norm, coldec)     

    print("Test Debug Line")