"""
    Utilities for the dataset.
    Plotting:
        * Source [1](https://debuggercafe.com/semantic-segmentation-using-pytorch-fcn-resnet/)
        * Source [2](https://www.pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/)
        * Source [3](https://github.com/spmallick/learnopencv/blob/master/PyTorch-Segmentation-torchvision/intro-seg.ipynb)
"""
import torch
import matplotlib.pyplot as plt
import os.path

from dataloader import SegDataset
import numpy as np

# abstract model - for model checks
import pytorch_lightning as pl
from torchsummary import summary

import transforms as tfs
import torchmetrics
import torch.nn

from PIL import Image
from datetime import datetime
from torchvision.transforms import ToPILImage, ToTensor


# colour code has to be the same as number of classes. RGB Channels
colour_code = np.array([(220, 220, 220), (128, 0, 0), (0, 128, 0),  # class
                        (192, 0, 0), (64, 128, 0), (192, 128, 0)])  # background

def plot_sample(img, mask, idx=None):
    """
        function to plot an image and a mask. parameters: numpy arrays for plotting
    """
    fig, axs = plt.subplots(1,2, figsize=(15, 10))
    axs[0].imshow(img, interpolation='nearest')
    axs[1].imshow(mask, interpolation='nearest')
    axs[0].set_title("Input")
    axs[1].set_title("Prediction")
    axs[0].axis('off')
    axs[1].axis('off')
    if idx is not None:
        fig.suptitle("Index: {}".format(idx))
    plt.show()

def save_sample(img, mask, path=None, idx=None):
    """
        function to save an image and mask in one figure
    """
    if path is None:
        print("No path specified, just plotting")
        plot_sample(img, mask, idx)
    else:
        fig, axs = plt.subplots(1,2, figsize=(15, 10))
        axs[0].imshow(img, interpolation='nearest')
        axs[1].imshow(mask, interpolation='nearest')
        axs[0].set_title("Input")
        axs[1].set_title("Prediction")
        axs[0].axis('off')
        axs[1].axis('off')
        if idx is not None:
            fig.suptitle("Index: {}".format(idx))
        plt.savefig(path)

def get_plotting_dataset(root_dir, transforms=None, width=None, height=None):
    """
        Returns a pytorch dataset object, so that the __getitem__ method can be used to return images
    """
    dataset = SegDataset(root_dir, transforms=transforms, width=width, height=height)
    return dataset

def plot_index(dataset, idx):
    """
        plotting a path. Requires a dataset as specified by get_plotting_dataset. Will create a pytorch dataset object and use the
        __getitem__ method to receive the image.
            Without any transforms
            Without any resizes
    """ 
    img, mask = dataset[idx]
    img = np.asarray(img)
    mask = np.asarray(mask)
    plot_sample(img, mask, idx)

def save_index(dataset, idx):
    """
        saving an index image from a path. Requires a dataset as specified by get_plotting_dataset and use the 
        __getitem__ method. Analogous to plot_index()
    """
    img, mask = dataset[idx]
    img = np.asarray(img)
    mask = np.asarray(mask)
    save_sample(img, mask, idx)

def plot_random_idx(dataset):
    """
        Function to inspect random index of the dataset. Requires a dataset object. 
    """
    idx = get_random_idx(dataset)
    plot_index(dataset, idx)

def get_random_idx(dataset):
    """
        Function to return a random index from the dataset
    """
    max_idx = len(dataset)
    idx = np.random.randint(0, max_idx)
    return idx

def plot_random_examples(root_dir, num_examples=4, model=None, transforms=None, width=None, height=None):
    """
        Function to plot random examples. If model is None, then only two images are shown.
        If model is set, the transforms need to be set too
    """
    ds = get_plotting_dataset(root_dir, transforms=transforms, width=width, height=height)
    IN = tfs.InverseNormalization()
    for i in range(num_examples):
        idx = get_random_idx(ds)
        print("Example Number: {}. Index: {}".format(i, idx))
        img, mask = ds[idx]
        truth_labels = mask.detach().cpu().numpy()

        decoded_truth = decode_colormap(truth_labels, 2)
        if model is not None:
            img = img.unsqueeze(0)
            if model.device.type != img.device.type:
                img = img.to(device='cuda')
            img_pred = model(img)
            labels = torch.argmax(img_pred.squeeze(), dim=0).detach().cpu().numpy()
            decoded_pred = decode_colormap(labels, 2)

            # unnormalize the image to plot the normal one!
            norm_img = IN(img, mask)
            plot_triplet([norm_img[0].squeeze(0).cpu(), decoded_truth, decoded_pred])
        else:
            plot_sample(img, mask, idx)


def decode_colormap(labels, num_classes):
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
    colour_map = colour_map.transpose(2,0,1)
    colour_map = torch.tensor(colour_map)
    # image = image.to("cpu").numpy().transpose(1, 2, 0)
    return colour_map

def plot_triplet(tensor_triplet, title=None):
    """
        Quick helper function to plot a triplet of images
    """
    np_trip = [img.numpy().transpose(2,1,0) for img in tensor_triplet]
    fig, axs = plt.subplots(1,3, figsize=(15, 10))
    axs[0].imshow(np_trip[0])
    axs[1].imshow(np_trip[1], interpolation='nearest')
    axs[2].imshow(np_trip[2], interpolation='nearest')
    for ax in axs:
        ax.axis('off')
    axs[0].set_title("Input")
    axs[1].set_title("Ground Truth")
    axs[2].set_title("Prediction")
    if title is not None:
        plt.suptitle(title)
    plt.show()

def onnx_export(model, filename, directory, height=192, width=256, pt_model=True):
    """
        Function to export the model using the onnx format.
        Also saves a normal pytorch file to be loaded with pytorch if pt_model is True
        Params:
            * model to export
            * filename where to store the model. Without .onnx extension
            * directory where the model should be stored
            * width and height of the images
                * same for semantic segmentation
            * pt_model to say whether a pytorch model should be saved alongside the onnx model
                * not using the state_dict function, but also including the shape, see documentation
                [here](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html#saving-and-loading-models-with-shapes)
    """
    # Name operation
    fname = os.path.join(directory, filename + ".onnx")

    # !
    # TODO: we need to triple check dimensions here! 
    # ! 
    shape = (1, 3, height, width)
    X = torch.ones(shape, dtype=torch.float, device="cuda" if model.device.type == "cuda" else "cpu")
    
    # Actual export step - coming from the detector_debugger notebook
    torch.onnx.export(model,
                    X,                          # example import
                    fname, 
                    export_params=True,         # storing the weights
                    opset_version=11,           # onnx version to export the model to
                    do_constant_folding=True,   # for optimization
                    input_names=['input'],      
                    output_names=['output'],
                    dynamic_axes={'input' : {0 : 'batch_size'}, 'output': {0 : 'batch_size'}})  # variable length axes

    # Alternative export as suggested by the pytorch documentation
    # inp_img = torch.zeros((1, 2, height, width))
    # torch.onnx.export(model, inp_img, fname)
    print("Done saving ONNX model {} with parameters: width: {}, height: {} to location: {}".format(type(model).__name__, width, height, fname))

    if pt_model:
        fname = os.path.join(directory, filename + ".pth")
        torch.save(model, fname)
        print("Also saved pytorch model in .pth format to: {}".format(fname))
    
def import_model(filename, directory):
    """
        Function to load a pytorch model
    """
    fname = os.path.join(directory, filename)
    model = torch.load(fname)
    return model

def import_model_from_path(path):
    """
        Function to load a pytorch model from a single path
    """
    model = torch.load(path)
    return model

# model checking function
def check_model(model, size):
    """
        Function to check the model. Checks for correct instances in inheritance order.
    """
    from model import DeepLab
    print("Torch Module instance: {}".format(isinstance(model, torch.nn.Module)))
    print("Lightning Module instance: {}".format(isinstance(model, pl.LightningModule)))
    if isinstance(model, DeepLab):
        print(model)
    else:
        summary(model, size, device="cpu")
    
# Checking the transforms
def plotTransform(model, transform, width, height, root_dir, reduction=None, idx=None):
    """
        Function to plot a single transform. Composes with ToTensor() and Normalize().
        Requires:
            * a model
            * a testable transform
            * an index to display - if None, randomly sample
            * a (width, height) tuple
    """
    # ensure model is in inference mode
    model.freeze()

    # Setup the transform
    transforms = tfs.Compose([
        transform,
        tfs.ToTensor(),
        tfs.Normalize()
    ])
    IN = tfs.InverseNormalization()

    ds = SegDataset(root_dir, transforms, width=width, height=height, reduction=reduction)
    if idx is None:
        idx = get_random_idx(ds)
    
    image, mask = ds[idx]
    truth_labels = mask.detach().cpu().numpy()

    decoded_truth = decode_colormap(truth_labels, 2)
    img = image.unsqueeze(0)      # adding a batch dimension
    if model.device.type != img.device.type:
        img = img.to(device='cuda')
    img_pred = model(img)
    labels = torch.argmax(img_pred.squeeze(), dim=0).detach().cpu().numpy()
    decoded_pred = decode_colormap(labels, 2)

    # unnormalize the image to plot the normal one!
    norm_img, _ = IN(img, mask)
    norm_img = norm_img.squeeze(0)

    # Getting some metrics:
    softm = torch.nn.Softmax(dim=1)  # ! check if dimension 1 is the right dimension
    soft = softm(img_pred)
    accuracy = torchmetrics.IoU(2)
    acc = accuracy(soft, mask)

    # Plotting stuff
    title = "Example number: {}, Resolution: {}x{}, transformation: {}. IoU Accuracy: {:.3f}".format(idx, width, height, type(transform).__name__, acc)
    plot_triplet([norm_img, decoded_truth, decoded_pred], title=title)


def image_overlay(image, segmented_image):
    alpha = 0.6
    TPL = ToPILImage()
    TTS = ToTensor()
    img = TPL(image)
    segm = TPL(segmented_image)
    overlay = Image.blend(img, segm, alpha)
    overlay = TTS(overlay)
    # overlay = alpha * image.detach().cpu() + (1-alpha) * segmented_image
    return overlay

class VideoInference(object):

    def __init__(self, model, transforms, device, num_classes):
        self.num_classes = num_classes
        self.transforms = transforms
        self.device = device
        model.to(device)
        self.model = model
    
    def __call__(self, img):
        with torch.no_grad():
            img = self.transforms(img)
            img = img.unsqueeze(0)
            img = img.to(self.device)
            out = self.model(img)
            label = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        decoded_cm = self.decode_colormap(label)
        return decoded_cm

    def decode_colormap(self, labels):
        """
            Function to decode the colormap. Receives a numpy array of the correct label
        """
        r = np.zeros_like(labels).astype(np.uint8)
        g = np.zeros_like(labels).astype(np.uint8)
        b = np.zeros_like(labels).astype(np.uint8)
        for class_idx in range(0, self.num_classes):
            idx = labels == class_idx
            r[idx] = colour_code[class_idx, 0]
            g[idx] = colour_code[class_idx, 1]
            b[idx] = colour_code[class_idx, 2]
        colour_map = np.stack([b, g, r], axis=2)
        # colour_map = colour_map.transpose(2,0,1)
        # colour_map = torch.tensor(colour_map)
        # image = image.to("cpu").numpy().transpose(1, 2, 0)
        return colour_map
