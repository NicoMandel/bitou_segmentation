"""
    General utility file for use across multiple scripts.
        * Load & Save images
        * basic transforms
        * displaying images
"""
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
import os.path
import torch
"""
    Section on plotting images
"""

class ColourDecoder:

    def __init__(self, coldict : dict) -> None:
        self.colour_dict = coldict
        self.colour_code = np.asarray([v for v in coldict.values()])
        # self.num_classes = len(coldict) if num_classes is None else num_classes

    def __call__(self, labels) -> np.ndarray:
        """
            Call function. Only requires either 1 or 2 arguments:
            First argument must be the labels, second the mask
        """
        return self._decode_colourmap(labels)

    def _decode_colourmap(self, labels : np.ndarray) -> np.ndarray:
        """
            Todo: make this function generic to also work with batches
        """
        out_arr_shape = (labels.shape[:] + (3,)) 
        out_img = np.zeros(out_arr_shape, dtype=np.uint8)
        for idx in range(0, labels.max() +1):
            out_img[labels == idx] = colour_code[idx]
        return out_img

    def __getitem__(self, key : int) -> list:
        return self.colour_code[key].tolist()

    def __setitem__(self, key : int, value : np.ndarray):
        raise NotImplementedError
        
    @classmethod
    def load_colours(cls, fpath):
        """
            Function to load a colour decoder from a json file string
        """
        fpath = check_path(fpath)
        print("Loading colour code from: {}".format(fpath))
        with open(fpath, 'r') as fp:
            cdict = json.load(fp)
        return cls(cdict)
        

def get_colour_decoder(fpath : str = None) -> ColourDecoder:
    if fpath is None:
        pdir = os.path.dirname(os.path.abspath(__file__))
        confdir = os.path.join(pdir, '..', '..', 'config')
        fpath = next(Path(confdir).glob('*_code.json'))
    cd = ColourDecoder.load_colours(fpath)
    return cd


class InverseNormalization(object):

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229,0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        z = image * torch.tensor(self.std).view(3, 1, 1)
        z = z + torch.tensor(self.mean).view(3,1,1)
        return z, target
    
    def __repr__(self):
        return type(self).__name__
    
    def getTransformParams(self):
        return {"mean": self.mean, "std": self.std}

"""
    Functions for loading labels and images - to be used across all sources - for consistency
"""

def load_label(fpath : str) -> np.ndarray:
    fpath = check_path(fpath)
    label = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    return label

def load_image(fpath : str) -> np.ndarray:
    fpath = check_path(fpath)
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    return img

def _validate_img(img : np.ndarray) -> bool:
    assert img.shape == 3
    return True

def _validate_label(label : np.ndarray) -> bool:
    assert label.shape == 2
    return True

colour_code = np.array([
                    # (0, 0, 0),      #black
                    (0, 0, 128),        #blue
                    (0, 128, 0),  # green
                    (192, 0, 0),        # red
                    (0, 128, 192),      # turqouise
                    (192, 128, 0),   # brown
        (192, 192, 192),      # Buildings
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

def decode_colormap(mask, labels, num_classes=2):
        """
            Function to decode the colormap. Receives a numpy array of the correct label
        """
        m = np.copy(mask)
        m = m.reshape((-1,3))
        for idx in range(0, num_classes):
            m[labels == idx] = colour_code[idx]
        colour_map = m.reshape(mask.shape)
        return colour_map

def decode_colormap_labels(labels : np.ndarray) -> np.ndarray:
    out_arr_shape = (labels.shape[:] + (3,)) 
    out_img = np.zeros(out_arr_shape, dtype=np.uint8)
    for idx in range(0, labels.max()):
        out_img[labels == idx] = colour_code[idx]
    return out_img

def disable_cluster(img : np.array, cluster : int, labels, color : list = [255,255,255]) -> np.array:
    """
        Function to disable a cluster from visualisation and return the image
    """
    masked = np.copy(img)
    masked = masked.reshape((-1,3))
    masked[labels == cluster] = color     # turning the specified cluster white
    masked = masked.reshape(img.shape)
    return masked

def overlay_images(img : np.ndarray, mask : np.ndarray, alpha : int = 0.7):
    """
        Function to overlay a mask on top of an image
    """
    overlaid = cv2.addWeighted(img, alpha, mask, (1-alpha), 0)
    return overlaid

"""
    Section on image function
"""
def get_image_list(dir : Path, f_ext : str = ".JPG") -> list:
    if isinstance(dir, str):
        dir = Path(dir)
    img_list = list([x.stem for x in dir.glob("*"+f_ext)])
    return img_list
    
def read_image(img_path : str) -> np.array:
    img_path = check_path(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def plot_images(img : np.ndarray, mask : np.ndarray, img_name : str, K : int) -> None:
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title('Original')

    axs[1].imshow(mask)
    axs[1].axis('off')
    axs[1].set_title('Clusters')
    fig.suptitle(f"Image: {img_name}, Clusters: {K}")
    plt.show()

def plot_grayscale(img : np.ndarray, mask : np.ndarray, title : str) -> None:
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title('Original')

    axs[1].imshow(mask, cmap='gray', vmin=0, vmax=255)
    axs[1].axis('off')
    axs[1].set_title('Grayscale')
    fig.suptitle(f"{title}")
    plt.show()

def plot_grayscales(mask_1 : np.ndarray, mask_2 : np.ndarray, title : str) -> None:
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(mask_1, cmap="gray", vmin=0, vmax=255)
    axs[0].axis('off')
    axs[0].set_title('Mask 1')

    axs[1].imshow(mask_2, cmap='gray', vmin=0, vmax=255)
    axs[1].axis('off')
    axs[1].set_title('Mask 2')
    fig.suptitle(f"{title}")
    plt.show()

def plot_grayscales_diff(mask_1 : np.ndarray, mask_2 : np.ndarray, title : str) -> None:
    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(mask_1, cmap="gray", vmin=0, vmax=255)
    axs[0,0].axis('off')
    axs[0,0].set_title('Mask 1')

    axs[0,1].imshow(mask_2, cmap='gray', vmin=0, vmax=255)
    axs[0,1].axis('off')
    axs[0,1].set_title('Mask 2')

    diff = cv2.absdiff(mask_1, mask_2)
    axs[1,0].imshow(diff, cmap="gray", vmin=0, vmax=255)
    axs[1,0].set_title("Difference")
    axs[1,0].axis('off')

    axs[1,1].imshow((255-diff), cmap="gray", vmin=0, vmax=255)
    axs[1,1].set_title("Inverted Difference")
    axs[1,1].axis('off')

    fig.suptitle(f"{title}")
    plt.show()

def plot_overlaid(mixed : np.ndarray, title : str = ""):
    fig = plt.figure()
    plt.imshow(mixed)
    plt.axis('off')
    plt.title(title)
    plt.show()

def save_image(outfig : str, mask : np.ndarray) -> None:
    cv2.imwrite(outfig, cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

def resize_img(img : np.array, scale_perc : int = 50) -> np.array:
    """
        function to resize the image to whatever is specified by the scale
    """
    width = int(img.shape[1] * scale_perc / 100)
    height = int(img.shape[0] * scale_perc / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    return resized

"""
    Conversion and control section
"""
def check_path(path) -> str:
    """
        Function to convert path object to string, if necessary
    """
    if isinstance(path, Path):
        path = str(path)
    return path

def to_Path(path) -> Path:
    """
        function to convert string object to path, if necessary
    """
    if isinstance(path, str):
        path = Path(path)
    return path

def to_numpy(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        img = np.array(img)
    return img

def to_Image(img) -> Image.Image:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img
