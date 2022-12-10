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
            out_img[labels == idx] = self.colour_code[idx]
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

    def __init__(self, mean : tuple =(0.485, 0.456, 0.406), std : tuple=(0.229,0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image : torch.Tensor, target : torch.Tensor) -> tuple:
        """
            TODO: Ensure image axes consistency
        """
        image = InverseNormalization.__check_tensor()
        z = image * torch.tensor(self.std).view(3, 1, 1)
        z = z + torch.tensor(self.mean).view(3,1,1)
        return z, target
    
    def __repr__(self) -> str:
        return type(self).__name__
    
    def getTransformParams(self) -> dict:
        return {"mean": self.mean, "std": self.std}
    
    @staticmethod
    def __check_tensor(tensor):
        """
            Function to check type conversion on a tensor
        """
        if isinstance(tensor, np.ndarray):
            tensor = torch.tensor(tensor)
        return tensor

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

def write_image(mask_dir : Path, im_fname : str, im_f : np.ndarray, f_ext : str = ".png") -> None:
    """
        OpenCVs saving image function:
        Can be loaded as class indices with cv2.IMREAD_UNCHANGED in the load function
    """
    mask_dir = to_Path(mask_dir)
    m_path = mask_dir / (im_fname + f_ext)
    if os.path.exists(m_path): raise OSError("{} already exists.".format(m_path))
    cv2.imwrite(str(m_path), im_f)

def _validate_img(img : np.ndarray) -> bool:
    assert img.shape == 3
    return True

def _validate_label(label : np.ndarray) -> bool:
    assert label.shape == 2
    return True

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


def resize_img(img : np.array, scale_perc : int = 50) -> np.array:
    """
        function to resize the image to whatever is specified by the scale
    """
    width = int(img.shape[1] * scale_perc / 100)
    height = int(img.shape[0] * scale_perc / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    return resized

def replace_image_values(img, r_img, label, exchange_class) -> np.ndarray:
    """
        Function to replace the pixels in the img with pixels from r_img, where the label is exchange_class
        # TODO: make this work with exchange class more than one. Either through *args or through a list, depending on argparse
    """
    # 1. make sure images are the same size
    r_img_ref = __check_image_sizes(img, r_img)
    
    # 2. copy the img
    out_img = np.copy(img)

    # 3. replace it
    out_img[label == exchange_class] = r_img_ref[label == exchange_class]

    return out_img

def __check_image_sizes(img_1: np.ndarray, img_2 : np.ndarray):
    """
        Resize the second to be the size of the first. Only if its bigger, otherwise raise assertion error
    """
    s_2 = img_2.shape[:-1]
    s_1 = img_1.shape[:-1]
    for i in range(len(s_2)):
        assert s_2[i] >= s_1[i]
    resized = cv2.resize(img_2, (s_1[1], s_1[0]))
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
