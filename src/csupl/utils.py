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

"""
    Section on plotting images
"""
colour_code = np.array([(0, 0, 0),      #black
                    (128, 0, 0),        #red
                    (0, 128, 0),  # green
                    (0, 0, 128),        # blue
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

def disable_cluster(img : np.array, cluster : int, labels, color : list = [255,255,255]) -> np.array:
    """
        Function to disable a cluster from visualisation and return the image
    """
    masked = np.copy(img)
    masked = masked.reshape((-1,3))
    masked[labels == cluster] = color     # turning the specified cluster white
    masked = masked.reshape(img.shape)
    return masked


"""
    Section on image function
"""
def get_image_list(dir : Path, f_ext : str = ".JPG") -> list:
    img_list = list([x.stem for x in dir.glob("*"+f_ext)])
    return img_list
    
def read_image(img_path : str) -> np.array:
    if isinstance(img_path, Path):
        img_path = str(img_path)
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
    print("Test line for debugging")

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