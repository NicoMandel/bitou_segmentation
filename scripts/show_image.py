import matplotlib.pyplot as plt
from csupl.utils import load_image
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Script to show an image with matplotlib")
    
    # Arguments
    parser.add_argument("-f", "--file", required=True, type=str, help="Path to the image.")
    args = parser.parse_args()
    return vars(args)

if __name__=="__main__":
    args = parse_args()
    img = load_image(args["file"])
    plt.imshow(img)
    plt.show()
