import rasterio
from rasterio.plot import show, reshape_as_image, reshape_as_raster, adjust_band
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Script to show a rasterfile")
    
    # Arguments
    parser.add_argument("-f", "--file", required=True, type=str, help="Path to the rasterfile.")
    args = parser.parse_args()
    return vars(args)

if __name__=="__main__":
    args = parse_args()
    with rasterio.open(args["file"], 'r') as res:
        img = res.read()
        # display it
    
        show(img)
