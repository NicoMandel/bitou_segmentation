import torch

from csupl.model import PetModel
# from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from csupl.dataloader import SimpleBitouPetDataset

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def visualize_results(model_trained, model_untrained, dl):
    batch = next(iter(dl))
    with torch.no_grad():
        model_trained.eval()
        logits_tr = model_trained(batch["image"])
        logits_untr = model_untrained(batch["image"])
    masks_trained = logits_tr.sigmoid()
    masks_untrained = logits_untr.sigmoid()

    for image, gt_mask, mask_trained, mask_untrained in zip(batch["image"], batch["mask"], masks_trained, masks_untrained):
        plt.figure(figsize=(10,5))

        plt.subplot(1,4,1)
        plt.imshow(image.numpy().transpose(1,2,0))
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1,4,2)
        plt.imshow(gt_mask.numpy().squeeze())
        plt.title("Mask")
        plt.axis("off")

        plt.subplot(1,4,3)
        plt.imshow(mask_trained.numpy().squeeze())
        plt.title("Trained")
        plt.axis("off")

        plt.subplot(1,4,4)
        plt.imshow(mask_untrained.numpy().squeeze())
        plt.title("Untrained")
        plt.axis("off")

        plt.show()

if __name__=="__main__":
    root = "data/bitou_crop"
    modeldir = "results/tmp/models/bitou/"
    modelf_trained = modeldir + "binary_trained.pt"
    modelf_untrained = modeldir + "binary_untrained.pt"

    test_dataset = SimpleBitouPetDataset(root, "valid")
    # test_dataset = SimpleOxfordPetDataset(root, "test")
    batch_size = 4
    num_workers = batch_size if batch_size < 12 else 12
    test_dl = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    model_tr = PetModel.load_from_checkpoint(modelf_trained)
    model_untr = PetModel.load_from_checkpoint(modelf_untrained)

    visualize_results(model_tr, model_untr, test_dl) 


