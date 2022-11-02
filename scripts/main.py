"""
    ! Tutorial from [here](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb#scrollTo=H4RKHF535Twz)
    ! basic binary segmentation example using smp and albumentations: https://github.com/catalyst-team/catalyst/blob/v21.02rc0/examples/notebooks/segmentation-tutorial.ipynb
    ! tutorial from pytorch lightning themselves: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/04-inception-resnet-densenet.html?highlight=segmentation%20task
"""

import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from csupl.model import PetModel

# lightning
import pytorch_lightning as pl
import matplotlib.pyplot as plt

def plot_sample(sample):
    plt.subplot(1,2,1)
    # transpose back to HWC format
    img = sample["image"]
    ref_img = img.transpose(1,2,0)
    plt.imshow(ref_img)
    plt.subplot(1,2,2)
    mask = sample["mask"]
    # removing the additional dimension
    mask = mask.squeeze()
    plt.imshow(mask)
    plt.show()
    print("Test debug line")

def visualize_results(model, dl):
    batch = next(iter(dl))
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()

    for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
        plt.figure(figsize=(10,5))

        plt.subplot(1,3,1)
        plt.imshow(image.numpy().transpose(1,2,0))
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(gt_mask.numpy().squeeze())
        plt.title("Mask")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(pr_mask.numpy().squeeze())
        plt.title("Prediction")
        plt.axis("off")

        plt.show()

if __name__=="__main__":
    save_model = True
    export_dir = "results/tmp/models/pets"
    root = "data"
    # SimpleOxfordPetDataset.download(root)
    train_dataset = SimpleOxfordPetDataset(root, "train")
    val_dataset = SimpleOxfordPetDataset(root, "valid")
    test_dataset = SimpleOxfordPetDataset(root, "test")

    print("Train size {}\t Val Size {}\t Test Size {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))

    batch_size = 24
    num_workers = batch_size if batch_size < 16 else 16
    train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,  shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    # looking at a few samples
    # TODO: take the dataloading structure out from here
    #! preprocessing in the super @staticmethod for the mask
    sample_train = train_dataset[0]
    sample_test = test_dataset[0]
    sample_val = val_dataset[0]

    plot_sample(sample_train)

    # model
    model = PetModel("FPN", "resnet34", in_channels=3, out_classes=1)

    # TODO: run an initial check here - how well does the model perform before training?! - does that change anything??
    # trainer
    trainer = pl.Trainer(
        accelerator= "gpu" if torch.cuda.is_available() else None,
        devices=torch.cuda.device_count(),
        max_epochs=5,
        limit_predict_batches=1,
    )

    if save_model:
        model.eval()
        trainer.predict(model, train_dl)
        modelpath = export_dir + "/pets_untrained.pt"
        trainer.save_checkpoint(modelpath)
        print(f"model saved to {modelpath}")
        model.unfreeze()

    # Training
    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        )

    # Validation dataset
    val_metrics = trainer.validate(model, dataloaders=val_dl, verbose=False)

    # Testing
    test_metrics = trainer.test(model, dataloaders=test_dl, verbose=False)

    visualize_results(model, test_dl)
    
    # Saving the model
    if save_model:
        model.eval()
        modelpath = export_dir + "/pets_trained.pt"
        trainer.save_checkpoint(modelpath)
        print(f"model saved to {modelpath}")
    print("Test Debug line")
