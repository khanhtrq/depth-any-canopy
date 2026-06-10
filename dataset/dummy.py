import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import matplotlib.pyplot as plt


class DummyImageMaskDataset(Dataset):
    def __init__(self, num_samples, image_size=(3, 64, 64), mask_size=(64, 64)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.mask_size = mask_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(self.image_size, dtype=torch.float32)
        mask = torch.rand(self.mask_size, dtype=torch.float32)
        return {"image": image, "mask": mask}
    
    def plot(self, image, mask, prediction=None, show_titles=True):
        if prediction is not None:
            prediction = prediction.clip(0, 1).float()
        # Convert image to [0, 1] range
        image = image.float()
        image = image - image.min()
        image = image / image.max()

        mask = mask.float()

        showing_prediction = prediction is not None
        ncols = 2 + int(showing_prediction)
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].axis("off")
        axs[1].imshow(
            mask.squeeze(), interpolation="none", cmap="Spectral_r", vmin=0, vmax=1
        )
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_prediction:
            axs[2].imshow(
                prediction.squeeze(),
                interpolation="none",
                cmap="Spectral_r",
                vmin=0,
                vmax=1,
            )
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")
        return fig

class DummyDatamodule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        size: int = 64, # Use this for H, W of image/mask
        **kwargs # Catch any extra args from hydra config
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = 1000 # A reasonable number for dummy data

        self.image_size = (3, size, size)
        self.mask_size = (size, size)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.save_hyperparameters() # Saves all __init__ arguments to self.hparams

    def setup(self, stage: str):
        if stage == "fit" or stage == "validate":
            self.train_dataset = DummyImageMaskDataset(self.num_samples, self.image_size, self.mask_size)
            self.val_dataset = DummyImageMaskDataset(self.num_samples // 10, self.image_size, self.mask_size)
        if stage == "test":
            self.test_dataset = DummyImageMaskDataset(self.num_samples // 10, self.image_size, self.mask_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
