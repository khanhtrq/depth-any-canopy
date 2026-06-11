import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

gedi_folder = "/kaggle/input/datasets/khanhtq2101/gedi-canopy-height-hoanglien/GEDI_filtered/GEDI_filtered"
sentinel_folder = "/kaggle/input/datasets/khanhtq2101/gedi-canopy-height-hoanglien/Sentinel-12band/Sentinel-12band"
regions = ["CucPhuong", "BaBe"]

class GediSentinelDataset(Dataset):
    def __init__(
        self, 
        gedi_folder,
        sentinel_folder,
        regions,
        mode = "train",
        ratio_train = 0.8, 
    ):
        self.regions = regions
        self.gedi_folder = gedi_folder
        self.sentinel_folder = sentinel_folder
        self.mode = mode
        self.ratio_train = ratio_train

        self.sentinel_paths = []
        self.gedi_paths = []

        for r in self.regions:
            # filtering patches with not enough GEDI points
            gedi_paths_all = [os.path.join(r, file_name) for file_name in os.listdir(os.path.join(self.gedi_folder, r))]
            sentinel_paths_all = [os.path.join(r, file_name) for file_name in os.listdir(os.path.join(self.sentinel_folder, r))]
            
            for i in range(len(gedi_paths_all)):
                gedi_path = os.path.join(self.gedi_folder, gedi_paths_all[i])
                gedi = np.load(gedi_path)
       
                if np.sum(~np.isnan(gedi)) >= 50:

                    self.gedi_paths.append(gedi_paths_all[i])
                    self.sentinel_paths.append(sentinel_paths_all[i])

        #Spliting data into train and test set
        rng = np.random.default_rng(seed=42)   # fixed seed
        # rng = np.random.default_rng(seed=2404) 
        file_idx_all = rng.permutation(len(self.gedi_paths)) 

        if self.mode == "train":
            file_idx_train = file_idx_all[:int(self.ratio_train * len(self.gedi_paths))]
            self.file_idx = file_idx_train
        elif self.mode == "test" or self.mode == "val":
            file_idx_test = file_idx_all[int(self.ratio_train * len(self.gedi_paths)):]
            self.file_idx = file_idx_test
        
        print("Dataset length:", len(self.file_idx))


    def __len__(self):
        return len(self.file_idx)

    def __getitem__(self, idx):
        input_file_idx = self.file_idx[idx]

        gedi_path = os.path.join(self.gedi_folder, self.gedi_paths[input_file_idx])
        sentinel_path = os.path.join(self.sentinel_folder, self.sentinel_paths[input_file_idx])

        gedi = np.load(gedi_path)
        sentinel = np.load(sentinel_path)

        print(gedi.shape, sentinel.shape)

        print(gedi.dtype, sentinel.dtype)

        #band 2, 3, 4, RGB?
        sample = {"image": sentinel[1:4], "mask": gedi}

        return sample

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


class GediSentinelDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size=4,
        num_workers=4,
        ratio_train=0.8,
    ):
        super().__init__()

        self.regions = regions
        self.gedi_folder = gedi_folder
        self.sentinel_folder = sentinel_folder

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ratio_train = ratio_train

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = GediSentinelDataset(
                regions=self.regions,
                gedi_folder=self.gedi_folder,
                sentinel_folder=self.sentinel_folder,
                mode="train",
                ratio_train=self.ratio_train,
            )

            self.val_dataset = GediSentinelDataset(
                regions=self.regions,
                gedi_folder=self.gedi_folder,
                sentinel_folder=self.sentinel_folder,
                mode="val",
                ratio_train=self.ratio_train,
            )

        if stage == "test" or stage is None:
            self.test_dataset = GediSentinelDataset(
                regions=self.regions,
                gedi_folder=self.gedi_folder,
                sentinel_folder=self.sentinel_folder,
                mode="test",
                ratio_train=self.ratio_train,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )