import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import torchvision.transforms as T


# gedi_folder = "/kaggle/input/datasets/khanhtq2101/gedi-canopy-height-hoanglien/GEDI_filtered/GEDI_filtered"
# sentinel_folder = "/kaggle/input/datasets/khanhtq2101/gedi-canopy-height-hoanglien/Sentinel-12band/Sentinel-12band"
# regions = ["CucPhuong", "BaBe"]

# gedi_folder = "/kaggle/input/datasets/khanhtq2101/gedi-canopy-height-hoanglien/GEDI_filtered_CaMau"
# sentinel_folder = "/kaggle/input/datasets/khanhtq2101/gedi-canopy-height-hoanglien/Sentinel_CaMau"
# regions = ["CaMau"]

class GediSentinelDataset(Dataset):
    def __init__(
        self, 
        gedi_folder,
        sentinel_folder,
        regions,
        mode = "train",
        ratio_train = 0.8, 
        predict=False,
        all_train_data=False,
        max_height = 30,
    ):
        self.regions = regions.split("-") if isinstance(regions, str) else list(regions)
        self.gedi_folder = gedi_folder
        self.sentinel_folder = sentinel_folder
        self.mode = mode
        self.ratio_train = ratio_train
        self.predict = predict
        self.max_height = max_height
        self.all_train_data = all_train_data
        if not 0 < ratio_train < 1:
            raise ValueError("ratio_train must be between 0 and 1")
        if len(set(self.regions)) != len(self.regions):
            raise ValueError("regions must be unique")
        self.gedi_paths, self.sentinel_paths = [], []
        self.sample_regions, self.sample_ids, self.file_idx = [], [], []
        self.split_counts = {}
        for region in self.regions:
            # Folder mappings allow zones stored under unrelated input roots.
            gedi_root = gedi_folder[region] if not isinstance(gedi_folder, str) else gedi_folder
            sentinel_root = sentinel_folder[region] if not isinstance(sentinel_folder, str) else sentinel_folder
            gedi_dir = os.path.join(gedi_root, region)
            sentinel_dir = os.path.join(sentinel_root, region)
            gedi_files = {f for f in os.listdir(gedi_dir) if f.endswith(".npy")}
            sentinel_files = {f for f in os.listdir(sentinel_dir) if f.endswith(".npy")}
            if gedi_files != sentinel_files:
                raise ValueError(f"Unmatched GEDI/Sentinel filenames in {region}")
            indices = []
            for name in sorted(gedi_files, key=lambda x: int(os.path.splitext(x)[0])):
                gedi_path = os.path.join(gedi_dir, name)
                if not predict and not np.isfinite(np.load(gedi_path)).any():
                    continue
                indices.append(len(self.gedi_paths))
                self.gedi_paths.append(gedi_path)
                self.sentinel_paths.append(os.path.join(sentinel_dir, name))
                self.sample_regions.append(region)
                self.sample_ids.append(os.path.splitext(name)[0])
            if predict:
                selected = indices
            else:
                shuffled = np.random.default_rng(seed=42).permutation(indices)
                cut = int(ratio_train * len(indices))
                if cut == 0 or cut == len(indices):
                    raise ValueError(f"{region} needs enough valid patches for train and test")
                selected = shuffled if mode == "train" and all_train_data else (
                    shuffled[:cut] if mode == "train" else shuffled[cut:])
            self.file_idx.extend(selected)
            self.split_counts[region] = len(selected)
        print(f"{mode if not predict else 'predict'} samples per zone: {self.split_counts}")

    @staticmethod
    def normalization():
        return T.Normalize(mean=[1445.2507821473719, 1494.0496883470857, 1695.0355912679454, 1564.6835706583588, 2027.4767477712417, 3214.2947198416723, 3624.9778571404872, 3675.896774446667, 3823.0191113997635, 3810.2311611275345, 2921.714671898899, 2096.318336982956],
                            std=[213.97085480597985, 236.57899563433898, 264.3780942144651, 334.3297911214344, 332.1163963382028, 504.0289211352316, 630.1565564561154, 696.8590235637502, 698.3457937838511, 635.1889167644749, 583.1581743442069, 504.8285260785615]
                )

    def __len__(self):
        return len(self.file_idx)

    def __getitem__(self, idx):
        input_file_idx = self.file_idx[idx]

        gedi_path = self.gedi_paths[input_file_idx]
        sentinel_path = self.sentinel_paths[input_file_idx]

        gedi = np.load(gedi_path)
        sentinel = np.load(sentinel_path)
        
        gedi = gedi.astype(np.float32)
        sentinel = sentinel.astype(np.float32)
        mangrove_mask = np.isfinite(sentinel).all(axis=0) & np.any(sentinel != 0, axis=0)
        sentinel = sentinel.transpose(1, 2, 0)  # from CHW to HWC for PyTorch

        # print(gedi.shape, sentinel.shape)


        transpose_sentinel = T.Compose([
                T.ToTensor(),
                # 12 bands
                # statistics on Cuc Phuong and Ba Be
                # ------------
                self.normalization()
            ])
        transpose_gedi = T.Compose([
            T.ToTensor()
        ])

        gedi = gedi / self.max_height
        sentinel = transpose_sentinel(sentinel)
        gedi = torch.from_numpy(gedi)

        # print(sentinel.shape, gedi.shape)
        # print(sentinel.dtype, gedi.dtype)

        # print("Sentinel", sentinel.min(), sentinel.max())
        # print("GEDI", gedi[~torch.isnan(gedi)].min(), gedi[~torch.isnan(gedi)].max())

        sample = {
            "image": sentinel,
            "zone": self.sample_regions[input_file_idx],
            "sample_id": self.sample_ids[input_file_idx],
            # sentinel[1:4, :, :],  # using RGB bands only
            "mask": gedi,
            "mangrove_mask": torch.from_numpy(mangrove_mask),
        }

        return sample

    def plot(self, image, mask, prediction=None, show_titles=True, mangrove_mask=None):
        if prediction is not None:
            prediction = prediction.clip(0, 1).float()
        # Restore reflectance before selecting display RGB channels.
        if image.shape[0] == 12:
            normalization = self.normalization()
            mean = image.new_tensor(normalization.mean)[:, None, None]
            std = image.new_tensor(normalization.std)[:, None, None]
            image = image * std + mean
        if mangrove_mask is None:
            mangrove_mask = torch.isfinite(image).all(dim=0) & (image.abs() > 1e-3).any(dim=0)
        mangrove_mask = mangrove_mask.bool()
        display_mask = torch.nn.functional.interpolate(
            mangrove_mask[None, None].float(), size=image.shape[-2:], mode="nearest"
        )[0, 0].bool()
        # Convert image to [0, 1] range
        if image.shape[0] == 12:
            image = image[[2, 1, 0], :, :]
        image = image.float()
        image = image - image.min()
        image = image / image.max().clamp(min=1e-8)
        image = image.masked_fill(~display_mask.unsqueeze(0), 1.0)

        mask = mask.float()

        showing_prediction = prediction is not None
        ncols = 2 + int(showing_prediction)
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].axis("off")
        
        im1 = axs[1].imshow(
            mask.squeeze(), interpolation="none", cmap="Spectral_r", vmin=0, vmax=1
        )
        axs[1].axis("off")
        cbar1 = plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        cbar1.set_label("Canopy Height (0-30m)", rotation=270, labelpad=15)
        
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_prediction:
            prediction_mask = torch.nn.functional.interpolate(
                mangrove_mask[None, None].float(), size=prediction.shape[-2:], mode="nearest"
            )[0, 0].bool()
            displayed_prediction = np.ma.masked_where(
                ~prediction_mask.cpu().numpy(), prediction.squeeze().cpu().numpy()
            )
            prediction_cmap = plt.get_cmap("Spectral_r").copy()
            prediction_cmap.set_bad("white")
            im2 = axs[2].imshow(
                displayed_prediction,
                interpolation="none",
                cmap=prediction_cmap,
                vmin=0,
                vmax=1,
            )
            axs[2].axis("off")
            cbar2 = plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
            cbar2.set_label("Canopy Height (0-30m)", rotation=270, labelpad=15)
            if show_titles:
                axs[2].set_title("Prediction")
        
        return fig

class GediSentinelDataModule(L.LightningDataModule):
    def __init__(
        self,
        regions,
        gedi_folder,
        sentinel_folder,
        batch_size=4,
        num_workers=4,
        ratio_train=0.8,
        all_train_data= False,
        max_height=30,
    ):
        super().__init__()

        self.max_height = max_height
        self.regions = regions
        self.gedi_folder = gedi_folder
        self.sentinel_folder = sentinel_folder

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ratio_train = ratio_train

        self.all_train_data = all_train_data

        print("All data in the train set:", all_train_data)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = GediSentinelDataset(
                regions=self.regions,
                max_height=self.max_height,
                gedi_folder=self.gedi_folder,
                sentinel_folder=self.sentinel_folder,
                mode="train",
                ratio_train=self.ratio_train,
                predict=False,
                all_train_data = self.all_train_data
            )

            self.val_dataset = GediSentinelDataset(
                regions=self.regions,
                max_height=self.max_height,
                gedi_folder=self.gedi_folder,
                sentinel_folder=self.sentinel_folder,
                mode="val",
                ratio_train=self.ratio_train,
                predict=False,
            )

        if stage == "test" or stage is None:
            self.test_dataset = GediSentinelDataset(
                regions=self.regions,
                max_height=self.max_height,
                gedi_folder=self.gedi_folder,
                sentinel_folder=self.sentinel_folder,
                mode="test",
                ratio_train=self.ratio_train,
                predict=False,
            )
        
        if stage == "predict":
            self.predict_dataset = GediSentinelDataset(
                regions=self.regions,
                max_height=self.max_height,
                gedi_folder=self.gedi_folder,
                sentinel_folder=self.sentinel_folder,
                predict=True,       # include all patches
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
            drop_last=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )