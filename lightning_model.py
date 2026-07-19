import argparse
import logging
import os
import pprint
import random
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import transformers
from kornia.geometry.transform import resize
from lightning import LightningModule
from model import DepthAnythingV2
from torch import nn
from torchmetrics import MetricCollection, classification, regression


class DepthAnythingV2Module(LightningModule):
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    size_map = {
        # "vits": "depth-anything/Depth-Anything-V2-Small-hf",
        "vits": "DarthReca/depth-any-canopy-small", 
        "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
        "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
        "vitg": None,
    }

    def __init__(
        self,
        encoder: Literal["vits", "vitb", "vitl", "vitg"],
        pretrained_from: Literal["depth-anything", "depth-anycanopy"],
        min_depth: float = 1e-4,
        max_depth: float = 30.0,
        lr: float = 0.000005,
        use_huggingface: bool = False,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if pretrained:
            if not use_huggingface:
                pretrained_from = f"base-checkpoints/{encoder}.pth"
                self.model = DepthAnythingV2(**{**self.model_configs[encoder]})
                # self.model.load_state_dict(
                #     {
                #         k: v
                #         for k, v in torch.load(pretrained_from, map_location="cpu").items()
                #         if "pretrained" in k
                #     },
                #     strict=False,
                # )
            else:
                if pretrained_from == "depth-anycanopy":
                    print("Loading model from Hugging Face: {}".format(self.size_map[encoder]))
                    self.model = transformers.AutoModelForDepthEstimation.from_pretrained(
                        self.size_map[encoder], cache_dir="cache"
                    ).train()
                elif pretrained_from == "depth-anything":
                    print("Loading model from Hugging Face: {}".format("depth-anything/Depth-Anything-V2-Small-hf"))
                    self.model = transformers.AutoModelForDepthEstimation.from_pretrained(
                        "depth-anything/Depth-Anything-V2-Small-hf", cache_dir="cache"
                    ).train()
        else:
            config = transformers.AutoConfig.from_pretrained(
                self.size_map[encoder],
                cache_dir="cache"
            )

            self.model = transformers.AutoModelForDepthEstimation.from_config(config).train()

            print(config)

        self.loss = nn.MSELoss()
        
        # Training metrics
        self.train_metric = MetricCollection(
            {
                "MSE": regression.MeanSquaredError(),
                "MAE": regression.MeanAbsoluteError(),
            },
            prefix="train_"
        )
        
        # Validation metrics
        self.val_metric = MetricCollection(
            {
                "MSE": regression.MeanSquaredError(),
                "MAE": regression.MeanAbsoluteError(),
            },
            prefix="val_"
        )
        
        self.classification_metrics = MetricCollection(
            [classification.JaccardIndex(task="binary")]
        )
        self.corr = MetricCollection([regression.PearsonCorrCoef()])
        self.predictions = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=self.hparams.lr,
            pct_start=0.05,
            cycle_momentum=False,
            div_factor=1e9,
            final_div_factor=1e4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        img, depth = self._preprocess_batch(batch)

        pred = self.model(img)
        # .predicted_depth
        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 1)

        valid_mask = ~torch.isnan(depth)
        num_valid = valid_mask.sum().item()

        loss = self.loss(
            pred[valid_mask],
            depth[valid_mask]
        )

        # Compute and log training metrics with custom names
        self.train_metric(pred[valid_mask], depth[valid_mask])
        
        # Log loss
        self.log("train_Loss(MSE)", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Log metrics with RMSE
        metrics = self.train_metric.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        # Manually compute and log RMSE
        rmse = torch.sqrt(metrics.get("train_MSE", torch.tensor(0.0)))
        self.log("train_RMSE", rmse, on_step=False, on_epoch=True, prog_bar=True)
        
        self.train_metric.reset()
        
        return loss


    def validation_step(self, batch, batch_idx):
        img, depth = self._preprocess_batch(batch)

        # pred = self.model(img).predicted_depth
        pred = self.model(img)
        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 30)

        valid_mask = ~torch.isnan(depth)
        num_valid = valid_mask.sum().item()

        loss = self.loss(
            pred[valid_mask],
            depth[valid_mask]
        )

        # Compute and log validation metrics with custom names
        self.val_metric(pred[valid_mask], depth[valid_mask])
        
        # Log loss with on_step=True to show per-batch updates
        self.log("val_Loss(MSE)", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Log metrics
        metrics = self.val_metric.compute()
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)
        
        # Manually compute and log RMSE
        rmse = torch.sqrt(metrics.get("val_MSE", torch.tensor(0.0)))
        self.log("val_RMSE", rmse, prog_bar=True, on_step=True, on_epoch=True)
        
        self.val_metric.reset()

        # print("Current epoch ", self.current_epoch, "Number of images", len(img), type(img), img.shape)
        # if batch_idx < 10 and self.logger is not None:
        if self.current_epoch % 20 == 0 and self.logger is not None:
            for i in range(img.shape[0]):
                fig = self.trainer.datamodule.val_dataset.plot(
                    img[i].cpu().detach(), depth[i].cpu().detach(), pred[i].cpu().detach()
                )
                self.logger.experiment.log_figure(
                    figure=fig, figure_name=f"val_{batch_idx*img.shape[0] + i}"
                )

                os.makedirs(f"validation_predictions/epoch_{self.current_epoch}", exist_ok=True)
                fig.savefig(
                    f"validation_predictions/epoch_{self.current_epoch}/{batch_idx*img.shape[0] + i}.png"
                )
                plt.close(fig)

        return loss    

    def test_step(self, batch, batch_idx):
        img, depth = self._preprocess_batch(batch)

        pred = self.model(img)
        # .predicted_depth

        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 1)

        valid_mask = ~torch.isnan(depth)
        num_valid = valid_mask.sum().item()
        # self.print(f"Valid pixels: {num_valid}")

        self.val_metric(pred[valid_mask], depth[valid_mask])
        metrics = self.val_metric.compute()
        self.log_dict(metrics)
        
        rmse = torch.sqrt(metrics.get("val_MSE", torch.tensor(0.0)))
        self.log("val_RMSE", rmse)

        self.classification_metrics(pred > 1e-4, depth > 1e-4)
        self.log_dict(self.classification_metrics)

        self.predictions.append(
            {
                "prediction": pred[depth > 1e-4].flatten().detach().cpu(),
                "depth": depth[depth > 1e-4].flatten().detach().cpu(),
            }
        )



    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # print(f"Processing batch {batch_idx}")
        img, depth = self._preprocess_batch(batch)

        pred = self.model(img).predicted_depth

        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 1)

        for i in range(img.shape[0]):
            fig = self.trainer.datamodule.val_dataset.plot(
                img[i].cpu().detach(), depth[i].cpu().detach(), pred[i].cpu().detach()
            )
            self.logger.experiment.log_figure(
                figure=fig, figure_name=f"val_{batch_idx*img.shape[0] + i}"
            )

            # base_dir = "inference_predictions"
            save_dir = "inference_predictions"

            # counter = 1
            # while os.path.exists(save_dir):
            #     save_dir = f"{base_dir}_{counter}"
            #     counter += 1

            os.makedirs(f"{save_dir}/image", exist_ok=True)
            os.makedirs(f"{save_dir}/np_array", exist_ok=True)
            fig.savefig(
                # f"{save_dir}/image/{batch_idx * img.shape[0] + i}.png"
                f"{save_dir}/image/{batch_idx * 4 + i}.png"
            )
            np.save(
                f"{save_dir}/np_array/{batch_idx * 4 + i}.npy",
                pred[i].cpu().detach().numpy()
            )
            plt.close(fig)

            # print("Batch ", batch_idx, "Image ", i, " saved to ", f"{save_dir}/image/{batch_idx * img.shape[0] + i}.png")

        return pred

    def _preprocess_batch(self, batch):
        img, depth = batch["image"], batch["mask"]

        img = resize(img, (518, 518), interpolation="bilinear")
        
        depth = torch.clamp(
            depth, min=self.hparams.min_depth, max=self.hparams.max_depth
        )

        return img, depth
