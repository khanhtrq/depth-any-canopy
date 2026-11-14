import json
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

from safetensors.torch import load_file
from prediction_map import SOURCE_SHAPES, default_layout, patch_position, save_region_map


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

        self.use_huggingface = use_huggingface

        if pretrained:
            if not use_huggingface:
                # pretrained_from = f"base-checkpoints/{encoder}.pth"
                pretrained_from = "/kaggle/input/datasets/khanhtq2101/gedi-canopy-height-hoanglien/depth_any_canopy_small.safetensors"
                self.model = DepthAnythingV2(**{**self.model_configs[encoder]})

                state_dict = load_file(pretrained_from)
                self.model.load_state_dict(
                    {
                        k: v
                        for k, v in state_dict.items()
                        if "pretrained" in k
                    },
                    strict=False,
                )
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

            # self.model = transformers.AutoModelForDepthEstimation.from_config(config).train()
            self.model = DepthAnythingV2(**{**self.model_configs[encoder]})

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

        if self.use_huggingface:
            # For Hugging Face models, the output is a dictionary with 'predicted_depth' key
            pred = self.model(img).predicted_depth
        else:
            pred = self.model(img)

        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 1)
        if "mangrove_mask" in batch:
            mangrove_mask = batch["mangrove_mask"].to(device=pred.device, dtype=torch.bool)
            if pred.ndim == 4:
                mangrove_mask = mangrove_mask.unsqueeze(1)
            pred = pred.masked_fill(~mangrove_mask, 0)

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

        if self.use_huggingface:
            # For Hugging Face models, the output is a dictionary with 'predicted_depth' key
            pred = self.model(img).predicted_depth
        else:
            pred = self.model(img)

        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 30)
        if "mangrove_mask" in batch:
            mangrove_mask = batch["mangrove_mask"].to(device=pred.device, dtype=torch.bool)
            if pred.ndim == 4:
                mangrove_mask = mangrove_mask.unsqueeze(1)
            pred = pred.masked_fill(~mangrove_mask, 0)

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
                    img[i].cpu().detach(), depth[i].cpu().detach(), pred[i].cpu().detach(),
                mangrove_mask=batch["mangrove_mask"][i].cpu() if "mangrove_mask" in batch else None,
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

        if self.use_huggingface:
            # For Hugging Face models, the output is a dictionary with 'predicted_depth' key
            pred = self.model(img).predicted_depth
        else:
            pred = self.model(img)


        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 1)
        if "mangrove_mask" in batch:
            mangrove_mask = batch["mangrove_mask"].to(device=pred.device, dtype=torch.bool)
            if pred.ndim == 4:
                mangrove_mask = mangrove_mask.unsqueeze(1)
            pred = pred.masked_fill(~mangrove_mask, 0)

        valid_mask = ~torch.isnan(depth)
        num_valid = valid_mask.sum().item()
        # self.print(f"Valid pixels: {num_valid}")

        # Keep per-patch sufficient statistics so DDP sampler padding can be
        # removed before reporting exact per-zone, per-valid-pixel metrics.
        for i, zone in enumerate(batch["zone"]):
            valid = torch.isfinite(depth[i])
            error = (pred[i][valid] - depth[i][valid]).double()
            self.test_records[(zone, batch["sample_id"][i])] = (
                error.numel(), error.abs().sum().item(), error.square().sum().item()
            )

    def on_test_epoch_start(self):
        self.test_records = {}

    def on_test_epoch_end(self):
        records = self.test_records
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, records)
            records = {key: value for part in gathered for key, value in part.items()}
        totals = {}
        for (zone, _), (count, absolute, squared) in records.items():
            total = totals.setdefault(zone, [0, 0.0, 0.0, 0])
            total[0] += count
            total[1] += absolute
            total[2] += squared
            total[3] += 1
        overall = [sum(values[i] for values in totals.values()) for i in range(4)]
        results = {"zones": {}, "overall": {}}
        scale = self.trainer.datamodule.max_height
        for zone, (count, absolute, squared, patches) in list(totals.items()) + [("overall", overall)]:
            metrics = {"patches": patches, "valid_pixels": count,
                       "MAE_m": absolute / count * scale if count else None,
                       "MSE_m2": squared / count * scale ** 2 if count else None,
                       "RMSE_m": (squared / count) ** 0.5 * scale if count else None}
            if zone == "overall":
                results["overall"] = metrics
            else:
                results["zones"][zone] = metrics
            for name in ("MAE_m", "MSE_m2", "RMSE_m"):
                if metrics[name] is not None:
                    self.log(f"test/{zone}/{name}", metrics[name], sync_dist=False)
        if self.trainer.is_global_zero:
            os.makedirs("evaluation_results", exist_ok=True)
            label = getattr(self, "evaluation_label", "test")
            with open(f"evaluation_results/{label}.json", "w") as handle:
                json.dump(results, handle, indent=2)
            print(json.dumps(results, indent=2))
        self.test_records.clear()

    def on_predict_start(self):
        dataset = self.trainer.datamodule.predict_dataset
        configured = self.trainer.datamodule.prediction_layouts
        self.prediction_layouts = {}
        for zone in dataset.regions:
            layout = configured.get(zone)
            if layout is None and zone in SOURCE_SHAPES:
                i = next(i for i in dataset.file_idx if dataset.sample_regions[i] == zone)
                shape = np.load(dataset.gedi_paths[i], mmap_mode="r").shape[-2:]
                layout = default_layout(zone, shape)
            if layout is None:
                raise ValueError(
                    f"Set dataset.prediction_layouts.{zone}={{rows: <rows>, columns: <columns>}} "
                    "to the original patch grid; numeric IDs alone do not encode its width."
                )
            for i in dataset.file_idx:
                if dataset.sample_regions[i] == zone:
                    patch_position(dataset.sample_ids[i], layout)
            self.prediction_layouts[zone] = layout

    def on_predict_epoch_end(self):
        # All ranks must finish writing their patches before rank zero merges them.
        self.trainer.strategy.barrier()
        if self.trainer.is_global_zero:
            dataset = self.trainer.datamodule.predict_dataset
            for zone, layout in self.prediction_layouts.items():
                output_dir = os.path.join("inference_predictions", zone)
                save_region_map(dataset, zone, layout, output_dir)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # print(f"Processing batch {batch_idx}")
        img, depth = self._preprocess_batch(batch)

        if self.use_huggingface:
            # For Hugging Face models, the output is a dictionary with 'predicted_depth' key
            pred = self.model(img).predicted_depth
        else:
            pred = self.model(img)

        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 1)
        if "mangrove_mask" in batch:
            mangrove_mask = batch["mangrove_mask"].to(device=pred.device, dtype=torch.bool)
            if pred.ndim == 4:
                mangrove_mask = mangrove_mask.unsqueeze(1)
            pred = pred.masked_fill(~mangrove_mask, 0)

        for i in range(img.shape[0]):
            fig = self.trainer.datamodule.predict_dataset.plot(
                img[i].cpu().detach(), depth[i].cpu().detach(), pred[i].cpu().detach(),
                mangrove_mask=batch["mangrove_mask"][i].cpu() if "mangrove_mask" in batch else None,
            )
            zone = batch["zone"][i]
            sample_id = batch["sample_id"][i]
            if self.logger is not None:
                self.logger.experiment.log_figure(
                    figure=fig, figure_name=f"predict/{zone}/{sample_id}"
                )
            save_dir = os.path.join("inference_predictions", zone)
            os.makedirs(os.path.join(save_dir, "image"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "np_array"), exist_ok=True)
            fig.savefig(os.path.join(save_dir, "image", f"{sample_id}.png"))
            np.save(os.path.join(save_dir, "np_array", f"{sample_id}.npy"),
                    pred[i].cpu().detach().numpy())
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
