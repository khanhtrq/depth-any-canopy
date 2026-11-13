"""Compare peak OneCycle learning rates using the existing Hydra configuration.

Example:
    python lr_experiments.py --learning-rates 1e-6 5e-6 1e-5 --epochs 50
    python lr_experiments.py --override dataset.gedi_folder=/path/to/GEDI

Losses use normalized heights; RMSE is also reported in metres. This runner
uses validation for selection, without evaluating or predicting the test set.
"""
import argparse
import csv
import gc
import json
import math
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightning as pl
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

from dataset.sentinel_gedi_canopy import GediSentinelDataModule
from lightning_model import DepthAnythingV2Module


def write_csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def plot_curves(path, histories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, rows in histories.items():
        for ax, key in zip(axes, ("train_mse", "val_mse")):
            ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], label=label)
    for ax, title in zip(axes, ("Training", "Validation")):
        ax.set(title=title, xlabel="Epoch", ylabel="MSE (normalized height)")
        ax.grid(alpha=0.3)
        if histories:
            ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


class EpochResults(Callback):
    """Aggregate batch MSE by number of observed GEDI pixels, not batch size."""
    def __init__(self, folder):
        self.folder = folder
        self.rows = []
        self.best = math.inf
        self.best_epoch = None
        self.train_total = self.train_count = 0
        self.val_total = self.val_count = 0

    def on_train_epoch_start(self, trainer, module):
        self.train_total = self.train_count = 0

    def on_validation_epoch_start(self, trainer, module):
        self.val_total = self.val_count = 0

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        count = (~torch.isnan(batch["mask"])).sum().item()
        if count:
            self.train_total += float(loss.detach()) * count
            self.train_count += count

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        count = (~torch.isnan(batch["mask"])).sum().item()
        if count:
            self.val_total += float(outputs.detach()) * count
            self.val_count += count

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        if not self.train_count or not self.val_count:
            raise RuntimeError("Training and validation must both contain valid GEDI pixels.")
        train_mse = self.train_total / self.train_count
        val_mse = self.val_total / self.val_count
        if not math.isfinite(train_mse) or not math.isfinite(val_mse):
            raise RuntimeError("Non-finite loss; inspect inputs and the learning rate.")
        row = {
            "epoch": trainer.current_epoch + 1,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_rmse": math.sqrt(train_mse),
            "val_rmse": math.sqrt(val_mse),
            "learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
        }
        self.rows.append(row)
        write_csv(self.folder / "epochs.csv", self.rows)
        plot_curves(self.folder / "learning_curves.png", {"loss": self.rows})
        if val_mse < self.best:
            self.best = val_mse
            self.best_epoch = row["epoch"]
            trainer.save_checkpoint(self.folder / "best.ckpt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-6, 3e-6, 5e-6, 1e-5, 3e-5])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("lr_experiments"))
    parser.add_argument("--override", action="append", default=[], help="Hydra override; repeat as needed")
    args = parser.parse_args()
    if any(not math.isfinite(lr) or lr <= 0 for lr in args.learning_rates):
        parser.error("Learning rates must be positive and finite.")
    if args.epochs is not None and args.epochs < 1:
        parser.error("Epochs must be positive.")
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parent / "configs"), version_base=None):
        config = compose(config_name="default", overrides=args.override)
    if config.dataset.all_train_data:
        parser.error("Set dataset.all_train_data=false to keep validation separate from training.")
    folder = args.output.resolve() / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True)
    (folder / "arguments.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    summaries, histories = [], {}
    torch.set_float32_matmul_precision("medium")
    for index, lr in enumerate(args.learning_rates):
        run_folder = folder / f"{index + 1:02d}_lr_{lr:g}"
        run_folder.mkdir()
        run_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        run_config.model.lr = lr
        if args.epochs is not None:
            run_config.trainer.max_epochs = args.epochs
        OmegaConf.save(run_config, run_folder / "config.yaml")
        result = {"learning_rate": lr, "seed": args.seed, "status": "failed", "run_directory": str(run_folder)}
        started = time.monotonic()
        trainer = model = data = None
        recorder = EpochResults(run_folder)
        try:
            pl.seed_everything(args.seed, workers=True)
            data = GediSentinelDataModule(
                regions=run_config.dataset.regions,
                gedi_folder=run_config.dataset.gedi_folder,
                sentinel_folder=run_config.dataset.sentinel_folder,
                all_train_data=False,
                batch_size=run_config.dataset.get("batch_size", 4),
                num_workers=run_config.dataset.get("num_workers", 4),
                ratio_train=run_config.dataset.get("ratio_train", 0.8),
            )
            model = DepthAnythingV2Module(**run_config.model)
            trainer_options = OmegaConf.to_container(run_config.trainer, resolve=True)
            trainer_options.update(logger=False, callbacks=[recorder], enable_checkpointing=False,
                                   num_sanity_val_steps=0, enable_model_summary=False)
            # Each candidate must run on one device so aggregation and output are unambiguous.
            trainer_options.update(devices=1, num_nodes=1, strategy="auto")
            trainer = pl.Trainer(**trainer_options)
            trainer.fit(model, datamodule=data)
            if not recorder.rows:
                raise RuntimeError("No validation epochs completed.")
            result.update(status="completed", best_epoch=recorder.best_epoch,
                          best_val_mse=recorder.best,
                          best_val_rmse=math.sqrt(recorder.best),
                          best_val_rmse_m=math.sqrt(recorder.best) * data.val_dataset.max_height,
                          final_train_mse=recorder.rows[-1]["train_mse"],
                          final_val_mse=recorder.rows[-1]["val_mse"],
                          epochs_completed=len(recorder.rows),
                          checkpoint=str(run_folder / "best.ckpt"))
        except Exception as error:
            result["error"] = str(error)
            (run_folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"LR {lr:g} failed: {error}")
        finally:
            result["duration_seconds"] = round(time.monotonic() - started, 2)
            summaries.append(result)
            ranked = sorted(summaries, key=lambda row: row.get("best_val_mse", math.inf))
            write_csv(folder / "summary.csv", ranked)
            if recorder.rows:
                histories[f"{index + 1}: peak LR {lr:g}"] = recorder.rows
                plot_curves(folder / "comparison_curves.png", histories)
            del trainer, model, data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"Results: {folder}")
    completed = [r for r in summaries if r["status"] == "completed"]
    if completed:
        best = min(completed, key=lambda row: row["best_val_mse"])
        print(f"Best peak LR: {best['learning_rate']:g}; validation RMSE: {best['best_val_rmse_m']:.4f} m")
    else:
        raise SystemExit("All experiments failed; see each run's error.txt.")


if __name__ == "__main__":
    main()
