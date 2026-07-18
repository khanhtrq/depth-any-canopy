import time

import comet_ml
import hydra
import lightning as pl
import torch
from dataset import EarthViewNEONDatamodule, DummyDatamodule, GediSentinelDataModule
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CometLogger
from lightning_model import DepthAnythingV2Module
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(args: DictConfig):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    
    print("Starting")

    print(args)
    data_module = GediSentinelDataModule(all_train_data = args.dataset.all_train_data,
                                         gedi_folder = args.dataset.gedi_folder,
                                         sentinel_folder = args.dataset.sentinel_folder,
                                         regions = args.dataset.regions)
    model = DepthAnythingV2Module(**args.model)

    experiment_id = time.strftime("%Y%m%d-%H%M%S")
    logger = False
    if args.logger:
        logger = CometLogger(
            project_name="depth-any-canopy",
            workspace="",
            experiment_name="",
            save_dir="comet-logs",
            offline=False,
        )
        experiment_id = logger.experiment.id

    checkpoint_callback = ModelCheckpoint(
        monitor="val_Loss(MSE)",
        dirpath=f"checkpoints/{experiment_id}",
        filename="depth-any-canopy-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_MAE", patience=20, mode="min", verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(refresh_rate=10)

    # callback = [checkpoint_callback, early_stopping, progress_bar]
    # No early stopping 
    callback = [checkpoint_callback, progress_bar]
    if logger:
        callback.append(lr_monitor)

    trainer = pl.Trainer(
        **args.trainer,
        logger=logger,
        callbacks=callback,
        log_every_n_steps=50,
        precision="32-true" if args.model.encoder == "vitl" else "32-true",
        limit_val_batches=1.0,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=False
    )

    print("Fitting to trainer")

    trainer.fit(model, datamodule=data_module)

    print("Testing best model")
    trainer.test(model, datamodule=data_module, ckpt_path="best")
    print("Testing last model")
    trainer.test(model, datamodule=data_module, ckpt_path="last")

    print("Prediction")
    # trainer.predict(model, datamodule=data_module, ckpt_path="best")
    trainer.predict(model, datamodule=data_module, ckpt_path="last")

if __name__ == "__main__":
    main()
