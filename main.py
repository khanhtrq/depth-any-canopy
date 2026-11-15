import time
import os
import sys

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
from omegaconf import DictConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record
from distributed_runtime import launch_workers, evaluation_checkpoint_paths
from prediction_map import merge_region_predictions


@hydra.main(config_path="configs", config_name="default", version_base=None)
@record
def main(args: DictConfig):
    driver = launch_workers(
        OmegaConf.to_container(args.trainer, resolve=True), torch.cuda.device_count(),
        __file__, sys.argv[1:],
    )
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    
    print("Starting")

    print(args)
    data_module = GediSentinelDataModule(all_train_data = args.dataset.all_train_data,
                                         gedi_folder = args.dataset.gedi_folder,
                                         sentinel_folder = args.dataset.sentinel_folder,
                                         regions = args.dataset.regions,
                                         batch_size=args.dataset.batch_size,
                                         num_workers=args.dataset.num_workers,
                                         ratio_train=args.dataset.ratio_train,
                                         max_height=args.dataset.max_height,
                                         prediction_layouts=args.dataset.get("prediction_layouts", {}))
    if driver:
        # All GPU processes have exited; large CPU maps cannot block DDP teardown.
        data_module.setup("predict")
        merge_region_predictions(data_module.predict_dataset, data_module.prediction_layouts)
        print("Prediction completed", flush=True)
        return
    model = DepthAnythingV2Module(**args.model)
    model.defer_map_merge = os.environ.get("CANOPY_DEFER_MAP_MERGE") == "1"

    experiment_id = os.environ.get("CANOPY_RUN_ID", time.strftime("%Y%m%d-%H%M%S"))
    logger = False
    if args.logger:
        logger = CometLogger(
            project_name="depth-any-canopy",
            workspace="",
            experiment_name="",
            save_dir="comet-logs",
            offline=False,
            auto_output_logging=False,
        )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_Loss(MSE)",
        dirpath=f"checkpoints/{experiment_id}",
        filename="depth-any-canopy-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        save_last=True,
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

    checkpoint_path = args.get("checkpoint_path")
    if checkpoint_path:
        print(f"Skipping training; evaluating and predicting from {checkpoint_path}", flush=True)
    else:
        print("Fitting to trainer", flush=True)
        trainer.fit(model, datamodule=data_module)

    best_path, last_path = evaluation_checkpoint_paths(
        trainer, checkpoint_callback, checkpoint_path
    )

    print("Testing best model", flush=True)
    model.evaluation_label = "best"
    trainer.test(model, datamodule=data_module, ckpt_path=best_path)
    print("Testing last model", flush=True)
    model.evaluation_label = "last"
    trainer.test(model, datamodule=data_module, ckpt_path=last_path)

    print(f"Prediction: output directory = {os.path.abspath('inference_predictions')}", flush=True)
    trainer.predict(model, datamodule=data_module, ckpt_path=last_path, return_predictions=False)
    if trainer.is_global_zero:
        message = "Prediction patches completed; driver will merge maps" if model.defer_map_merge else "Prediction completed"
        print(message, flush=True)

if __name__ == "__main__":
    main()
