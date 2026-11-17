"""Save epoch metrics and training plots independently of Comet."""
import csv
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class TrainingCurves(Callback):
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.rows = []

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Training curves: {self.output_dir.resolve()}", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        row = {"epoch": trainer.current_epoch + 1, "step": trainer.global_step}
        for stage in ("train", "val"):
            for metric in ("Loss(MSE)", "MAE", "RMSE"):
                key = f"{stage}_{metric}"
                value = trainer.callback_metrics.get(key + "_epoch")
                if value is None:
                    value = trainer.callback_metrics.get(key)
                if value is not None:
                    # All ranks participate, including non-writing workers.
                    value = trainer.strategy.reduce(value.detach().clone(), reduce_op="mean")
                    row[key] = value.cpu().item()
        row["lr"] = trainer.optimizers[0].param_groups[0]["lr"]
        if trainer.is_global_zero:
            self.rows.append(row)
            self._save()

    def _save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fields = ["epoch", "step", "train_Loss(MSE)", "val_Loss(MSE)",
                  "train_MAE", "val_MAE", "train_RMSE", "val_RMSE", "lr"]
        with (self.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        fig = Figure(figsize=(12, 8), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2).ravel()
        epochs = [row["epoch"] for row in self.rows]
        for ax, metric, title in zip(axes, ("Loss(MSE)", "MAE", "RMSE"), ("MSE loss", "MAE", "RMSE")):
            for stage in ("train", "val"):
                key = f"{stage}_{metric}"
                if any(key in row for row in self.rows):
                    ax.plot(epochs, [row.get(key, float("nan")) for row in self.rows], label=stage)
            ax.set(title=title, xlabel="Epoch", ylabel="Normalized units squared" if metric == "Loss(MSE)" else "Normalized height units")
            if ax.lines:
                ax.legend()
            ax.grid(alpha=0.3)
        axes[3].plot(epochs, [row["lr"] for row in self.rows])
        axes[3].set(title="Learning rate (end of epoch)", xlabel="Epoch", ylabel="Learning rate")
        axes[3].grid(alpha=0.3)
        fig.suptitle("Training history (epoch batch averages)")
        fig.savefig(self.output_dir / "training_curves.png", dpi=160)
        fig.clear()
