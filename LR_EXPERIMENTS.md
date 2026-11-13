# Learning-rate experiments

Run in the same Python environment used for `main.py`:

```bash
python lr_experiments.py --learning-rates 1e-6 3e-6 5e-6 1e-5 3e-5 --epochs 50
```

The default dataset paths target Kaggle. Override them when running elsewhere:

```bash
python lr_experiments.py --learning-rates 1e-6 5e-6 1e-5 --epochs 50 --override dataset.gedi_folder=/path/to/GEDI --override dataset.sentinel_folder=/path/to/Sentinel --override dataset.num_workers=0
```

Use quoted overrides if paths contain spaces. Repeat `--override` for other existing Hydra configuration options. Defaults come from `configs/default.yaml`; omit `--epochs` to use the configured epoch count. `--seed` defaults to 42. `--output` changes the parent output directory.

Each invocation creates a new timestamped directory under `lr_experiments/`:

- `summary.csv`: completed runs ranked by best validation MSE, followed by failed runs. Includes peak LR, seed, best epoch, best validation MSE/RMSE, validation RMSE in metres, final losses, duration, and checkpoint path.
- `comparison_curves.png`: overlaid training and validation MSE curves for all candidates.
- Per candidate: `epochs.csv`, `learning_curves.png`, `config.yaml`, and `best.ckpt`. Failed runs have `error.txt`; completed epochs remain available.
- `arguments.json`: invocation settings for later reference.

Each candidate starts from a fresh model with the same seed and deterministic dataset split. All candidates use one device, the same epoch budget, and the existing OneCycleLR schedule: the candidate LR is its **peak**, not a constant learning rate. Comet logging, prediction-image generation, and test-set evaluation are disabled. Run on a single process; distributed launchers are not supported.

Epoch MSE is weighted by observed GEDI pixel count. RMSE is the square root of this MSE. Heights are normalized by the dataset maximum height (currently 30 m); the summary also converts RMSE to metres. Training curves aggregate losses observed while weights change during an epoch. Validation curves evaluate the weights at the end of that epoch. Selection uses validation only; no independent test performance is claimed. `all_train_data=true` is rejected to prevent overlap with validation.

The runner preserves current model behavior, including training prediction clamping to [0, 1] and validation prediction clamping to [0, 30]. This difference can affect comparisons between training and validation curves. Sparse GEDI supervision and existing mangrove masking are also preserved. A single seed gives an initial comparison; repeat with other seeds to assess variability.
