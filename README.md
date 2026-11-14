<div align="center">
  
## Depth Any Canopy: Leveraging Depth Foundation Models for Canopy Height Estimation

[**Daniele Rege Cambrin**](https://darthreca.github.io/)<sup>1</sup> · [**Isaac Corley**](https://isaacc.dev/)<sup>2</sup> · [**Paolo Garza**](https://dbdmg.polito.it/dbdmg_web/people/paolo-garza/)<sup>1</sup>

<sup>1</sup>Politecnico di Torino, Italy&emsp;&emsp;&emsp;&emsp;<sup>2</sup>University of Texas at San Antonio, USA

**[ECCV 2024 CV4E Workshop](https://cv4e.netlify.app/)**

<a href="https://arxiv.org/abs/2408.04523"><img src='https://img.shields.io/badge/arXiv-Depth%20Any%20Canopy-red' alt='Paper PDF'></a>
<a href='https://huggingface.co/DarthReca/depth-any-canopy-small'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Small%20Version-yellow'></a>
</div>

**In this paper, we propose transferring the representations learned by recent depth estimation foundation models to the remote sensing domain for measuring canopy height.** Our findings suggest that our proposed Depth Any Canopy, the result of fine-tuning the Depth Anything v2 model for canopy height estimation, provides a performant and efficient solution, surpassing the current state-of-the-art with superior or comparable performance using only a fraction of the computational resources and parameters. Furthermore, our approach requires less than \$1.30 in compute and results in an estimated carbon footprint of 0.14 kgCO2.

### Getting Started

Install the dependencies of the *requirements.txt* file. Make sure to edit the config files in the `configs/` folder. Then simply run *main.py*

### Pre-Trained Models

Pre-trained checkpoints are available on HuggingFace.

| Model | Parameters | Checkpoint | 
|:---|:---:|:---:|
| Depth-Any-Canopy-Small | 24.8M | [Download](https://huggingface.co/DarthReca/depth-any-canopy-small) |
| Depth-Any-Canopy-Base  | 97.5M | [Download](https://huggingface.co/DarthReca/depth-any-canopy-base) |

You can easily load them with *pipelines* or *AutoModel*:

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("depth-estimation", model="DarthReca/depth-any-canopy-base")

# Load model directly
from transformers import AutoModelForDepthEstimation

model = AutoModelForDepthEstimation.from_pretrained("DarthReca/depth-any-canopy-base")
```

## License

This project is licensed under the **Apache 2.0 license**. See [LICENSE](LICENSE) for more information.

## Citation

If you find this project useful, please consider citing:

```bibtex
@inbook{RegeCambrin2025,
  title = {Depth Any Canopy: Leveraging Depth Foundation Models for Canopy Height Estimation},
  ISBN = {9783031923876},
  ISSN = {1611-3349},
  url = {http://dx.doi.org/10.1007/978-3-031-92387-6_5},
  DOI = {10.1007/978-3-031-92387-6_5},
  booktitle = {Computer Vision – ECCV 2024 Workshops},
  publisher = {Springer Nature Switzerland},
  author = {Rege Cambrin,  Daniele and Corley,  Isaac and Garza,  Paolo},
  year = {2025},
  pages = {71–86}
}
```


## Training one model across zones

Set `dataset.regions` to a list and supply a folder mapping when each zone has
its own Kaggle input directory. Each folder value is the directory containing
that zone's subdirectory (e.g. `GEDI_zone4/zone4/*.npy`). GEDI and Sentinel
patch filenames must match within each zone.

```bash
python main.py model.encoder=vits model.pretrained_from=depth-anycanopy model.pretrained=true \
  trainer.max_epochs=100 model.lr=5e-6 dataset.all_train_data=false \
  'dataset.regions=[zone3,zone4]' dataset.ratio_train=0.8 \
  'dataset.gedi_folder={zone3:/kaggle/input/datasets/khanhtq2101/gedi-canopy-height-hoanglien/zone3/zone3/GEDI_zone3,zone4:/kaggle/input/datasets/khanhtq2101/gedi-canopy-height-hoanglien/zone4/zone4/GEDI_zone4}' \
  'dataset.sentinel_folder={zone3:/kaggle/input/datasets/khanhtq2101/gedi-canopy-height-hoanglien/zone3/zone3/Sentinel_zone3_duplicated,zone4:/kaggle/input/datasets/khanhtq2101/gedi-canopy-height-hoanglien/zone4/zone4/Sentinel_zone4_duplicated}' \
  dataset.max_height=30 trainer.devices=2 \
  +trainer.strategy=ddp_find_unused_parameters_true model.use_huggingface=false
```

Use your actual zone paths. In a notebook, prefix `python` with `!`.
If zones share parent directories, the existing string folder settings also work.
The legacy `dataset.regions=zone3-zone4` spelling remains supported.

Splits use seed 42 independently within each zone after excluding patches with no
finite GEDI labels. Each zone contributes `floor(ratio_train * valid_patches)`
training patches and the remaining patches for evaluation. This preserves the
same ratio (up to integer rounding), not the same number of samples per zone.
Validation and testing share the held-out partition, as in the original workflow;
a separate untouched test set is not created. Keep `all_train_data=false` for
disjoint training and evaluation samples.

Outputs (relative to the run's working directory):

- `evaluation_results/best.json` and `last.json`: per-zone and overall MAE (m),
  MSE (m?), RMSE (m), valid pixel counts, and patch counts. Metrics weight each
  valid GEDI pixel equally and remove duplicate patches from DDP sampler padding.
- `inference_predictions/<zone>/image/<patch_id>.png`: comparison figures.
- `inference_predictions/<zone>/np_array/<patch_id>.npy`: predictions normalized
  by `dataset.max_height`; multiply by that value to convert to meters.
- `inference_predictions/<zone>/map.npy`: merged 2D prediction map, using the
  same normalized units as the patches. Missing grid cells are NaN; pixels
  outside the Sentinel mangrove mask retain the patch value of zero.
- `inference_predictions/<zone>/map.png`: merged heatmap with the patches'
  `Spectral_r` color scale and white pixels outside the mangrove mask.
- `inference_predictions/<zone>/rgb.png`: separate RGB mosaic visualization
  using Sentinel bands 2, 1, 0, per-patch min/max stretching like the patch
  figures, and white pixels outside the mangrove mask.

RGB and heatmap visualizations use previews with a maximum side length of 4096
pixels to keep rendering large regions within RAM limits. `map.npy` retains
every predicted pixel at the original patch resolution.

The preprocessing notebook numbers patches left to right, then top to bottom,
starting at zero (`id = row * columns + column`). Reconstruction preserves
these positions even when patch IDs are missing. Only complete patches were
exported by preprocessing, so discarded source-image edges cannot be recovered.
The notebook's source sizes supply defaults for BaBe, HoangLien, CucPhuong
(10980 × 10980), and CaMau (2609 × 6033). Override these for a different crop.
Region metadata also supplies these defaults (dimensions are height x width):

| Region | Source pixels | Patch grid (rows x columns, 256-pixel patches) | Merged pixels |
|:---|:---|:---|:---|
| zone1 | 13359 x 19482 | 52 x 76 | 13312 x 19456 |
| zone2 | 13360 x 13915 | 52 x 54 | 13312 x 13824 |
| zone4 | 32283 x 42302 | 126 x 165 | 32256 x 42240 |

No layout settings are needed for these regions. Grids are calculated using the
actual patch size. To override a default or supply an unknown region's layout:

```bash
python main.py ... 'dataset.prediction_layouts={zone3:{rows:42,columns:42}}'
```

Replace the example dimensions with the actual number of patch rows and columns.
Each layout also accepts `first_id` (default 0) and `order` (default `row-major`,
or `column-major`). Unknown regions without a layout fail before prediction.
Automatic candidate-grid search has been removed. Previously generated candidate
files are left on disk; new runs save `map.npy`, `map.png`, and `rgb.png` directly
in each region's output directory.

If the metadata's dimension order is uncertain, rebuild an explicit alternative
from already saved predictions without retraining or running inference again:

```bash
python prediction_map.py --zone zone1 --rows 52 --columns 76 \
  --sentinel-dir /path/to/Sentinel_duplicated/zone1
```

This saves `map.npy`, `map.png`, `rgb.png`, and `layout.json` in the separate
`inference_predictions/zone1/remerge_52x76_row-major/` folder. Original predictions
and maps are preserved. Use `--patch-dir` for predictions saved elsewhere and
`--order column-major` to check column-first patch numbering. Swapping grid
dimensions changes where patches are placed; it does not rotate the individual
patches. Inspect the original raster's `height` and `width` when available before
changing the default layout. Sparse white areas can also reflect masked pixels
or missing patches, so the screenshot alone cannot establish the correct grid.

For single-node multi-GPU DDP runs, `python main.py` automatically starts one
set of workers using `torch.distributed.run` (torchrun). The same workers execute
training, both evaluations, and prediction without relaunching the script between
stages. Keep `trainer.devices=2` and
`+trainer.strategy=ddp_find_unused_parameters_true` to use both Kaggle GPUs.
The driver waits for all workers to exit successfully, then merges their saved
patches on the CPU. This keeps map rendering out of distributed teardown.
Comet console capture is disabled, while metric and figure logging stay enabled.
Worker exceptions are recorded by TorchElastic with full Hydra tracebacks.

Single-GPU runs and directly launched `torchrun` jobs merge on rank zero after
the patches are saved. The output directory must be shared across ranks.

Prediction uses the last checkpoint and includes all patches in each selected
zone, including training patches and patches without GEDI labels, matching the
previous prediction behavior. Evaluation uses only held-out labeled patches.
Existing output files with the same paths are overwritten on subsequent runs.
