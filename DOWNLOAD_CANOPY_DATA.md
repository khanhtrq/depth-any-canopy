Run with Python 3.11 or newer in an environment with these dependencies:

```powershell
python -m pip install earthengine-api numpy requests pyproj
python download_canopy_data.py --project YOUR_CLOUD_PROJECT --authenticate --bbox 104.70 8.55 105.10 8.90 --start 2020-01-01 --end 2021-01-01 --output data/camau_2020 --region CaMau
```

The bounds above are illustrative, in **west longitude, south latitude, east
longitude, north latitude** order. Replace them with your area. The Google Cloud
project must be registered for Earth Engine with its API enabled and your account
authorized. Omit `--authenticate` after credentials have been saved. Use a new
output directory for each run. Start with a small area to check data availability.

Dates apply to both Sentinel-2 and GEDI. End dates are exclusive. Use first-of-month
boundaries because GEDI is a monthly collection. Images are composited over the
whole selected interval, not matched individually to each GEDI observation date;
use a short interval when possible. A scene cloud percentage below 3% is the
default from the article; in persistently cloudy regions try `--cloud-percent 20`.
Pixel-level SCL cloud/shadow masking is still applied. SCL classification is not
perfect, so inspect the outputs before training.

The script automatically chooses a UTM CRS and constructs a fixed 10 m grid.
It resamples Sentinel-2 source imagery with bilinear interpolation before a median
composite. GEDI and the categorical GMW mask use nearest neighbour. Patches are
256 x 256 pixels by default. Edge patches are padded outside the requested area.
Patches without usable mangrove imagery are skipped; add `--require-gedi` to also
skip patches without usable heights. A failed run is marked partial in the
manifest; there is no automatic resume. Earlier completed patches remain on disk.

Files for each saved patch:

| Path | Contents |
| --- | --- |
| `Sentinel/CaMau/0.npy` | float32 `(12,256,256)`; harmonized SR multiplied by 10000; zero outside valid mangrove imagery |
| `GEDI/CaMau/0.npy` | float32 `(256,256)`; RH98 in metres; NaN outside supervision |
| `mask/CaMau/0.npz` | boolean `mangrove`, `optical_valid`, `supervision` arrays |
| `gedi_source/CaMau/0.npz` | int64 `month`, `native_x`, `native_y`; -9999 outside supervision |
| `metadata/CaMau/0.json` | CRS, transform, band order, patch offsets and counts |
| `manifest.json` | Settings, source scene IDs, collection versions, saved/skipped IDs and completion status |

Band order is B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12. Band normalization has
**not** been applied: compute normalization statistics on valid pixels from your
training split only. Existing statistics for a different region/band set may not
be appropriate. Do not pass unmasked NaN targets into a loss:

```python
import numpy as np

x = np.load('data/camau_2020/Sentinel/CaMau/0.npy')
y = np.load('data/camau_2020/GEDI/CaMau/0.npy')
with np.load('data/camau_2020/mask/CaMau/0.npz') as masks:
    valid = masks['supervision']
# For a prediction array with shape (256,256):
# loss = ((prediction[valid] - y[valid]) ** 2).mean()
# Skip the loss when valid has no True values.
```

GEDI filters: quality_flag=1, degrade_flag=0, sensitivity>=0.95, 0<=RH98<=70 m.
The script selects the latest valid monthly raster value at each location.
**These are sparse ~25 m measurements aligned to a 10 m grid.** Nearest-neighbour
sampling can repeat a source cell across output pixels. Equal
`(month,native_x,native_y)` identifiers must stay together when assigning GEDI to
input versus supervision/evaluation, including across patch boundaries. These
are raster-cell identifiers, not original shot numbers. Spatially separated
training/testing blocks are preferable to random pixel splits. For footprint-level
evaluation or retaining every individual shot, use the original vector GEDI data.

GMW is the community-hosted **2020 10 m v4.0.19 baseline**, not a map of the
selected image year. Later mangrove gains/losses and classification errors can
affect masking. Selection means a pixel is mapped as mangrove; GEDI footprints
near boundaries may also include non-mangrove cover. Background areas are masked,
not removed from the rectangular array. Preserve the mask for inference as well.

Source documentation:

- https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
- https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI02_A_002_MONTHLY
- https://gee-community-catalog.org/projects/mangrove/ (GMW asset and attribution)
- https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl

GMW attribution: Global Mangrove Watch, Bunting et al.; community Earth Engine
ingestion by Samapriya Roy. Follow the dataset's CC BY 4.0 attribution requirements
when distributing derived products.
