"""Download mangrove-only Sentinel-2/GEDI NumPy patches from Earth Engine.

Install: python -m pip install earthengine-api numpy requests pyproj
Example (bounds are illustrative; end date is exclusive):
  python download_canopy_data.py --project YOUR_PROJECT --authenticate \
    --bbox 104.70 8.55 105.10 8.90 --start 2020-01-01 --end 2021-01-01 \
    --output data/canopy --region CaMau

Output: Sentinel/<region>/<id>.npy (12,H,W, float32, SR * 10000),
GEDI/<region>/<id>.npy (H,W, float32 metres, NaN for missing),
mask/<region>/<id>.npz (mangrove, optical_valid, supervision),
gedi_source/<region>/<id>.npz (month, native_x, native_y),
metadata/<region>/<id>.json and manifest.json.

GEDI is a sparse ~25 m footprint measurement, NOT 10 m ground truth. The
monthly raster is sampled with nearest neighbour on the output grid. Repeated
(month, native_x, native_y) values identify the SAME source raster cell, not
independent observations. Keep these groups together in input/target splits.
Monthly rasterization cannot recover observations discarded during compositing.
GMW is the community-hosted 10 m 2020 v4.0.19 baseline, not a current-year map.
"""

import argparse
from datetime import date
import io
import json
import math
from pathlib import Path
import time

import ee
import numpy as np
from pyproj import Transformer
import requests


BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
S2 = 'COPERNICUS/S2_SR_HARMONIZED'
GEDI = 'LARSE/GEDI/GEDI02_A_002_MONTHLY'
GMW = 'projects/sat-io/open-datasets/GMW/annual-extent/GMW_MNG_2020'
NODATA = -9999


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project', required=True, help='Earth Engine enabled Cloud project ID')
    p.add_argument('--authenticate', action='store_true', help='Run interactive Google authentication')
    p.add_argument('--bbox', type=float, nargs=4, required=True, metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'))
    p.add_argument('--start', required=True, help='Inclusive YYYY-MM-DD')
    p.add_argument('--end', required=True, help='Exclusive YYYY-MM-DD; also used for GEDI filtering')
    p.add_argument('--output', type=Path, required=True, help='New output directory (prevents accidental mixing of runs)')
    p.add_argument('--region', default='study_area')
    p.add_argument('--patch-size', type=int, default=256, help='Pixels, from 1 to 512')
    p.add_argument('--cloud-percent', type=float, default=3, help='Maximum scene cloud percentage, exclusive')
    p.add_argument('--sensitivity', type=float, default=0.95)
    p.add_argument('--min-height', type=float, default=0)
    p.add_argument('--max-height', type=float, default=70)
    p.add_argument('--require-gedi', action='store_true', help='Save only patches with valid supervision')
    a = p.parse_args()
    w, s, e, n = a.bbox
    if not (-180 <= w < e <= 180 and -51.6 <= s < n <= 51.6):
        p.error('Bounds must be ordered W S E N, without crossing the dateline, within GEDI latitudes +/-51.6.')
    if e - w > 6 or s < 0 < n:
        p.error('Use a regional bbox within one hemisphere and no wider than 6 degrees for the UTM grid.')
    if date.fromisoformat(a.start) >= date.fromisoformat(a.end):
        p.error('start must precede end')
    if not 1 <= a.patch_size <= 512 or not 0 < a.cloud_percent <= 100:
        p.error('patch-size must be 1..512 and cloud-percent must be >0 and <=100')
    if not 0 <= a.sensitivity <= 1 or not 0 <= a.min_height <= a.max_height:
        p.error('Invalid GEDI sensitivity or height thresholds')
    if not a.region or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.region):
        p.error('region must contain only letters, numbers, underscore or hyphen')
    return a


def grid(bounds):
    w, s, e, n = bounds
    zone = min(60, int(((w + e) / 2 + 180) // 6) + 1)
    crs = 'EPSG:{}'.format((32600 if (s + n) / 2 >= 0 else 32700) + zone)
    project = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    # Densify the boundary so a curved projected edge is not clipped.
    lon = np.linspace(w, e, 101)
    lat = np.linspace(s, n, 101)
    xs, ys = project.transform(np.concatenate([lon, lon, np.full(101, w), np.full(101, e)]),
                               np.concatenate([np.full(101, s), np.full(101, n), lat, lat]))
    left = math.floor(min(xs) / 10) * 10
    top = math.ceil(max(ys) / 10) * 10
    width = math.ceil((max(xs) - left) / 10)
    height = math.ceil((top - min(ys)) / 10)
    return crs, left, top, width, height


def optical_image(image):
    scl = image.select('SCL')
    # Retain vegetation, bare ground, water and unclassified; exclude shadows,
    # cloud classes, cirrus, snow, defective pixels and no-data.
    clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    selected = image.select(BANDS)
    valid = selected.mask().reduce(ee.Reducer.min()).And(clear)
    # Resample each source image before compositing (avoid resampling a composite
    # whose default projection may be WGS84 at one degree).
    return selected.updateMask(valid).resample('bilinear')


def gedi_image(image, args):
    rh = image.select('rh98')
    good = (image.select('quality_flag').eq(1)
            .And(image.select('degrade_flag').eq(0))
            .And(image.select('sensitivity').gte(args.sensitivity))
            .And(rh.gte(args.min_height)).And(rh.lte(args.max_height)))
    native = ee.Image.pixelCoordinates(rh.projection()).rename(['native_x', 'native_y'])
    month = ee.Image.constant(ee.Number.parse(image.date().format('YYYYMM'))).rename('month')
    # All bands share one mask; mosaic selects height and identifiers together.
    return rh.addBands(native).addBands(month).updateMask(good).toDouble()


def download(image, crs, transform, size):
    for attempt in range(4):
        try:
            url = image.getDownloadURL({'format': 'NPY', 'crs': crs,
                                        'crs_transform': transform, 'dimensions': [size, size]})
            response = requests.get(url, timeout=(30, 300))
            response.raise_for_status()
            result = np.load(io.BytesIO(response.content), allow_pickle=False)
            if result.shape != (size, size) or result.dtype.names is None:
                raise ValueError('Unexpected Earth Engine NumPy layout: {}'.format(result.shape))
            return result
        except (requests.RequestException, ee.EEException):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    a = arguments()
    if a.authenticate:
        ee.Authenticate()
    ee.Initialize(project=a.project)
    roi = ee.Geometry.Rectangle(a.bbox, proj='EPSG:4326', geodesic=False)
    crs, left, top, width, height = grid(a.bbox)
    sentinel = (ee.ImageCollection(S2).filterBounds(roi).filterDate(a.start, a.end)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', a.cloud_percent)))
    gedi = ee.ImageCollection(GEDI).filterBounds(roi).filterDate(a.start, a.end)
    counts = ee.Dictionary({'sentinel_scenes': sentinel.size(), 'gedi_months': gedi.size()}).getInfo()
    if not counts['sentinel_scenes']:
        raise RuntimeError('No Sentinel scenes. Widen the dates or increase --cloud-percent.')
    if not counts['gedi_months']:
        raise RuntimeError('No GEDI monthly assets in this area/date interval. Choose different dates.')
    print('Available assets:', counts, flush=True)
    # Nearest-neighbour is the default for categorical GMW and GEDI; do not
    # apply bilinear interpolation to either of these products.
    mangrove = (ee.ImageCollection(GMW).filterBounds(roi).mosaic().select('b1')
                .unmask(0, False).eq(1).clip(roi).rename('mangrove'))
    optical = sentinel.map(optical_image).median().clip(roi)
    optical_valid = optical.mask().reduce(ee.Reducer.min()).unmask(0, False).rename('optical_valid')
    heights = gedi.sort('system:time_start').map(lambda image: gedi_image(image, a)).mosaic().clip(roi)
    stack = (optical.updateMask(mangrove).unmask(NODATA, False)
             .addBands(heights.updateMask(mangrove).unmask(NODATA, False))
             .addBands(mangrove.unmask(0, False)).addBands(optical_valid).toDouble())
    nx, ny = math.ceil(width / a.patch_size), math.ceil(height / a.patch_size)
    print('{} x {} patches to inspect on {} at 10 m'.format(nx, ny, crs), flush=True)
    # A new directory is required so stale patches cannot silently enter training.
    a.output.mkdir(parents=True, exist_ok=False)
    folders = {}
    for name in ['Sentinel', 'GEDI', 'mask', 'gedi_source', 'metadata']:
        folders[name] = a.output / name / a.region
        folders[name].mkdir(parents=True)
    manifest = {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'crs': crs, 'pixel_size_m': 10, 'bands': BANDS,
        'sentinel_units': 'harmonized surface reflectance multiplied by 10000; not standardized',
        'gedi_units': 'metres; NaN means missing',
        'collections': {'sentinel': S2, 'gedi': GEDI, 'gmw': GMW},
        'gmw_epoch': 2020, 'counts': counts,
        'sentinel_scene_ids': sentinel.aggregate_array('system:index').getInfo(),
        'gedi_month_ids': gedi.aggregate_array('system:index').getInfo(),
        'optical_composite': 'pixelwise median of SCL-masked scenes',
        'gedi_composite': 'latest valid monthly raster cell; nearest neighbour on 10 m grid',
        'gedi_group_key': 'month,native_x,native_y (source raster cell, not original shot ID)',
        'saved': [], 'skipped': [], 'status': 'running',
    }
    manifest_path = a.output / 'manifest.json'

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    save_manifest()
    try:
        for row in range(ny):
            for col in range(nx):
                patch_id = row * nx + col
                print('Inspecting patch {} / {}'.format(patch_id + 1, nx * ny), flush=True)
                transform = [10, 0, left + col * a.patch_size * 10,
                             0, -10, top - row * a.patch_size * 10]
                data = download(stack, crs, transform, a.patch_size)
                mask = data['mangrove'] == 1
                valid = mask & (data['optical_valid'] > 0)
                x = np.stack([data[b] for b in BANDS]).astype(np.float32)
                valid &= np.isfinite(x).all(axis=0) & (x != NODATA).all(axis=0)
                y = data['rh98'].astype(np.float32)
                supervision = valid & np.isfinite(y) & (y != NODATA)
                if not valid.any() or (a.require_gedi and not supervision.any()):
                    manifest['skipped'].append(patch_id)
                    continue
                x[:, ~valid] = 0
                y[~supervision] = np.nan
                np.save(folders['Sentinel'] / '{}.npy'.format(patch_id), x)
                np.save(folders['GEDI'] / '{}.npy'.format(patch_id), y)
                np.savez_compressed(folders['mask'] / '{}.npz'.format(patch_id),
                                    mangrove=mask, optical_valid=valid, supervision=supervision)
                identifiers = {}
                for key in ['month', 'native_x', 'native_y']:
                    identifiers[key] = np.where(supervision, data[key], NODATA).astype(np.int64)
                np.savez_compressed(folders['gedi_source'] / '{}.npz'.format(patch_id), **identifiers)
                metadata = {'crs': crs, 'transform': transform, 'height': a.patch_size,
                            'width': a.patch_size, 'row_start': row * a.patch_size,
                            'col_start': col * a.patch_size, 'bands': BANDS,
                            'mangrove_pixels': int(mask.sum()), 'optical_valid_pixels': int(valid.sum()),
                            'supervised_pixels': int(supervision.sum()),
                            'transform_convention': 'x=a*col+b*row+c; y=d*col+e*row+f; pixel upper-left corner'}
                (folders['metadata'] / '{}.json'.format(patch_id)).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
                manifest['saved'].append(patch_id)
                save_manifest()
                print('Saved {}: {} mangrove pixels, {} supervised pixels'.format(patch_id, mask.sum(), supervision.sum()), flush=True)
        manifest['status'] = 'complete'
        print('Finished: {} saved, {} skipped. {}'.format(len(manifest['saved']), len(manifest['skipped']), a.output))
    except Exception:
        manifest['status'] = 'failed; output is partial'
        raise
    finally:
        save_manifest()


if __name__ == '__main__':
    main()
