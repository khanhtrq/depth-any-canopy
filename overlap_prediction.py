"""Bounded-memory core stitching and historical weighted-overlap mosaicking."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from overlap_layout import blend_weights
from halo import crop_core


def save_overlap_map(dataset, zone, layout, output_dir, max_preview_size=4096, patch_dir=None):
    from prediction_map import patch_position
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    patches = Path(patch_dir) if patch_dir else output / 'np_array'
    entries = [i for i in dataset.file_idx if dataset.sample_regions[i] == zone]
    if not entries:
        return
    stride = int(layout['stride'])
    first = np.load(patches / (dataset.sample_ids[entries[0]] + '.npy'))
    ph, pw = first.shape[-2:]
    halo = layout.get('input_halo', 0)
    if halo and (stride != ph or stride != pw):
        raise ValueError('Halo predictions must be central, non-overlapping target tiles')
    if not 0 < stride <= min(ph, pw):
        raise ValueError('Invalid stride for prediction patch dimensions')
    shape = (int(layout.get('height', (layout['rows']-1)*stride+ph)),
             int(layout.get('width', (layout['columns']-1)*stride+pw)))
    weights = np.ones((ph, pw), np.float32) if halo else blend_weights(ph, pw)
    scale = min(1., max_preview_size / max(shape))
    ys = np.linspace(0, shape[0]-1, max(1, round(shape[0]*scale))).astype(int)
    xs = np.linspace(0, shape[1]-1, max(1, round(shape[1]*scale))).astype(int)
    rgb_sum = np.zeros((len(ys), len(xs), 3), np.float32)
    rgb_den = np.zeros((len(ys), len(xs)), np.float32)
    preview = np.full((len(ys), len(xs)), np.nan, np.float32)
    positions = []
    for i in entries:
        row, col = patch_position(dataset.sample_ids[i], layout)
        top, left = row*stride, col*stride
        h, w = min(ph, shape[0]-top), min(pw, shape[1]-left)
        if h <= 0 or w <= 0:
            raise ValueError('Patch outside output extent: ' + dataset.sample_ids[i])
        positions.append((i, top, left, h, w))
    # Write each output stripe once. No full-region sum/weight mappings: for
    # zone 4 these would touch >12 GB of pages and overwhelm a laptop.
    with (output/'map.npy').open('wb') as handle:
        np.lib.format.write_array_header_1_0(handle, dict(
            descr=np.lib.format.dtype_to_descr(np.dtype('float32')),
            fortran_order=False, shape=shape))
        stripe_height = max(1, min(128, (32*1024*1024)//(8*shape[1])))
        for stripe_top in range(0, shape[0], stripe_height):
            stripe_bottom = min(shape[0], stripe_top+stripe_height)
            total = np.zeros((stripe_bottom-stripe_top, shape[1]), np.float32)
            denom = np.zeros_like(total)
            for i, top, left, h, w in positions:
                if top >= stripe_bottom or top+h <= stripe_top:
                    continue
                sid = dataset.sample_ids[i]
                patch = np.load(patches/(sid+'.npy'))
                sentinel = np.load(dataset.sentinel_paths[i])
                if halo:
                    sentinel = crop_core(sentinel, (ph, pw), halo)
                if patch.size != ph*pw or patch.shape[-2:] != (ph, pw) or sentinel.shape[-2:] != (ph, pw):
                    raise ValueError('Inconsistent patch shape: ' + sid)
                patch = patch.reshape(ph, pw)[:h, :w]
                valid = np.isfinite(sentinel[:, :h, :w]).all(0) & np.any(sentinel[:, :h, :w] != 0, axis=0)
                a, b = max(top, stripe_top), min(top+h, stripe_bottom)
                source = np.s_[a-top:b-top, :w]
                target = np.s_[a-stripe_top:b-stripe_top, left:left+w]
                weight = weights[:h, :w][source] * (valid[source] & np.isfinite(patch[source]))
                if halo:
                    # Each retained target pixel belongs to one tile. Copy it;
                    # context pixels never enter the mosaic or its RGB preview.
                    total[target] = np.where(weight > 0, patch[source], 0)
                    denom[target] = weight
                else:
                    total[target] += np.where(weight > 0, patch[source], 0)*weight
                    denom[target] += weight
                # Build RGB exactly once per patch, on its first stripe.
                if not stripe_top <= top < stripe_bottom:
                    continue
                yi = np.flatnonzero((ys >= top) & (ys < top+h))
                xi = np.flatnonzero((xs >= left) & (xs < left+w))
                sample = np.ix_(ys[yi]-top, xs[xi]-left)
                display_weight = weights[:h, :w][sample]*valid[sample]
                target = np.ix_(yi, xi)
                rgb_sum[target] += np.nan_to_num(sentinel[[3, 2, 1], :h, :w].transpose(1, 2, 0)[sample])*display_weight[..., None]
                rgb_den[target] += display_weight
            np.divide(total, denom, out=total, where=denom > 0)
            total[denom == 0] = np.nan
            preview_rows = np.flatnonzero((ys >= stripe_top) & (ys < stripe_bottom))
            preview[preview_rows] = total[np.ix_(ys[preview_rows]-stripe_top, xs)]
            handle.write(total.tobytes())
    rgb = np.divide(rgb_sum, rgb_den[..., None], out=np.zeros_like(rgb_sum), where=rgb_den[..., None] > 0)
    visible = rgb_den > 0
    if visible.any():
        low, high = np.percentile(rgb[visible], [2, 98])
        rgb = np.clip((rgb-low)/max(high-low, 1e-8), 0, 1)
    rgb[~visible] = 1
    Image.fromarray((rgb*255).astype('uint8')).save(output/'rgb.png')
    cmap = plt.get_cmap('Spectral_r').copy()
    cmap.set_bad('white')
    fig, ax = plt.subplots(figsize=(12, 10))
    try:
        image = ax.imshow(np.ma.masked_invalid(preview), cmap=cmap, vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(zone)
        fig.colorbar(image, ax=ax).set_label('Canopy height (normalized; multiply by %g m)' % dataset.max_height)
        fig.savefig(output/'map.png', dpi=150, bbox_inches='tight')
    finally:
        plt.close(fig)
    (output/'layout.json').write_text(json.dumps(dict(layout, units='normalized canopy height', max_height=dataset.max_height), indent=2))
