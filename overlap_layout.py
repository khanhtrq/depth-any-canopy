"""Shared patch layout and blending helpers (no model dependencies)."""
import numpy as np
import math


def grid(bounds):
    from pyproj import Transformer
    w, s, e, n = bounds
    zone = min(60, int(((w + e) / 2 + 180) // 6) + 1)
    crs = 'EPSG:{}'.format((32600 if (s + n) / 2 >= 0 else 32700) + zone)
    project = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    lon, lat = np.linspace(w, e, 101), np.linspace(s, n, 101)
    xs, ys = project.transform(np.concatenate([lon, lon, np.full(101, w), np.full(101, e)]),
                               np.concatenate([np.full(101, s), np.full(101, n), lat, lat]))
    left, top = math.floor(min(xs) / 10) * 10, math.ceil(max(ys) / 10) * 10
    return crs, left, top, math.ceil((max(xs)-left)/10), math.ceil((top-min(ys))/10)


def make_layout(width, height, patch_size=256, stride=128, **metadata):
    if not 0 < stride <= patch_size:
        raise ValueError('stride must be between 1 and patch size')
    columns = max(1, int(np.ceil(max(0, width-patch_size)/stride)) + 1)
    rows = max(1, int(np.ceil(max(0, height-patch_size)/stride)) + 1)
    return dict(rows=rows, columns=columns, width=width, height=height,
                patch_size=patch_size, stride=stride, order='row-major', first_id=0,
                blend='centre-weighted', **metadata)


def blend_weights(height, width):
    # Positive endpoints retain coverage even when only an edge is available.
    y = np.minimum(np.arange(height)+1, np.arange(height, 0, -1))
    x = np.minimum(np.arange(width)+1, np.arange(width, 0, -1))
    return np.outer(y/y.max(), x/x.max()).astype(np.float32)


def patch_window(row, column, patch_size, stride):
    return np.s_[row*stride:row*stride+patch_size,
                 column*stride:column*stride+patch_size]


def spatial_patch_split(layout, sample_id, ratio_train=0.8):
    """Shared spatial assignment, including intersecting regional exports."""
    row = (int(sample_id)-int(layout.get('first_id', 0))) // layout['columns']
    start = row*layout['stride']
    if 'transform' in layout:
        if layout.get('crs') != 'EPSG:32648':
            raise ValueError('Overlap spatial split currently requires EPSG:32648')
        north = layout['transform'][5]-start*10
        south = north-layout['patch_size']*10
        band = int(np.floor(north/10000))
        if south < band*10000+30 or north > (band+1)*10000-30:
            return None
        return 'train' if (band*0.6180339887498949) % 1 < ratio_train else 'test'
    cut = int(layout['height']*ratio_train)
    if start+layout['patch_size'] <= cut-3:
        return 'train'
    return 'test' if start >= cut+3 else None
