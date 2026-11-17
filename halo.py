"""Geometry shared by halo inputs, model outputs and region-map assembly."""
import math


def make_halo_layout(width, height, patch_size=256, input_halo=32, **georeference):
    """Describe a core tile grid; no imagery is read, padded or downloaded."""
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
           for value in (width, height, patch_size)):
        raise ValueError('Region dimensions and patch_size must be positive integers')
    if not isinstance(input_halo, int) or isinstance(input_halo, bool) or input_halo < 0:
        raise ValueError('input_halo must be a non-negative integer')
    return dict(rows=math.ceil(height/patch_size), columns=math.ceil(width/patch_size),
                height=height, width=width, patch_size=patch_size, stride=patch_size,
                input_halo=input_halo, input_shape=[patch_size+2*input_halo]*2,
                first_id=0, order='row-major', merge='core-stitch', **georeference)


def crop_core(array, target_shape, halo):
    if not isinstance(halo, int) or isinstance(halo, bool) or halo < 0:
        raise ValueError('input_halo must be a non-negative integer')
    height, width = map(int, target_shape)
    expected = (height+2*halo, width+2*halo)
    if tuple(array.shape[-2:]) != expected:
        raise ValueError(f'Expected Sentinel input {expected} for target {(height, width)} and halo {halo}; got {tuple(array.shape[-2:])}')
    return array[..., halo:halo+height, halo:halo+width]


def prediction_core(prediction, input_shape, target_shape, resize):
    """Restore the full input footprint before cropping; preserve autograd."""
    ih, iw = map(int, input_shape)
    th, tw = map(int, target_shape)
    dh, dw = ih-th, iw-tw
    if dh < 0 or dw < 0 or dh != dw or dh % 2:
        raise ValueError('Input must equal the target size plus a symmetric square halo')
    restored = resize(prediction, (ih, iw), interpolation='bilinear')
    return crop_core(restored, (th, tw), dh//2)
