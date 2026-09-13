# -*- coding:utf-8 -*-
"""Tooth-region ROI: from the coarse 72^3 mask to a crop box on the original grid and back.

All boxes are in numpy array-index order ``(z, y, x)`` = SimpleITK index ``(k, j, i)`` and are
half-open ``[lo, hi)``.  The arithmetic is exactly the original ``json_process_test0.py``:

    lo = int(min_idx / 72 * dim) - 10,   hi = int(max_idx / 72 * dim) + 10   (clamped)
    crop = volume[lo:hi], resampled to 128^3 with cubic spline interpolation.
"""
import json

import numpy as np

from .data.preprocessing import zoom_to


def mask_to_roi(mask_small, orig_shape, margin=10):
    """Bounding box of the non-zero voxels of ``mask_small`` (any resolution), scaled to
    ``orig_shape`` and dilated by ``margin`` voxels.  Returns ``[[z0, z1], [y0, y1], [x0, x1]]``."""
    idx = np.nonzero(mask_small)
    if idx[0].size == 0:
        raise ValueError('empty tooth-region mask')
    box = []
    for ax in range(3):
        n_small, n_orig = mask_small.shape[ax], orig_shape[ax]
        lo = int((idx[ax].min() / n_small) * n_orig)
        hi = int((idx[ax].max() / n_small) * n_orig)
        lo = max(0, lo - margin)
        hi = min(n_orig, hi + margin)
        box.append([int(lo), int(hi)])
    return box


def crop_roi(volume, box):
    (z0, z1), (y0, y1), (x0, x1) = box
    return volume[z0:z1, y0:y1, x0:x1]


def crop_and_resize(volume, box, size=128, order=3):
    return zoom_to(crop_roi(volume, box), (size, size, size), order=order)


def index_orig_to_roi(index_zyx, box, size=128):
    """Original-grid continuous index -> index in the resampled ROI (``new / crop_size * size``)."""
    out = []
    for v, (lo, hi) in zip(index_zyx, box):
        out.append((v - lo) / float(hi - lo) * size)
    return out


def index_roi_to_orig(index_zyx, box, size=128):
    """Inverse of :func:`index_orig_to_roi` (``roi / size * crop_size + lo``)."""
    out = []
    for v, (lo, hi) in zip(index_zyx, box):
        out.append(v / float(size) * (hi - lo) + lo)
    return out


def save_roi_meta(path, box, orig_shape, size, extra=None):
    meta = {'box_zyx': box, 'orig_shape_zyx': [int(v) for v in orig_shape], 'roi_size': int(size)}
    if extra:
        meta.update(extra)
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


def load_roi_meta(path):
    with open(path) as f:
        return json.load(f)
