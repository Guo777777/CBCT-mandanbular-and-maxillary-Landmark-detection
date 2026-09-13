# -*- coding:utf-8 -*-
"""Intensity / geometry preprocessing used by the two stages.

The two stages normalise differently; both are reproduced here exactly as in the original
scripts so that released checkpoints behave as during training.

Stage 1 (whole CBCT -> 72^3, ``tooth_region_pred.py``):
    x = zoom(volume, 72^3)                       # scipy.ndimage.zoom, cubic spline
    x = x / (x.max() - x.min() + 1e-5)           # ``range_normalize`` (dataset __getitem__)
    x = clip(x, q01, q99); x = (x - min) / (max - min)   # ``quantile_minmax`` (inference script)

Stage 2 (tooth ROI -> 128^3, ``SkullWidthCBCT.py``):
    x = zoom(crop, 128^3)
    x = x / (x.max() - x.min() + 1e-5)           # ``range_normalize`` only
"""
import numpy as np
from scipy import ndimage


def range_normalize(data, epsilon=1e-5):
    """Divide by the intensity range (note: the minimum is *not* subtracted; this is what the
    original dataset did and what the checkpoints were trained with)."""
    return data / (np.max(data) - np.min(data) + epsilon)


def quantile_minmax(data, low=0.01, high=0.99, smooth=1e-10):
    """Clip to the [low, high] quantiles, then min-max scale to [0, 1] (original ``normalization``)."""
    a, b = np.quantile(data, low), np.quantile(data, high)
    data = np.clip(data, a, b)
    return (data - data.min()) / (data.max() - data.min() + smooth)


def zoom_to(array, target_shape, order=3):
    """Resample a 3-D array to ``target_shape`` with ``scipy.ndimage.zoom`` (cubic by default,
    as in the original code; use ``order=0`` for masks)."""
    factors = [t / s for t, s in zip(target_shape, array.shape)]
    return ndimage.zoom(array, factors, order=order)


def thresholding(data, t):
    """Binarise: 1 where ``data >= t`` else 0."""
    return (data >= t).astype(np.uint8)
