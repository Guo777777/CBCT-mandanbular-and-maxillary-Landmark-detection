# -*- coding:utf-8 -*-
"""Gaussian heat-map generation for landmark training labels.

Reproduces ``generate_3d_heatmap`` from the original ``json_process*.py`` scripts:
radius 5, sigma 2, peak value 1, merged into the *same* channel with ``np.maximum``.
"""
import numpy as np


def gaussian_kernel_3d(radius=5, sigma=2.0):
    diameter = 2 * radius + 1
    m = n = k = (diameter - 1) / 2
    zz, xx, yy = np.ogrid[-m:m + 1, -n:n + 1, -k:k + 1]
    h = np.exp(-(xx * xx + yy * yy + zz * zz) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius=5, sigma=2.0):
    """Paste a 3-D Gaussian centred at ``center`` (array-index order z, y, x; floats are
    truncated as in the original code) into ``heatmap`` in place using element-wise maximum."""
    z0, x0, y0 = center
    h = gaussian_kernel_3d(radius, sigma)
    z_limit, x_limit, y_limit = heatmap.shape[:3]
    z_min, z_max = min(z0, radius), min(z_limit - z0 + 1, radius + 1)
    x_min, x_max = min(x0, radius), min(x_limit - x0 + 1, radius + 1)
    y_min, y_max = min(y0, radius), min(y_limit - y0 + 1, radius + 1)

    masked_heatmap = heatmap[int(z0 - z_min):int(z0 + z_max),
                             int(x0 - x_min):int(x0 + x_max),
                             int(y0 - y_min):int(y0 + y_max)]
    masked_gaussian = h[int(radius - z_min):int(radius + z_max),
                        int(radius - x_min):int(radius + x_max),
                        int(radius - y_min):int(radius + y_max)]
    masked_gaussian = masked_gaussian[:masked_heatmap.shape[0], :masked_heatmap.shape[1], :masked_heatmap.shape[2]]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap


def landmarks_to_heatmap(centers, shape=(128, 128, 128), radius=5, sigma=2.0):
    """Single-channel heat-map with one Gaussian per landmark (array-index order z, y, x)."""
    heatmap = np.zeros(shape, dtype=np.float32)
    for c in centers:
        draw_gaussian(heatmap, c, radius, sigma)
    return heatmap
