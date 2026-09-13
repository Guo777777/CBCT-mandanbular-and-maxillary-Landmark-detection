# -*- coding:utf-8 -*-
"""Predicted heat-map -> four landmark coordinates -> basal bone widths.

Reproduces the post-processing of ``json_process_test0.py``:
    1. binarise the heat-map at 0.1,
    2. label connected components (26-connectivity), keep the 4 largest,
    3. take their centroids in the 128^3 ROI grid,
    4. map back to the original voxel grid (``roi.index_roi_to_orig``) and to physical LPS mm.

Landmark naming (paper): L1/L2 = left/right maxillary, L3/L4 = left/right mandibular point
(centre of the root bifurcation).  Maxillary width = |L1-L2|, mandibular width = |L3-L4|.
The original evaluation code ordered the four centroids by ``z + y`` index and then matched
distances to the closest ground-truth pair; here the assignment is made explicitly from the
physical coordinates: the two most superior points (largest S) are maxillary and, within each
jaw, the point with the larger L coordinate is the left one.
"""
import numpy as np
from skimage import measure

from .roi import index_roi_to_orig

LANDMARK_NAMES = ('L1', 'L2', 'L3', 'L4')


def heatmap_to_peaks(heatmap, threshold=0.1, n_landmarks=4):
    """Centroids (z, y, x, float) of the ``n_landmarks`` largest connected components of
    ``heatmap >= threshold``, largest first.  Fewer are returned if fewer components exist."""
    binary = (heatmap >= threshold).astype(np.uint8)
    labels = measure.label(binary, connectivity=3)
    props = measure.regionprops(labels)
    props = sorted(props, key=lambda r: r.area, reverse=True)[:n_landmarks]
    return [list(map(float, r.centroid)) for r in props]


def index_to_physical(sitk_image, index_zyx):
    """Continuous array index (z, y, x) -> physical LPS point (x, y, z) in mm."""
    i, j, k = float(index_zyx[2]), float(index_zyx[1]), float(index_zyx[0])
    return list(sitk_image.TransformContinuousIndexToPhysicalPoint((i, j, k)))


def physical_to_index(sitk_image, point_xyz):
    """Physical LPS point (x, y, z) -> continuous array index (z, y, x)."""
    i, j, k = sitk_image.TransformPhysicalPointToContinuousIndex([float(v) for v in point_xyz])
    return [k, j, i]


def peaks_to_physical(peaks_roi, box, sitk_image, size=128):
    return [index_to_physical(sitk_image, index_roi_to_orig(p, box, size)) for p in peaks_roi]


def assign_landmarks(points_xyz):
    """Assign up to four LPS points to L1..L4 (see module docstring).  Missing -> ``None``."""
    pts = [np.asarray(p, dtype=float) for p in points_xyz]
    out = {n: None for n in LANDMARK_NAMES}
    if len(pts) < 4:
        # not enough detections: return them unnamed in order of appearance
        for n, p in zip(LANDMARK_NAMES, pts):
            out[n] = p.tolist()
        return out
    order = np.argsort([-p[2] for p in pts])           # most superior first (S = +z in LPS)
    maxilla, mandible = [pts[i] for i in order[:2]], [pts[i] for i in order[2:4]]
    maxilla.sort(key=lambda p: -p[0])                   # left (L = +x) first
    mandible.sort(key=lambda p: -p[0])
    out['L1'], out['L2'] = maxilla[0].tolist(), maxilla[1].tolist()
    out['L3'], out['L4'] = mandible[0].tolist(), mandible[1].tolist()
    return out


def euclid(a, b):
    if a is None or b is None:
        return float('nan')
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def basal_bone_widths(landmarks):
    """``{'maxillary_width_mm', 'mandibular_width_mm'}`` from an L1..L4 dict."""
    return {'maxillary_width_mm': euclid(landmarks['L1'], landmarks['L2']),
            'mandibular_width_mm': euclid(landmarks['L3'], landmarks['L4'])}


def heatmap_to_landmarks(heatmap, box, sitk_image, size=128, threshold=0.1):
    """Full chain: ROI heat-map -> ``{'L1':[x,y,z], ...}`` in LPS mm plus the raw peaks."""
    peaks = heatmap_to_peaks(heatmap, threshold=threshold)
    phys = peaks_to_physical(peaks, box, sitk_image, size)
    return assign_landmarks(phys), peaks
