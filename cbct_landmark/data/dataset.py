# -*- coding:utf-8 -*-
"""Datasets for the two stages.

History: this replaces ``SkullWidthCBCT.py``.  The original class read the pre-cropped
128^3 ROI images from a hard-coded ``four_point_save_128`` folder and merged every
``*heatmap*`` file of the case with ``np.maximum`` into ONE channel.  That behaviour is
kept (``LandmarkROIDataset``); the whole-volume loader that ``tooth_region_pred.py`` relied
on is ``WholeVolumeDataset``.

Case folder layout expected by ``WholeVolumeDataset`` / ``read_case_image``::

    <data_root>/<NNN>_<name>/
        *.nrrd (or *.nii / *.nii.gz)   the CBCT volume
        *.mrk.json                     one 3D Slicer fiducial per landmark (LPS)
"""
import os

import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset

from .preprocessing import range_normalize, zoom_to

IMAGE_SUFFIXES = ('.nrrd', '.nii.gz', '.nii', '.mha', '.mhd')


def read_case_list(txt_file_path):
    with open(txt_file_path, mode='r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def find_case_image(case_dir):
    """Path of the CBCT volume inside a case folder (first file with an image suffix)."""
    for fn in sorted(os.listdir(case_dir)):
        if fn.lower().endswith(IMAGE_SUFFIXES):
            return os.path.join(case_dir, fn)
    raise FileNotFoundError('no image (%s) in %s' % ('/'.join(IMAGE_SUFFIXES), case_dir))


def read_case_image(case_dir):
    return sitk.ReadImage(find_case_image(case_dir))


class DatasetTransforms(object):
    """Apply a list of batchgenerators transforms to a ``{'data':..., 'seg':...}`` dict."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, batch_input):
        for t in self.transforms:
            batch_input = t(**batch_input)
        return batch_input


class WholeVolumeDataset(Dataset):
    """Stage 1 input: the whole CBCT resampled to ``target_size`` and range-normalised.

    Returns ``(image[1, D, H, W] float32, case_name)``.
    """

    def __init__(self, data_root_dir, cases, target_size=(72, 72, 72)):
        self.data_root_dir = data_root_dir
        self.cases = list(cases)
        self.target_size = tuple(target_size)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        image = sitk.GetArrayFromImage(read_case_image(os.path.join(self.data_root_dir, case))).astype(np.float32)
        image = zoom_to(image, self.target_size)
        image = range_normalize(image).astype(np.float32)
        return image[np.newaxis, ...], case


class LandmarkROIDataset(Dataset):
    """Stage 2 input: pre-cropped ROI ``<case>_image.nii.gz`` plus merged heat-map label.

    ``roi_dir`` is produced by ``scripts/3_crop_tooth_roi.py``.  All files of the case whose
    name contains ``heatmap`` are max-merged into a single channel (so either one merged file
    or one file per landmark works).  Returns ``(image[1,...], heatmap[1,...], case_name)``.
    """

    def __init__(self, roi_dir, cases, require_heatmap=True):
        self.roi_dir = roi_dir
        self.cases = list(cases)
        self.require_heatmap = require_heatmap
        self._files = sorted(os.listdir(roi_dir))

    def __len__(self):
        return len(self.cases)

    def heatmap_files(self, case):
        return [f for f in self._files if f.startswith(case + '_') and 'heatmap' in f]

    def __getitem__(self, index):
        case = self.cases[index]
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.roi_dir, case + '_image.nii.gz')))
        image = image.astype(np.float32)

        heatmap = np.zeros_like(image, dtype=np.float32)
        files = self.heatmap_files(case)
        if not files and self.require_heatmap:
            raise FileNotFoundError('no *heatmap* file for case %s in %s' % (case, self.roi_dir))
        for fn in files:
            h = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.roi_dir, fn))).astype(np.float32)
            heatmap = np.maximum(heatmap, h)

        image = range_normalize(image).astype(np.float32)
        return image[np.newaxis, ...], heatmap[np.newaxis, ...], case
