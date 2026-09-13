#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""End-to-end inference on new CBCT scans: image -> 4 landmarks (LPS mm) -> basal bone widths.

This is the single-scan equivalent of running scripts 2 -> 3 -> 5 -> 6 and is the recommended
entry point for external validation.

    python inference.py --image /path/case.nrrd \
        --global-ckpt weights/tooth_region_bestmodel.pth \
        --local-ckpt  weights/landmark_Unet_model08.pt \
        --out-dir results/ [--save-intermediate]

Per scan the following files are written to ``--out-dir``:
    <case>_landmarks.mrk.json   L1..L4 as 3D Slicer fiducials (LPS, mm)
    <case>_result.json          landmarks, widths, ROI box, number of detected peaks
    <case>_tooth_mask_72.nii.gz / <case>_roi_128.nii.gz / <case>_heatmap_128.nii.gz  (--save-intermediate)
and a ``results.csv`` summarising every scan.

Pipeline (identical to training-time preprocessing, see README "Preprocessing details"):
    1. whole volume -> 72^3 (cubic), /range, clip 1-99 %, min-max       -> UNet3D_simple -> mask >= 0.5
    2. mask bbox -> original grid, +10 voxels, crop, -> 128^3 (cubic), /range -> UNet3d -> heat-map
    3. heat-map >= 0.1, 26-connected components, 4 largest centroids -> original index -> LPS mm
    4. L1/L2 = the two most superior points (maxilla), L3/L4 = mandible; left = larger L (+x)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbct_landmark.data import (find_case_image, write_markups, range_normalize, quantile_minmax,  # noqa: E402
                                zoom_to, IMAGE_SUFFIXES)
from cbct_landmark.models import build_global_net, build_local_net, load_checkpoint  # noqa: E402
from cbct_landmark.postprocess import heatmap_to_landmarks, basal_bone_widths, LANDMARK_NAMES  # noqa: E402
from cbct_landmark.roi import mask_to_roi, crop_and_resize  # noqa: E402


def load_models(global_ckpt, local_ckpt, device):
    g = load_checkpoint(build_global_net(), global_ckpt, device).to(device).eval()
    l = load_checkpoint(build_local_net(), local_ckpt, device).to(device).eval()
    return g, l


@torch.no_grad()
def predict_tooth_mask(volume, global_net, device, size=72, threshold=0.5):
    x = zoom_to(volume.astype(np.float32), (size, size, size))
    x = quantile_minmax(range_normalize(x))
    x = torch.from_numpy(np.ascontiguousarray(x))[None, None].float().to(device)
    prob = global_net(x)[0, 0].cpu().numpy()
    return (prob >= threshold).astype(np.uint8)


@torch.no_grad()
def predict_heatmap(roi, local_net, device):
    x = range_normalize(roi.astype(np.float32))
    x = torch.from_numpy(np.ascontiguousarray(x))[None, None].float().to(device)
    return local_net(x)[0, 0].cpu().numpy()


def run_inference(sitk_image, global_net, local_net, device, coarse_size=72, roi_size=128, margin=10,
                  mask_threshold=0.5, heatmap_threshold=0.1):
    """Returns ``(result_dict, intermediates)`` for one SimpleITK image."""
    volume = sitk.GetArrayFromImage(sitk_image).astype(np.float32)      # (z, y, x)
    warnings = []

    mask = predict_tooth_mask(volume, global_net, device, coarse_size, mask_threshold)
    try:
        box = mask_to_roi(mask, volume.shape, margin=margin)
    except ValueError:
        warnings.append('empty tooth-region mask; using the whole volume as ROI')
        box = [[0, s] for s in volume.shape]

    roi = crop_and_resize(volume, box, size=roi_size)
    heatmap = predict_heatmap(roi, local_net, device)
    landmarks, peaks = heatmap_to_landmarks(heatmap, box, sitk_image, size=roi_size, threshold=heatmap_threshold)
    if len(peaks) < 4:
        warnings.append('only %d heat-map peaks found (expected 4)' % len(peaks))

    result = {'landmarks_lps_mm': landmarks, 'n_peaks': len(peaks), 'roi_box_zyx': box,
              'image_size_xyz': list(sitk_image.GetSize()), 'spacing_xyz': list(sitk_image.GetSpacing()),
              'warnings': warnings}
    result.update(basal_bone_widths(landmarks))
    return result, {'mask': mask, 'roi': roi, 'heatmap': heatmap}


def collect_inputs(args):
    paths = list(args.image or [])
    if args.input_dir:
        for entry in sorted(os.listdir(args.input_dir)):
            p = os.path.join(args.input_dir, entry)
            if os.path.isdir(p):
                try:
                    paths.append(find_case_image(p))
                except FileNotFoundError:
                    pass
            elif entry.lower().endswith(IMAGE_SUFFIXES):
                paths.append(p)
    if not paths:
        sys.exit('no input images (use --image and/or --input-dir)')
    return list(dict.fromkeys(os.path.abspath(p) for p in paths))     # de-duplicate, keep order


def case_name(path):
    parent = os.path.basename(os.path.dirname(path))
    base = os.path.basename(path)
    for s in IMAGE_SUFFIXES:
        if base.lower().endswith(s):
            base = base[:-len(s)]
            break
    return parent + '__' + base if parent else base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', nargs='*', help='CBCT volume(s): .nrrd / .nii(.gz) / .mha')
    ap.add_argument('--input-dir', help='folder of volumes, or of case sub-folders each holding one volume')
    ap.add_argument('--global-ckpt', required=True, help='stage-1 tooth-region weights (bestmodel.pth)')
    ap.add_argument('--local-ckpt', required=True, help='stage-2 landmark weights (Unet_model08.pt)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--coarse-size', type=int, default=72)
    ap.add_argument('--roi-size', type=int, default=128)
    ap.add_argument('--margin', type=int, default=10)
    ap.add_argument('--mask-threshold', type=float, default=0.5)
    ap.add_argument('--heatmap-threshold', type=float, default=0.1)
    ap.add_argument('--save-intermediate', action='store_true', help='also write mask / ROI / heat-map volumes')
    args = ap.parse_args()

    device = torch.device(args.device)
    global_net, local_net = load_models(args.global_ckpt, args.local_ckpt, device)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for path in collect_inputs(args):
        name = case_name(path)
        print('==>', name)
        sitk_image = sitk.ReadImage(path)
        result, inter = run_inference(sitk_image, global_net, local_net, device, args.coarse_size, args.roi_size,
                                      args.margin, args.mask_threshold, args.heatmap_threshold)
        result['image'] = path
        for w in result['warnings']:
            print('   WARNING:', w)

        lm = result['landmarks_lps_mm']
        pts = [lm[n] for n in LANDMARK_NAMES if lm[n] is not None]
        write_markups(os.path.join(args.out_dir, name + '_landmarks.mrk.json'), pts,
                      [n for n in LANDMARK_NAMES if lm[n] is not None])
        with open(os.path.join(args.out_dir, name + '_result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        if args.save_intermediate:
            sitk.WriteImage(sitk.GetImageFromArray(inter['mask']), os.path.join(args.out_dir, name + '_tooth_mask_72.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(inter['roi']), os.path.join(args.out_dir, name + '_roi_128.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(inter['heatmap']), os.path.join(args.out_dir, name + '_heatmap_128.nii.gz'))

        row = {'case': name, 'n_peaks': result['n_peaks'],
               'maxillary_width_mm': result['maxillary_width_mm'], 'mandibular_width_mm': result['mandibular_width_mm']}
        for n in LANDMARK_NAMES:
            for axis, v in zip('xyz', lm[n] or [np.nan] * 3):
                row['%s_%s' % (n, axis)] = v
        rows.append(row)
        print('   maxillary width %.2f mm, mandibular width %.2f mm' % (row['maxillary_width_mm'], row['mandibular_width_mm']))

    with open(os.path.join(args.out_dir, 'results.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('done ->', args.out_dir)


if __name__ == '__main__':
    main()
