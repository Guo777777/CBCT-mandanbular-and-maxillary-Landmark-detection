#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Step 3 - crop the tooth ROI from the original CBCT, resample to 128^3, build heat-map labels.

History: the crop / heat-map part of ``json_process_test0.py`` (whose output folder was
``four_point_save_128``).  For every case:

    box  = bounding box of the 72^3 mask, scaled to the original grid, dilated by 10 voxels
    roi  = volume[box] resampled to 128x128x128            -> <case>_image.nii.gz
    meta = box + original shape                            -> <case>_roi.json
    heat = max over landmarks of Gaussian(r=5, sigma=2)    -> <case>_heatmap.nii.gz  (ONE channel)

Landmarks are read from every ``*.mrk.json`` in the case folder (3D Slicer, LPS) and mapped
physical -> voxel with the image geometry.  Use ``--no-heatmap`` for un-annotated cases.

    python scripts/3_crop_tooth_roi.py --data-dir /data/imageStandardData \
        --mask-dir /data/tooth_region --out-dir /data/four_point_save_128
"""
import argparse
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbct_landmark.data import read_case_image, read_case_landmarks, landmarks_to_heatmap, read_case_list  # noqa: E402
from cbct_landmark.roi import mask_to_roi, crop_and_resize, index_orig_to_roi, save_roi_meta  # noqa: E402
from cbct_landmark.postprocess import physical_to_index  # noqa: E402


def process_case(case, args):
    case_dir = os.path.join(args.data_dir, case)
    sitk_image = read_case_image(case_dir)
    volume = sitk.GetArrayFromImage(sitk_image)

    mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.mask_dir, case + '_pred.nii.gz')))
    box = mask_to_roi(mask, volume.shape, margin=args.margin)
    roi = crop_and_resize(volume.astype(np.float32), box, size=args.size)

    sitk.WriteImage(sitk.GetImageFromArray(roi), os.path.join(args.out_dir, case + '_image.nii.gz'))
    save_roi_meta(os.path.join(args.out_dir, case + '_roi.json'), box, volume.shape, args.size,
                  extra={'spacing': list(sitk_image.GetSpacing()), 'origin': list(sitk_image.GetOrigin()),
                         'direction': list(sitk_image.GetDirection())})

    if args.no_heatmap:
        return box, 0
    landmarks = read_case_landmarks(case_dir)
    centers = []
    for label, pos in landmarks:
        idx = physical_to_index(sitk_image, pos)                 # (z, y, x) on the original grid
        if not all(lo <= v < hi for v, (lo, hi) in zip(idx, box)):
            print('  WARNING %s: landmark %s at %s lies outside the ROI box %s' % (case, label, np.round(idx, 1), box))
        centers.append(index_orig_to_roi(idx, box, args.size))
    if len(centers) != args.n_landmarks:
        print('  WARNING %s: expected %d landmarks, found %d' % (case, args.n_landmarks, len(centers)))
    heatmap = landmarks_to_heatmap(centers, (args.size,) * 3, radius=args.radius, sigma=args.sigma)
    sitk.WriteImage(sitk.GetImageFromArray(heatmap), os.path.join(args.out_dir, case + '_heatmap.nii.gz'))
    return box, len(centers)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, help='imageStandardData (case folders with image + *.mrk.json)')
    ap.add_argument('--mask-dir', required=True, help='72^3 masks from step 2 (<case>_pred.nii.gz)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--lists', nargs='*', default=None, help='restrict to these case lists (default: every mask)')
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--margin', type=int, default=10)
    ap.add_argument('--radius', type=int, default=5)
    ap.add_argument('--sigma', type=float, default=2.0)
    ap.add_argument('--n-landmarks', type=int, default=4)
    ap.add_argument('--no-heatmap', action='store_true', help='images only (no annotations available)')
    args = ap.parse_args()

    if args.lists:
        cases = [c for p in args.lists for c in read_case_list(p)]
    else:
        cases = sorted(f[:-len('_pred.nii.gz')] for f in os.listdir(args.mask_dir) if f.endswith('_pred.nii.gz'))
    os.makedirs(args.out_dir, exist_ok=True)
    for case in cases:
        try:
            box, n = process_case(case, args)
            print(case, 'box(z,y,x)=', box, 'landmarks=', n)
        except Exception as e:  # keep going, report at the end
            print('ERROR', case, ':', e)
    print('done ->', args.out_dir)


if __name__ == '__main__':
    main()
