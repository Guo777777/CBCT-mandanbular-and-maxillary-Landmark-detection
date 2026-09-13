#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Step 6 - heat-maps -> landmarks -> metrics (MRE, SDR, basal bone width errors).

History: ``json_process_test0.py`` / ``json_process_test2.py``.  For every ``<case>_pred.nii.gz``:
threshold 0.1, connected components, 4 largest centroids, map back to the original grid with
``<case>_roi.json`` and to LPS mm, assign L1..L4 (see ``cbct_landmark.postprocess``), compare
with the ground-truth ``*.mrk.json`` of the case and write:

    <out_dir>/landmarks.csv   per case: L1..L4 pred & GT (mm), radial errors, widths
    <out_dir>/summary.json    MRE per landmark, SDR @2/2.5/3/4 mm, width errors
    <out_dir>/json/<case>_pred.mrk.json   predicted fiducials for 3D Slicer

    python scripts/6_evaluate.py --pred-dir /data/test_save --roi-dir /data/four_point_save_128 \
        --data-dir /data/imageStandardData --out-dir /data/eval
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbct_landmark.data import read_case_image, read_case_landmarks, write_markups  # noqa: E402
from cbct_landmark.postprocess import (heatmap_to_landmarks, assign_landmarks, basal_bone_widths,  # noqa: E402
                                       euclid, LANDMARK_NAMES)
from cbct_landmark.roi import load_roi_meta  # noqa: E402


def evaluate_case(case, args):
    sitk_image = read_case_image(os.path.join(args.data_dir, case))
    meta = load_roi_meta(os.path.join(args.roi_dir, case + '_roi.json'))
    heatmap = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.pred_dir, case + '_pred.nii.gz')))
    pred, peaks = heatmap_to_landmarks(heatmap, meta['box_zyx'], sitk_image, size=meta['roi_size'],
                                       threshold=args.threshold)
    gt_points = [pos for _, pos in read_case_landmarks(os.path.join(args.data_dir, case))]
    gt = assign_landmarks(gt_points)

    row = {'case': case, 'n_peaks': len(peaks)}
    for n in LANDMARK_NAMES:
        for axis, v in zip('xyz', pred[n] or [np.nan] * 3):
            row['%s_pred_%s' % (n, axis)] = v
        for axis, v in zip('xyz', gt[n] or [np.nan] * 3):
            row['%s_gt_%s' % (n, axis)] = v
        row['%s_radial_error_mm' % n] = euclid(pred[n], gt[n])
        # error to the closest GT landmark of any name (what the original script reported)
        row['%s_nearest_gt_error_mm' % n] = min([euclid(pred[n], g) for g in gt_points] or [np.nan])
    wp, wg = basal_bone_widths(pred), basal_bone_widths(gt)
    for k in wp:
        row[k + '_pred'] = wp[k]
        row[k + '_gt'] = wg[k]
        row[k + '_abs_error'] = abs(wp[k] - wg[k])

    if args.save_json:
        os.makedirs(os.path.join(args.out_dir, 'json'), exist_ok=True)
        pts = [pred[n] for n in LANDMARK_NAMES if pred[n] is not None]
        labels = [n for n in LANDMARK_NAMES if pred[n] is not None]
        write_markups(os.path.join(args.out_dir, 'json', case + '_pred.mrk.json'), pts, labels)
    return row


def summarise(rows, radii=(2.0, 2.5, 3.0, 4.0)):
    summary = {'n_cases': len(rows)}
    all_err = []
    for n in LANDMARK_NAMES:
        err = np.array([r['%s_radial_error_mm' % n] for r in rows], dtype=float)
        err = err[~np.isnan(err)]
        all_err.append(err)
        summary[n] = {'MRE_mm': float(err.mean()) if err.size else None, 'SD_mm': float(err.std()) if err.size else None}
    all_err = np.concatenate(all_err) if all_err else np.array([])
    summary['all'] = {'MRE_mm': float(all_err.mean()) if all_err.size else None,
                      'SD_mm': float(all_err.std()) if all_err.size else None,
                      'SDR': {'%.1fmm' % r: float((all_err <= r).mean()) if all_err.size else None for r in radii}}
    for k in ('maxillary_width_mm', 'mandibular_width_mm'):
        e = np.array([r[k + '_abs_error'] for r in rows], dtype=float)
        e = e[~np.isnan(e)]
        summary[k] = {'mean_abs_error_mm': float(e.mean()) if e.size else None, 'SD_mm': float(e.std()) if e.size else None}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred-dir', required=True, help='output of step 5 (<case>_pred.nii.gz)')
    ap.add_argument('--roi-dir', required=True, help='output of step 3 (<case>_roi.json)')
    ap.add_argument('--data-dir', required=True, help='imageStandardData (image geometry + GT *.mrk.json)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--threshold', type=float, default=0.1)
    ap.add_argument('--no-json', dest='save_json', action='store_false')
    args = ap.parse_args()

    cases = sorted(f[:-len('_pred.nii.gz')] for f in os.listdir(args.pred_dir) if f.endswith('_pred.nii.gz'))
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for case in cases:
        try:
            row = evaluate_case(case, args)
        except Exception as e:
            print('ERROR', case, ':', e)
            continue
        rows.append(row)
        print('%s  peaks=%d  radial errors (mm): %s  width abs err: max %.2f mand %.2f' % (
            case, row['n_peaks'], ' '.join('%.2f' % row['%s_radial_error_mm' % n] for n in LANDMARK_NAMES),
            row['maxillary_width_mm_abs_error'], row['mandibular_width_mm_abs_error']))

    if rows:
        with open(os.path.join(args.out_dir, 'landmarks.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    summary = summarise(rows)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
