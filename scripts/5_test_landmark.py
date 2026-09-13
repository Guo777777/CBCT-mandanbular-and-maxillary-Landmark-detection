#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Step 5 - predict ROI heat-maps for a case list with a trained stage-2 checkpoint.

History: ``3_test.py``.  Writes ``<out_dir>/<case>_pred.nii.gz`` (float heat-map, 128^3) and,
when labels exist, ``<case>_label.nii.gz`` for side-by-side viewing in ITK-SNAP / 3D Slicer.
Landmark coordinates and metrics are produced by step 6.

    python scripts/5_test_landmark.py --roi-dir /data/four_point_save_128 --list /data/test.txt \
        --ckpt weights/landmark_Unet_model08.pt --out-dir /data/test_save
"""
import argparse
import os
import sys
import time

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbct_landmark.data import LandmarkROIDataset, read_case_list  # noqa: E402
from cbct_landmark.models import build_local_net, load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roi-dir', required=True)
    ap.add_argument('--list', required=True, help='test.txt')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--num-workers', type=int, default=2)
    args = ap.parse_args()

    device = torch.device(args.device)
    net = load_checkpoint(build_local_net(), args.ckpt, device).to(device).eval()
    os.makedirs(args.out_dir, exist_ok=True)
    dataset = LandmarkROIDataset(args.roi_dir, read_case_list(args.list), require_heatmap=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    t0 = time.time()
    with torch.no_grad():
        for image, heatmap, (case,) in loader:
            pred = net(image.float().to(device))[0, 0].cpu().numpy()
            sitk.WriteImage(sitk.GetImageFromArray(pred), os.path.join(args.out_dir, case + '_pred.nii.gz'))
            if dataset.heatmap_files(case):
                sitk.WriteImage(sitk.GetImageFromArray(heatmap.numpy()[0, 0]),
                                os.path.join(args.out_dir, case + '_label.nii.gz'))
            print(case, 'saved')
    print('%d cases in %.1f s' % (len(dataset), time.time() - t0))


if __name__ == '__main__':
    main()
