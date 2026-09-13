#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Step 2 - stage 1: coarse tooth-region mask for every case (whole CBCT -> 72^3).

History: ``tooth_region_pred.py``.  The whole volume is resampled to 72x72x72, range
normalised, clipped to the 1st/99th percentile and min-max scaled, pushed through
``UNet3D_simple`` and thresholded at 0.5.  The 72^3 binary mask is saved as
``<out_dir>/<case>_pred.nii.gz`` and consumed by step 3.

    python scripts/2_predict_tooth_region.py --data-dir /data/imageStandardData \
        --lists /data/train.txt /data/valid.txt /data/test.txt \
        --ckpt weights/tooth_region_bestmodel.pth --out-dir /data/tooth_region
"""
import argparse
import os
import sys

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbct_landmark.data import WholeVolumeDataset, read_case_list, quantile_minmax  # noqa: E402
from cbct_landmark.models import build_global_net, load_checkpoint  # noqa: E402


def collect_cases(args):
    if args.lists:
        cases = [c for p in args.lists for c in read_case_list(p)]
    else:
        cases = sorted(d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)))
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--ckpt', required=True, help='tooth-region checkpoint, UNet3D_simple state_dict (train your own; the paper weights are lost)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--lists', nargs='*', default=None, help='train.txt valid.txt test.txt (default: every case folder)')
    ap.add_argument('--size', type=int, default=72)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--num-workers', type=int, default=2)
    args = ap.parse_args()

    device = torch.device(args.device)
    net = load_checkpoint(build_global_net(), args.ckpt, device).to(device).eval()
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = WholeVolumeDataset(args.data_dir, collect_cases(args), target_size=(args.size,) * 3)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    with torch.no_grad():
        for image, (case,) in loader:
            x = torch.from_numpy(quantile_minmax(image.numpy()[0]))[None].float().to(device)
            pred = net(x)[0, 0].cpu().numpy()
            mask = (pred >= args.threshold).astype(np.uint8)
            sitk.WriteImage(sitk.GetImageFromArray(mask), os.path.join(args.out_dir, case + '_pred.nii.gz'))
            print(case, 'mask voxels:', int(mask.sum()))
    print('done:', len(dataset), 'cases ->', args.out_dir)


if __name__ == '__main__':
    main()
