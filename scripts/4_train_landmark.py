#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Step 4 - train the stage-2 landmark network on the 128^3 ROIs.

History: ``2_train_and_valid.py``.  Settings reproduced from the original script:
``UNet3d(n_class=1)``, heat-map focal loss, Adam lr 1e-3, StepLR(step 40, gamma 0.9),
batch size 1, augmentation = contrast (0.3-3.0) + mirroring of the two in-plane axes
(rotation / scaling via ``SpatialTransform`` was defined but disabled; enable with
``--rotation``), early stopping on validation loss (patience 15).  The best model is written
as ``{'epoch', 'state_dict', 'optimizer_state_dict'}`` to ``<out_dir>/<ckpt_name>``.

    python scripts/4_train_landmark.py --roi-dir /data/four_point_save_128 \
        --train-list /data/train.txt --valid-list /data/valid.txt --out-dir /model/UNet3d_stage1
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbct_landmark.data import LandmarkROIDataset, DatasetTransforms, read_case_list  # noqa: E402
from cbct_landmark.losses import focal_loss  # noqa: E402
from cbct_landmark.models import build_local_net  # noqa: E402


def build_augmentation(patch_size, rotation=False):
    from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
    from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
    transforms = [ContrastAugmentationTransform((0.3, 3.), data_key='data', preserve_range=True)]
    if rotation:
        transforms.append(SpatialTransform(patch_size, tuple(p // 2 for p in patch_size),
                                           do_elastic_deform=False, do_rotation=True,
                                           angle_x=(0, 0.1 * np.pi), angle_y=(0, 0.1 * np.pi),
                                           do_scale=True, scale=(0.9, 1.1),
                                           border_mode_data='constant', border_cval_data=0, order_data=1,
                                           random_crop=False))
    transforms.append(MirrorTransform(axes=(1, 2)))
    return DatasetTransforms(transforms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roi-dir', required=True, help='output of step 3')
    ap.add_argument('--train-list', required=True)
    ap.add_argument('--valid-list', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--ckpt-name', default='Unet_model08.pt')
    ap.add_argument('--resume', default=None, help='checkpoint to continue from')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--step-size', type=int, default=40)
    ap.add_argument('--gamma', type=float, default=0.9)
    ap.add_argument('--max-epochs', type=int, default=5000)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--no-augment', action='store_true')
    ap.add_argument('--rotation', action='store_true', help='also enable SpatialTransform (rotation + scale)')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)

    model = build_local_net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state.get('epoch', 0)
        print('resumed from', args.resume, 'epoch', start_epoch)

    train_set = LandmarkROIDataset(args.roi_dir, read_case_list(args.train_list))
    valid_set = LandmarkROIDataset(args.roi_dir, read_case_list(args.valid_list))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
    patch_size = tuple(train_set[0][0].shape[1:])
    augment = None if args.no_augment else build_augmentation(patch_size, rotation=args.rotation)

    log_path = os.path.join(args.out_dir, 'train_log.csv')
    with open(log_path, 'a', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'valid_loss', 'lr', 'seconds'])

    best_loss, no_improve = float('inf'), 0
    for epoch in range(start_epoch, args.max_epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        for image, heatmap, _ in tqdm(train_loader, desc='epoch %d' % (epoch + 1), leave=False):
            if augment is not None:
                batch = augment({'data': image.numpy(), 'seg': heatmap.numpy()})
                image, heatmap = torch.from_numpy(batch['data']), torch.from_numpy(batch['seg'])
            image, heatmap = image.float().to(device), heatmap.float().to(device)
            loss = focal_loss(model(image), heatmap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for image, heatmap, _ in valid_loader:
                image, heatmap = image.float().to(device), heatmap.float().to(device)
                valid_losses.append(focal_loss(model(image), heatmap).item())

        train_loss, valid_loss = float(np.mean(train_losses)), float(np.mean(valid_losses))
        lr = optimizer.param_groups[0]['lr']
        print('Epoch %d  train %.4f  valid %.4f  lr %.2e  (%.0fs)' % (epoch + 1, train_loss, valid_loss, lr, time.time() - t0))
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, train_loss, valid_loss, lr, round(time.time() - t0)])

        if valid_loss < best_loss:
            print('  validation loss improved %.4f -> %.4f, saving %s' % (best_loss, valid_loss, ckpt_path))
            best_loss, no_improve = valid_loss, 0
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
        else:
            no_improve += 1
            print('  no improvement for %d epoch(s) (best %.4f)' % (no_improve, best_loss))
        if no_improve >= args.patience:
            print('Early stopping')
            break


if __name__ == '__main__':
    main()
