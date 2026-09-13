# -*- coding:utf-8 -*-
"""Self-contained consistency test (no weights or data needed).

    python tests/test_synthetic_roundtrip.py      # or: pytest tests/

1. both networks accept / return the expected shapes;
2. label generation (landmark -> ROI heat-map) and post-processing (heat-map -> landmark)
   are inverse to each other to within half a voxel;
3. the L1..L4 assignment and the widths are recovered.
"""
import os
import sys
import tempfile

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbct_landmark.data import landmarks_to_heatmap, write_markups, read_case_landmarks  # noqa: E402
from cbct_landmark.models import build_global_net, build_local_net, load_checkpoint  # noqa: E402
from cbct_landmark.postprocess import heatmap_to_landmarks, physical_to_index, assign_landmarks, basal_bone_widths  # noqa: E402
from cbct_landmark.roi import mask_to_roi, crop_and_resize, index_orig_to_roi  # noqa: E402


def test_network_shapes():
    g, l = build_global_net().eval(), build_local_net().eval()
    with torch.no_grad():
        assert tuple(g(torch.rand(1, 1, 24, 24, 24)).shape) == (1, 1, 24, 24, 24)
        assert tuple(l(torch.rand(1, 1, 32, 32, 32)).shape) == (1, 1, 32, 32, 32)


def test_checkpoint_formats():
    with tempfile.TemporaryDirectory() as d:
        g, l = build_global_net(), build_local_net()
        torch.save(g.state_dict(), os.path.join(d, 'g.pth'))
        torch.save({'epoch': 1, 'state_dict': l.state_dict(), 'optimizer_state_dict': {}}, os.path.join(d, 'l.pt'))
        load_checkpoint(build_global_net(), os.path.join(d, 'g.pth'))
        load_checkpoint(build_local_net(), os.path.join(d, 'l.pt'))


def test_label_postprocess_roundtrip(roi_size=128):
    shape = (120, 110, 100)                                     # (z, y, x)
    img = sitk.GetImageFromArray(np.zeros(shape, np.float32))
    img.SetSpacing((0.2, 0.2, 0.2))
    img.SetOrigin((-10.0, -12.0, -8.0))
    truth_idx = {'L1': (80.3, 50.2, 70.4), 'L2': (79.6, 51.1, 30.7), 'L3': (45.2, 55.5, 66.1), 'L4': (44.8, 54.3, 33.9)}
    truth = {k: list(img.TransformContinuousIndexToPhysicalPoint((x, y, z))) for k, (z, y, x) in truth_idx.items()}

    with tempfile.TemporaryDirectory() as d:                   # Slicer JSON round trip
        for k, p in truth.items():
            write_markups(os.path.join(d, k + '.mrk.json'), [p], [k])
        read_back = dict(read_case_landmarks(d))
    assert all(np.allclose(read_back[k], truth[k]) for k in truth)

    mask = np.zeros((72, 72, 72), np.uint8)                     # coarse mask of a 'tooth box'
    mask[18:57, 16:56, 14:58] = 1
    box = mask_to_roi(mask, shape, margin=10)
    centers = [index_orig_to_roi(physical_to_index(img, truth[k]), box, roi_size) for k in truth]
    heatmap = landmarks_to_heatmap(centers, (roi_size,) * 3, radius=5, sigma=2.0)
    assert heatmap.max() == 1.0

    pred, peaks = heatmap_to_landmarks(heatmap, box, img, size=roi_size, threshold=0.1)
    assert len(peaks) == 4
    # one ROI voxel in mm; draw_gaussian truncates the centre to an integer voxel (original behaviour),
    # so the worst case is sqrt(3) voxels -> allow 2
    voxel_mm = 2 * 0.2 * max(hi - lo for lo, hi in box) / roi_size
    for k in truth:
        err = np.linalg.norm(np.asarray(pred[k]) - np.asarray(truth[k]))
        assert err < voxel_mm, (k, err, voxel_mm)
    w_pred, w_true = basal_bone_widths(pred), basal_bone_widths(assign_landmarks(list(truth.values())))
    for key in w_pred:
        assert abs(w_pred[key] - w_true[key]) < voxel_mm

    roi = crop_and_resize(np.zeros(shape, np.float32), box, size=roi_size)
    assert roi.shape == (roi_size,) * 3


if __name__ == '__main__':
    test_network_shapes(); print('network shapes ok')
    test_checkpoint_formats(); print('checkpoint formats ok')
    test_label_postprocess_roundtrip(); print('label <-> post-processing round trip ok')
