# -*- coding:utf-8 -*-
"""Heat-map focal loss (CornerNet / CenterNet style) used to train the landmark network.

History: this was ``heatmap.py`` in the original code base.
"""
import torch


def focal_loss(predict, target):
    """``predict``: sigmoid heat-map in (0, 1); ``target``: Gaussian heat-map with peaks == 1.
    Voxels with target > 0.9 are positives; negatives are down-weighted by (1 - target)^4."""
    pos_inds = target.gt(0.9)
    neg_inds = target.lt(0.9)
    neg_weights = torch.pow(1 - target[neg_inds], 4)

    pos_pred = predict[pos_inds]
    neg_pred = predict[neg_inds]

    pos_loss = torch.log2(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log2(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos
