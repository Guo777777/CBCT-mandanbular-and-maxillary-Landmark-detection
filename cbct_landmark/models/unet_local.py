# -*- coding:utf-8 -*-
"""Stage 2 ("local") network: landmark heat-map regression on the 128x128x128 tooth ROI.

History: this file was ``UNet.py`` in the original code base (renamed ``UNet_local.py`` on
GitHub in April 2025).  ``2_train_and_valid.py`` / ``3_test.py`` instantiated
``UNet3d(n_class=1, act='relu')`` and saved/loaded ``UNet3d_stage1/Unet_model08.pt``
(a dict with keys ``epoch``, ``state_dict``, ``optimizer_state_dict``).

Differences from the global network: InstanceNorm instead of BatchNorm, trilinear
up-sampling instead of transposed convolutions.  Names are unchanged for checkpoint
compatibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LUConv(nn.Module):
    """Conv3d(k=3) -> InstanceNorm3d -> activation."""

    def __init__(self, in_chan, out_chan, act):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError('Expected act: relu, prelu, elu, but got {}'.format(act))

    def forward(self, x):
        return self.activation(self.bn1(self.conv1(x)))


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth + 1)), act)
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), act)
    else:
        layer1 = LUConv(in_channel, 32 * (2 ** depth), act)
        layer2 = LUConv(32 * (2 ** depth), 32 * (2 ** depth) * 2, act)
    return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act):
        super().__init__()
        self.ops = _make_nConv(in_channel, depth, act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth, act):
        super().__init__()
        self.depth = depth
        self.ops = _make_nConv(inChans + outChans // 2, depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        concat = torch.cat((out_up_conv, skip_x), 1)
        return self.ops(concat)


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):
        super().__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.final_conv(x))


class UNet3d(nn.Module):
    """4-level 3-D U-Net, sigmoid output.  ``n_class=1``: one heat-map channel containing all
    four landmark Gaussians (see README, "Heat-map format")."""

    def __init__(self, n_class=1, act='relu'):
        super().__init__()
        self.down_tr64 = DownTransition(1, 0, act)
        self.down_tr128 = DownTransition(64, 1, act)
        self.down_tr256 = DownTransition(128, 2, act)
        self.down_tr512 = DownTransition(256, 3, act)

        self.up_tr256 = UpTransition(512, 512, 2, act)
        self.up_tr128 = UpTransition(256, 256, 1, act)
        self.up_tr64 = UpTransition(128, 128, 0, act)
        self.out_tr = OutputTransition(64, n_class)

    def forward(self, x):
        out64, skip_out64 = self.down_tr64(x)
        out128, skip_out128 = self.down_tr128(out64)
        out256, skip_out256 = self.down_tr256(out128)
        out512, skip_out512 = self.down_tr512(out256)

        out_up_256 = self.up_tr256(out512, skip_out256)
        out_up_128 = self.up_tr128(out_up_256, skip_out128)
        out_up_64 = self.up_tr64(out_up_128, skip_out64)
        return self.out_tr(out_up_64)
