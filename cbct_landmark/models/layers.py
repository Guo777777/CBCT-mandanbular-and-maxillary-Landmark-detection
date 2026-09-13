# -*- coding:utf-8 -*-
"""Building blocks shared by the global (tooth-region) U-Net.

History: this file was ``utils_xhy.py`` in the original code base, later renamed
``utils.py``.  Class and attribute names are kept unchanged on purpose: the released
checkpoints store parameters under these names (e.g. ``down_tr64.ops.0.conv1.weight``).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LUConv3D(nn.Module):
    """Conv3d(k=3) -> BatchNorm3d -> activation."""

    def __init__(self, in_channel, out_channel, activation):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channel)

        if activation == 'relu':
            self.activation = nn.ReLU(out_channel)
        elif activation == 'prelu':
            self.activation = nn.PReLU(out_channel)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError('Expected activation: relu, prelu, elu, but got {}'.format(activation))

    def forward(self, x):
        return self.activation(self.bn1(self.conv1(x)))


def _make_nConv3D(in_channel, depth, activation, double_channel=False):
    if double_channel:
        layer1 = LUConv3D(in_channel, 32 * (2 ** (depth + 1)), activation)
        layer2 = LUConv3D(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), activation)
    else:
        layer1 = LUConv3D(in_channel, 32 * (2 ** depth), activation)
        layer2 = LUConv3D(32 * (2 ** depth), 32 * (2 ** depth) * 2, activation)
    return nn.Sequential(layer1, layer2)


class DownTransition3D(nn.Module):
    def __init__(self, in_channel, depth, activation):
        super().__init__()
        self.ops = _make_nConv3D(in_channel, depth, activation)
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


class UpTransition3D(nn.Module):
    def __init__(self, in_channel, out_channel, depth, activation):
        super().__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(in_channel, out_channel, kernel_size=2, stride=2)
        self.ops = _make_nConv3D(in_channel + out_channel // 2, depth, activation, double_channel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        return self.ops(concat)


class OutputTransition3D(nn.Module):
    def __init__(self, in_channel, n_labels, out_activation):
        super().__init__()
        self.final_conv = nn.Conv3d(in_channel, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.out_activation = out_activation

    def forward(self, x):
        if self.out_activation == "softmax":
            return self.softmax(self.final_conv(x))
        elif self.out_activation == "sigmoid":
            return self.sigmoid(self.final_conv(x))
        return self.final_conv(x)
