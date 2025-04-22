# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContBatchNorm2D(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            True, self.momentum, self.eps)


class ContBatchNorm3D(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, x):
        if x.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            True, self.momentum, self.eps)


class LUConv2D(nn.Module):
    def __init__(self, in_channel, out_channel, activation):
        super(LUConv2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)

        if activation == 'relu':
            self.activation = nn.ReLU(out_channel)
        elif activation == 'prelu':
            self.activation = nn.PReLU(out_channel)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError('Expected activation: relu, prelu, elu, but got {}'.format(activation))

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


class LUConv3D(nn.Module):
    def __init__(self, in_channel, out_channel, activation):
        super(LUConv3D, self).__init__()
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
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv2D(in_channel, depth, activation, double_channel=False):
    if double_channel:
        layer1 = LUConv2D(in_channel, 32 * (2 ** (depth + 1)), activation)
        layer2 = LUConv2D(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), activation)
    else:
        layer1 = LUConv2D(in_channel, 32 * (2 ** depth), activation)
        layer2 = LUConv2D(32 * (2 ** depth), 32 * (2 ** depth) * 2, activation)
    return nn.Sequential(layer1, layer2)


def _make_nConv3D(in_channel, depth, activation, double_channel=False):
    if double_channel:
        layer1 = LUConv3D(in_channel, 32 * (2 ** (depth + 1)), activation)
        layer2 = LUConv3D(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), activation)
    else:
        layer1 = LUConv3D(in_channel, 32 * (2 ** depth), activation)
        layer2 = LUConv3D(32 * (2 ** depth), 32 * (2 ** depth) * 2, activation)

    return nn.Sequential(layer1, layer2)


class DownTransition2D(nn.Module):
    def __init__(self, in_channel, depth, activation):
        super(DownTransition2D, self).__init__()
        self.ops = _make_nConv2D(in_channel, depth, activation)
        self.maxpool = nn.MaxPool2d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool


class DownTransition3D(nn.Module):
    def __init__(self, in_channel, depth, activation):
        super(DownTransition3D, self).__init__()
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


class UpTransition2D(nn.Module):
    def __init__(self, in_channel, out_channel, depth, activation):
        super(UpTransition2D, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.ops = _make_nConv2D(in_channel + out_channel // 2, depth, activation, double_channel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        out = self.ops(concat)
        return out


class UpTransition3D(nn.Module):
    def __init__(self, in_channel, out_channel, depth, activation):
        super(UpTransition3D, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(in_channel, out_channel, kernel_size=2, stride=2)
        self.ops = _make_nConv3D(in_channel + out_channel // 2, depth, activation, double_channel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        out = self.ops(concat)
        return out


class OutputTransition2D(nn.Module):
    def __init__(self, in_channel, n_labels):
        super(OutputTransition2D, self).__init__()
        self.final_conv = nn.Conv2d(in_channel, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out


class OutputTransition3D(nn.Module):
    def __init__(self, in_channel, n_labels, out_activation):
        super(OutputTransition3D, self).__init__()
        self.final_conv = nn.Conv3d(in_channel, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.out_activation = out_activation

    def forward(self, x):
        if self.out_activation == "softmax":
            out = self.softmax(self.final_conv(x))
        elif self.out_activation == "sigmoid":
            out = self.sigmoid(self.final_conv(x))
        else:
            out = self.final_conv(x)
        return out


class Classify(nn.Module):
    def __init__(self, in_dim, mid_dim_1, mid_dim_2, out_dim_cl, out_dim_ud, out_dim_rl):  ###512,256,96,8,1,1
        super(Classify, self).__init__()
        self.linear_layer_1_cl = nn.Sequential(
            nn.Linear(in_dim, mid_dim_1),
            nn.ReLU(mid_dim_1),
            nn.Dropout(p=0.5)
        )
        self.linear_layer_2_cl = nn.Sequential(
            nn.Linear(mid_dim_1, mid_dim_2),
            nn.ReLU(mid_dim_2),
            nn.Dropout(p=0.5)
        )
        self.linear_layer_1_ud = nn.Sequential(
            nn.Linear(in_dim, mid_dim_1),
            nn.ReLU(mid_dim_1),
            nn.Dropout(p=0.5)
        )
        self.linear_layer_2_ud = nn.Sequential(
            nn.Linear(mid_dim_1, mid_dim_2),
            nn.ReLU(mid_dim_2),
            nn.Dropout(p=0.5)
        )
        self.linear_layer_1_rl = nn.Sequential(
            nn.Linear(in_dim, mid_dim_1),
            nn.ReLU(mid_dim_1),
            nn.Dropout(p=0.5)
        )
        self.linear_layer_2_rl = nn.Sequential(
            nn.Linear(mid_dim_1, mid_dim_2),
            nn.ReLU(mid_dim_2),
            nn.Dropout(p=0.5)
        )
        self.layer_out_cl = nn.Linear(mid_dim_2, out_dim_cl)
        self.layer_out_ud = nn.Linear(mid_dim_2, out_dim_ud)
        self.layer_out_rl = nn.Linear(mid_dim_2, out_dim_rl)

    def forward(self, inputs):
        self.gap = torch.mean(inputs, dim=(2, 3, 4))
        self.linear_in_cl = self.linear_layer_1_cl(self.gap)
        self.linear_mid_cl = self.linear_layer_2_cl(self.linear_in_cl)
        self.outputs_cl = self.layer_out_cl(self.linear_mid_cl)
        self.linear_in_ud = self.linear_layer_1_ud(self.gap)
        self.linear_mid_ud = self.linear_layer_2_ud(self.linear_in_ud)
        self.outputs_ud = self.layer_out_ud(self.linear_mid_ud)
        self.linear_in_rl = self.linear_layer_1_rl(self.gap)
        self.linear_mid_rl = self.linear_layer_2_rl(self.linear_in_rl)
        self.outputs_rl = self.layer_out_rl(self.linear_mid_rl)
        return self.outputs_cl, self.outputs_ud, self.outputs_rl


if __name__ == '__main__':
    net = Classify(256, 512, 256, 4096, 8)
    import numpy as np

    x = torch.from_numpy(np.zeros((2, 256, 28, 14, 14))).float()
    print(net)
    y = net(x)
    print(y)
