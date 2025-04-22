# !/usr/bin/env python
# -*- coding:utf-8 -*-

# torch模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# 自定义模块
from utils_xhy import *


class UNet2D(nn.Module):
    def __init__(self, n_class=1, activation='relu'):
        super(UNet2D, self).__init__()

        self.down_tr64 = DownTransition2D(1, 0, activation)
        self.down_tr128 = DownTransition2D(64, 1, activation)
        self.down_tr256 = DownTransition2D(128, 2, activation)
        self.down_tr512 = DownTransition2D(256, 3, activation)

        self.up_tr256 = UpTransition2D(512, 512, 2, activation)
        self.up_tr128 = UpTransition2D(256, 256, 1, activation)
        self.up_tr64 = UpTransition2D(128, 128, 0, activation)
        self.out_tr = OutputTransition2D(64, n_class)

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128, self.skip_out128 = self.down_tr128(self.out64)
        self.out256, self.skip_out256 = self.down_tr256(self.out128)
        self.out512, self.skip_out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512, self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)

        return self.out


class UNet3D(nn.Module):
    def __init__(self, n_class=1, activation='relu', out_activation='sigmoid'):
        super(UNet3D, self).__init__()
        self.down_tr64 = DownTransition3D(1, 0, activation)
        self.down_tr128 = DownTransition3D(64, 1, activation)
        self.down_tr256 = DownTransition3D(128, 2, activation)
        self.down_tr512 = DownTransition3D(256, 3, activation)
        self.up_tr256 = UpTransition3D(512, 512, 2, activation)
        self.up_tr128 = UpTransition3D(256, 256, 1, activation)
        self.up_tr64 = UpTransition3D(128, 128, 0, activation)
        self.out_tr = OutputTransition3D(64, n_class, out_activation)

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128, self.skip_out128 = self.down_tr128(self.out64)
        self.out256, self.skip_out256 = self.down_tr256(self.out128)
        self.out512, self.skip_out512 = self.down_tr512(self.out256)
        self.out_up_256 = self.up_tr256(self.out512, self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)
        return self.out, self.skip_out512


class UNet3D_simple(nn.Module):
    def __init__(self, n_class=1, activation='relu', out_activation='sigmoid'):
        super(UNet3D_simple, self).__init__()
        self.down_tr64 = DownTransition3D(1, 0, activation)
        self.down_tr128 = DownTransition3D(64, 1, activation)
        self.down_tr256 = DownTransition3D(128, 2, activation)
        self.down_tr512 = DownTransition3D(256, 3, activation)
        self.up_tr256 = UpTransition3D(512, 512, 2, activation)
        self.up_tr128 = UpTransition3D(256, 256, 1, activation)
        self.up_tr64 = UpTransition3D(128, 128, 0, activation)
        self.out_tr = OutputTransition3D(64, n_class, out_activation)

    def forward(self, input):
        self.out64, self.skip_out64 = self.down_tr64(input)
        self.out128, self.skip_out128 = self.down_tr128(self.out64)
        self.out256, self.skip_out256 = self.down_tr256(self.out128)
        # self.out512, self.skip_out512 = self.down_tr512(self.out256)
        # self.out_up_256 = self.up_tr256(self.out512, self.skip_out256)
        self.out_up_128 = self.up_tr128(self.skip_out256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)
        return self.out


class Net(nn.Module):
    def __init__(self, out_chanel, out_activation, classify_sign=False):
        super(Net, self).__init__()
        self.Segment = UNet3D(n_class=out_chanel, out_activation=out_activation)
        self.classify_sign = classify_sign
        if self.classify_sign:
            self.classify = Classify(512, 256, 96, 8, 1, 1)

    def forward(self, input):
        self.seg_out, self.cla_in = self.Segment(x=input)
        if self.classify_sign:
            self.cla_out_cl, self.cla_out_ud, self.cla_out_rl = self.classify(self.cla_in)
            return self.seg_out, self.cla_out_cl, self.cla_out_ud, self.cla_out_rl
        else:
            return self.seg_out


class PreNet(nn.Module):
    def __init__(self, in_channel=1):
        super(PreNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=5, stride=2, padding=2)
        self.activation1 = nn.ReLU(16)
        self.activation2 = nn.ReLU(32)
        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.conv4 = nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.activation1(x1)
        x2 = self.conv2(x1)
        x2 = self.activation1(x2)
        x3 = self.upconv3(x2)
        x4 = self.conv4(x3)
        x4 = self.activation(x4)
        return x4


class MAR_DetNet(nn.Module):
    def __init__(self):
        super(MAR_DetNet, self).__init__()
        self.Preprocess = PreNet()
        self.Segment = UNet3D_simple()

    def forward(self, input):
        weight = self.Preprocess(x=input)
        x = weight * input
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        out = self.Segment(input=x)
        return out


#
if __name__ == '__main__':
    net = MAR_DetNet()
    x = torch.randn((2, 1, 64, 96, 128))
    y = net(x)
    print(y)
