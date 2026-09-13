# -*- coding:utf-8 -*-
"""Stage 1 ("global") network: coarse tooth-region segmentation on the whole CBCT.

History: this file was ``UNet_xhy.py`` in the original code base (renamed ``UNet_global.py``
on GitHub in April 2025).  ``tooth_region_pred.py`` instantiated ``UNet3D_simple(n_class=1)``
and loaded ``tooth_best_model/bestmodel.pth`` (a plain ``state_dict``).

Only the classes needed for the released pipeline are kept here; the 2-D variant, ``PreNet``,
``MAR_DetNet`` and the classification head that lived in the same file are available in the git
history but were never used by the landmark pipeline.
"""
import torch.nn as nn

from .layers import DownTransition3D, UpTransition3D, OutputTransition3D


class UNet3D(nn.Module):
    """Full 4-level 3-D U-Net (64-128-256-512).  Returns (output, bottleneck features)."""

    def __init__(self, n_class=1, activation='relu', out_activation='sigmoid'):
        super().__init__()
        self.down_tr64 = DownTransition3D(1, 0, activation)
        self.down_tr128 = DownTransition3D(64, 1, activation)
        self.down_tr256 = DownTransition3D(128, 2, activation)
        self.down_tr512 = DownTransition3D(256, 3, activation)
        self.up_tr256 = UpTransition3D(512, 512, 2, activation)
        self.up_tr128 = UpTransition3D(256, 256, 1, activation)
        self.up_tr64 = UpTransition3D(128, 128, 0, activation)
        self.out_tr = OutputTransition3D(64, n_class, out_activation)

    def forward(self, x):
        out64, skip_out64 = self.down_tr64(x)
        out128, skip_out128 = self.down_tr128(out64)
        out256, skip_out256 = self.down_tr256(out128)
        out512, skip_out512 = self.down_tr512(out256)
        out_up_256 = self.up_tr256(out512, skip_out256)
        out_up_128 = self.up_tr128(out_up_256, skip_out128)
        out_up_64 = self.up_tr64(out_up_128, skip_out64)
        return self.out_tr(out_up_64), skip_out512


class UNet3D_simple(nn.Module):
    """3-level 3-D U-Net used for tooth-region localisation (input 72x72x72, sigmoid output).

    NOTE: ``down_tr512`` and ``up_tr256`` are constructed but *not* used in ``forward`` (this is
    how the model was trained).  They must stay so that ``load_state_dict(strict=True)`` matches
    the released checkpoint, which contains their (untrained) parameters.
    """

    def __init__(self, n_class=1, activation='relu', out_activation='sigmoid'):
        super().__init__()
        self.down_tr64 = DownTransition3D(1, 0, activation)
        self.down_tr128 = DownTransition3D(64, 1, activation)
        self.down_tr256 = DownTransition3D(128, 2, activation)
        self.down_tr512 = DownTransition3D(256, 3, activation)      # unused, kept for checkpoint compatibility
        self.up_tr256 = UpTransition3D(512, 512, 2, activation)     # unused, kept for checkpoint compatibility
        self.up_tr128 = UpTransition3D(256, 256, 1, activation)
        self.up_tr64 = UpTransition3D(128, 128, 0, activation)
        self.out_tr = OutputTransition3D(64, n_class, out_activation)

    def forward(self, x):
        out64, skip_out64 = self.down_tr64(x)
        out128, skip_out128 = self.down_tr128(out64)
        out256, skip_out256 = self.down_tr256(out128)
        out_up_128 = self.up_tr128(skip_out256, skip_out128)
        out_up_64 = self.up_tr64(out_up_128, skip_out64)
        return self.out_tr(out_up_64)
