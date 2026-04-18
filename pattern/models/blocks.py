"""
=============================================================================
SCRIPT NAME: blocks.py
=============================================================================
DESCRIPTION:
Single CNN building block from PRD §6.
Each block: Conv2d → BatchNorm2d → LeakyReLU → MaxPool2d
=============================================================================
"""

import torch.nn as nn


class ConvBlock(nn.Module):
    """
    One convolutional building block (PRD §6):
        Conv2d → BatchNorm2d → LeakyReLU → MaxPool2d

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count.
        conv_kernel:  (height, width) kernel size.
        conv_stride:  (height, width) stride.
        conv_padding: (height, width) padding (symmetric).
        conv_dilation:(height, width) dilation.
        pool_kernel:  (height, width) MaxPool kernel.
        leaky_slope:  LeakyReLU negative slope.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel: tuple  = (5, 3),
        conv_stride: tuple  = (3, 1),
        conv_padding: tuple = (12, 1),
        conv_dilation: tuple= (2, 1),
        pool_kernel: tuple  = (2, 1),
        leaky_slope: float  = 0.01,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size  = conv_kernel,
                stride       = conv_stride,
                padding      = conv_padding,
                dilation     = conv_dilation,
                bias         = False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel),
        )

    def forward(self, x):
        return self.block(x)
