"""RadioUNet-style adaptation for controlled UrbanRadio3D comparison.

This is an independently maintained PyTorch adaptation, not an unmodified copy
of the authors' notebook. Report it as "RadioUNet-style (adapted)".
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class RadioUNetBaseline(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.encoders = nn.ModuleList()
        current = in_channels
        for width in widths:
            self.encoders.append(DoubleConv(current, width))
            current = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(widths[-1], widths[-1] * 2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = widths[-1] * 2
        for width in reversed(widths):
            self.upconvs.append(nn.ConvTranspose2d(current, width, 2, stride=2))
            self.decoders.append(DoubleConv(width * 2, width))
            current = width
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.output(x)
