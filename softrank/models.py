"""ResUNet backbone — Zhang, Liu, Wang (2018) variant used in the paper.

The architecture matches Elezi et al. 2026 [13]:
    – 3 down-sampling residual blocks → bridge → 3 up-sampling residual blocks
    – pre-activation Conv–BN–ReLU residual cells
    – sigmoid head producing a single-channel score map ŷ ∈ [0, 1]

Default `filters=[16,16,16,16]` is the paper-default capacity.
"""

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn


class ResidualConv(nn.Module):
    """Pre-activation residual cell used at every encoder/decoder level."""

    def __init__(self, in_dim: int, out_dim: int, stride: int, padding: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, kernel_size=3,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3,
                      stride=stride, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x) + self.skip(x)


class Upsample(nn.Module):
    """ConvTranspose2d wrapper for spatial up-sampling."""

    def __init__(self, in_dim: int, out_dim: int, kernel: int, stride: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_dim, out_dim,
                                           kernel_size=kernel, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class ResUNet(nn.Module):
    """3-level Residual U-Net producing a 1-channel sigmoid score map."""

    def __init__(self, in_channels: int, out_channels: int = 1,
                 filters: List[int] = (16, 16, 16, 16)):
        super().__init__()
        f = list(filters)

        # ---------- input cell ----------
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, f[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(f[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(f[0], f[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Conv2d(in_channels, f[0], kernel_size=3, padding=1)

        # ---------- encoder ----------
        self.down1 = ResidualConv(f[0], f[1], stride=2, padding=1)
        self.down2 = ResidualConv(f[1], f[2], stride=2, padding=1)

        # ---------- bridge ----------
        self.bridge = ResidualConv(f[2], f[3], stride=2, padding=1)

        # ---------- decoder ----------
        self.up1 = Upsample(f[3], f[3], kernel=2, stride=2)
        self.dec1 = ResidualConv(f[3] + f[2], f[2], stride=1, padding=1)

        self.up2 = Upsample(f[2], f[2], kernel=2, stride=2)
        self.dec2 = ResidualConv(f[2] + f[1], f[1], stride=1, padding=1)

        self.up3 = Upsample(f[1], f[1], kernel=2, stride=2)
        self.dec3 = ResidualConv(f[1] + f[0], f[0], stride=1, padding=1)

        # ---------- head ----------
        self.head = nn.Sequential(
            nn.Conv2d(f[0], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # bridge
        x4 = self.bridge(x3)
        # decode  (skip connections via concat)
        u1 = self.up1(x4)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.dec1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.dec2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.dec3(u3)

        return self.head(u3)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------
def build_model(cfg) -> ResUNet:
    """Create a ResUNet from a `softrank.config.Config` instance."""
    return ResUNet(in_channels=cfg.channels, out_channels=1,
                   filters=cfg.filters)
