"""Road segmentation networks: a p4-equivariant U-Net and a standard U-Net baseline.

Both networks output one logit per pixel (road / not road). The equivariant network's
output rotates exactly with its input for 90-degree rotations, by construction.
"""
from __future__ import annotations

import torch
from torch import nn

from .p4conv import (
    P4BatchNorm,
    P4GroupConv,
    P4LiftingConv,
    group_pool,
    p4_max_pool2d,
    p4_upsample2x,
)


# --------------------------------------------------------------------------- equivariant
class P4Block(nn.Module):
    def __init__(self, cin: int, cout: int, lifting: bool = False):
        super().__init__()
        first = P4LiftingConv(cin, cout) if lifting else P4GroupConv(cin, cout)
        self.layers = nn.ModuleList([first, P4BatchNorm(cout), P4GroupConv(cout, cout), P4BatchNorm(cout)])
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.layers[1](self.layers[0](x)))
        return self.act(self.layers[3](self.layers[2](x)))


class P4UNet(nn.Module):
    """U-Net whose every layer is p4-equivariant; group pooling at the end gives a planar map.

    `width` is the number of channels per orientation, so the effective feature width is 4*width
    while the parameter count stays close to a standard U-Net of the same `width`*2.
    """

    def __init__(self, in_channels: int = 4, width: int = 8, depth: int = 3, out_channels: int = 1):
        super().__init__()
        chs = [width * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([P4Block(in_channels, chs[0], lifting=True)])
        self.down.extend(P4Block(chs[i], chs[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(P4Block(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chs[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else p4_max_pool2d(x))
            skips.append(x)
        skips.pop()
        for block in self.up:
            x = block(torch.cat([p4_upsample2x(x), skips.pop()], dim=1))
        # Pooling over orientations makes features invariant per pixel -> output rotates with input.
        return self.head(group_pool(x))


# --------------------------------------------------------------------------- baseline
class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Standard U-Net baseline (no built-in symmetry)."""

    def __init__(self, in_channels: int = 4, width: int = 16, depth: int = 3, out_channels: int = 1):
        super().__init__()
        chs = [width * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([ConvBlock(in_channels, chs[0])])
        self.down.extend(ConvBlock(chs[i], chs[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(ConvBlock(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth)))
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.head = nn.Conv2d(chs[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        skips.pop()
        for block in self.up:
            x = block(torch.cat([self.upsample(x), skips.pop()], dim=1))
        return self.head(x)


def build_model(name: str, in_channels: int = 4, out_channels: int = 1) -> nn.Module:
    """out_channels=1: binary road segmentation; out_channels=3: background / road / built-up."""
    if name == "p4unet":
        return P4UNet(in_channels, width=8, out_channels=out_channels)
    if name == "unet":
        return UNet(in_channels, width=16, out_channels=out_channels)
    raise ValueError(f"unknown model {name!r} (expected 'p4unet' or 'unet')")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
