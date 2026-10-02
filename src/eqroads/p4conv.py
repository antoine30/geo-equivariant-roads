"""Group-equivariant convolutions for the p4 group (translations + rotations by 90 degrees).

Overhead imagery has no canonical orientation: a road is a road whatever its heading.
These layers build that symmetry into the network instead of hoping data augmentation
teaches it. Rotations by multiples of 90 degrees map the pixel grid exactly onto itself,
so equivariance here is exact (no interpolation), and it is verified by unit tests.

Feature maps carry an extra "orientation" axis of size 4:
    lifting layer : (B, C_in, H, W)     -> (B, C_out, 4, H, W)
    group layer   : (B, C_in, 4, H, W)  -> (B, C_out, 4, H, W)
    group pooling : (B, C, 4, H, W)     -> (B, C, H, W)   (rotation-invariant per pixel)

Equivariance property (tested in tests/test_equivariance.py):
    f(rot90(x)) == roll(rot90(f(x)), shift=1, dim=orientation)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

N_ROT = 4  # order of the rotation group C4


def rotate_spatial(x: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate the two last (spatial) dimensions by k * 90 degrees counter-clockwise."""
    return torch.rot90(x, k, dims=(-2, -1))


class P4LiftingConv(nn.Module):
    """Lifts a planar image to a function on p4: one response per filter orientation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1, "odd kernel size keeps the rotation centre on a pixel"
        self.out_channels = out_channels
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stack the 4 rotated copies of the filter bank: (4*C_out, C_in, k, k)
        w = torch.cat([rotate_spatial(self.weight, r) for r in range(N_ROT)], dim=0)
        b = self.bias.repeat(N_ROT) if self.bias is not None else None
        y = F.conv2d(x, w, b, padding=self.padding)
        B, _, H, W = y.shape
        # (B, 4, C_out, H, W) -> (B, C_out, 4, H, W)
        return y.view(B, N_ROT, self.out_channels, H, W).transpose(1, 2).contiguous()


class P4GroupConv(nn.Module):
    """Convolution on p4: filters are rotated spatially AND cyclically shifted along orientation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, N_ROT, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _filter_bank(self) -> torch.Tensor:
        banks = []
        for r in range(N_ROT):
            # Rotating a function on p4 by r: rotate the plane and shift the orientation axis by r.
            w = torch.roll(self.weight, shifts=r, dims=2)
            w = rotate_spatial(w, r)
            banks.append(w.reshape(self.out_channels, self.in_channels * N_ROT, *w.shape[-2:]))
        return torch.cat(banks, dim=0)  # (4*C_out, C_in*4, k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, G, H, W = x.shape
        assert G == N_ROT and C == self.in_channels
        b = self.bias.repeat(N_ROT) if self.bias is not None else None
        y = F.conv2d(x.reshape(B, C * G, H, W), self._filter_bank(), b, padding=self.padding)
        return y.view(B, N_ROT, self.out_channels, H, W).transpose(1, 2).contiguous()


class P4BatchNorm(nn.Module):
    """Batch norm sharing statistics across orientations (required to keep equivariance)."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, 4, H, W) is a valid 3D-BN input
        return self.bn(x)


def p4_max_pool2d(x: torch.Tensor) -> torch.Tensor:
    """2x2 spatial max pooling, applied per orientation. Equivariant for even H, W."""
    B, C, G, H, W = x.shape
    y = F.max_pool2d(x.view(B, C * G, H, W), 2)
    return y.view(B, C, G, H // 2, W // 2)


def p4_upsample2x(x: torch.Tensor) -> torch.Tensor:
    B, C, G, H, W = x.shape
    y = F.interpolate(x.view(B, C * G, H, W), scale_factor=2, mode="nearest")
    return y.view(B, C, G, 2 * H, 2 * W)


def group_pool(x: torch.Tensor, mode: str = "max") -> torch.Tensor:
    """Pool over the orientation axis -> rotation-invariant features per pixel."""
    return x.amax(dim=2) if mode == "max" else x.mean(dim=2)
