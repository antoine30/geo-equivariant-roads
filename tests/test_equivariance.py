"""Rotation equivariance is a mathematical property of the architecture: test it exactly."""
import pytest
import torch

from eqroads.p4conv import P4GroupConv, P4LiftingConv, rotate_spatial
from eqroads.segnet import P4UNet, UNet

torch.manual_seed(0)
ATOL = 1e-5


def rot_p4(y: torch.Tensor, k: int) -> torch.Tensor:
    """Action of a rotation by k*90 degrees on a feature map over p4: rotate space, shift orientation."""
    return torch.roll(rotate_spatial(y, k), shifts=k, dims=2)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_lifting_conv_equivariant(k):
    layer = P4LiftingConv(4, 6, kernel_size=5)
    x = torch.randn(2, 4, 16, 16)
    assert torch.allclose(layer(rotate_spatial(x, k)), rot_p4(layer(x), k), atol=ATOL)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_group_conv_equivariant(k):
    layer = P4GroupConv(3, 5)
    x = torch.randn(2, 3, 4, 16, 16)
    assert torch.allclose(layer(rot_p4(x, k)), rot_p4(layer(x), k), atol=ATOL)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_p4unet_output_rotates_with_input(k):
    model = P4UNet(in_channels=4, width=4).eval()
    x = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(rotate_spatial(x, k)), rotate_spatial(model(x), k), atol=1e-4)


def test_plain_unet_is_not_equivariant():
    """Sanity check: the property is not trivial, a standard U-Net does not have it."""
    model = UNet(in_channels=4, width=4).eval()
    x = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        assert not torch.allclose(model(rotate_spatial(x, 1)), rotate_spatial(model(x), 1), atol=1e-3)
