"""Tests for UNet2D."""

import pytest
import torch

from totton_audio_de_mirroring.models.unet import UNet2D


def test_unet_output_shape_and_range() -> None:
    """Test that UNet output matches shape and stays within [0, 1]."""
    model = UNet2D(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        num_downsamples=2,
    )
    inputs = torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == inputs.shape
    assert torch.min(outputs) >= 0.0
    assert torch.max(outputs) <= 1.0


def test_unet_invalid_input_dim() -> None:
    """Test that invalid input dimensions raise ValueError."""
    model = UNet2D()
    bad_input = torch.rand(1, 32, 32)

    with pytest.raises(ValueError, match="features must be 4D"):
        _ = model(bad_input)
