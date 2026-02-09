"""Tests for lightweight NMSE model."""

import pytest
import torch

from totton_audio_de_mirroring.data.filters import design_band_split_filters
from totton_audio_de_mirroring.models.nmse_light import NMSELight, NMSELightConfig


def test_nmse_light_default_parameter_budget() -> None:
    """Default lightweight architecture should stay within 5-7M params."""
    lowpass, highpass = design_band_split_filters(
        cutoff_hz=20_000.0,
        sample_rate=88_200,
        num_taps=129,
    )
    model = NMSELight(
        sample_rate=88_200,
        cutoff_hz=20_000.0,
        energy_cap=1.0,
        envelope_floor=0.0,
        lowpass_taps=lowpass,
        highpass_taps=highpass,
    )
    params = sum(parameter.numel() for parameter in model.parameters())
    assert 5_000_000 <= params <= 7_000_000


def test_nmse_light_forward_shape() -> None:
    """Lightweight NMSE should preserve input shape."""
    lowpass, highpass = design_band_split_filters(
        cutoff_hz=20_000.0,
        sample_rate=88_200,
        num_taps=129,
    )
    model = NMSELight(
        sample_rate=88_200,
        cutoff_hz=20_000.0,
        energy_cap=1.0,
        envelope_floor=0.0,
        lowpass_taps=lowpass,
        highpass_taps=highpass,
    )
    signal = torch.randn(2, 4096)
    with torch.no_grad():
        output = model(signal)
    assert output.shape == signal.shape


def test_nmse_light_config_rejects_invalid_channels() -> None:
    """Config should reject non-positive channels."""
    with pytest.raises(ValueError, match="base_channels"):
        _ = NMSELightConfig(base_channels=0)
