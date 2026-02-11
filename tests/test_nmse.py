"""Tests for NMSE high-band processing."""

import pytest
import torch

from totton_audio_de_mirroring.data.filters import design_band_split_filters
from totton_audio_de_mirroring.models.nmse import (
    NMSE,
    STFTConfig,
    _apply_fir_filter,
)
from totton_audio_de_mirroring.models.safety_constraints import apply_energy_cap
from totton_audio_de_mirroring.models.unet import UNet2D


def test_nmse_preserves_low_band() -> None:
    """Test that NMSE preserves low-band content by structure."""
    sample_rate = 88_200
    duration = 0.05
    num_samples = int(sample_rate * duration)

    time = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    low_tone = torch.sin(2 * torch.pi * 1_000 * time)
    high_tone = 0.2 * torch.sin(2 * torch.pi * 30_000 * time)
    signal = (low_tone + high_tone).unsqueeze(0)

    lowpass, highpass = design_band_split_filters(
        cutoff_hz=20_000.0,
        sample_rate=sample_rate,
        num_taps=513,
    )
    nmse = NMSE(
        sample_rate=sample_rate,
        cutoff_hz=20_000.0,
        stft_config=STFTConfig(n_fft=256, hop_length=64, win_length=256),
        energy_cap=1.0e9,
        lowpass_taps=lowpass,
        highpass_taps=highpass,
    )

    with torch.no_grad():
        output = nmse(signal)

    lowpass_taps = nmse.lowpass_taps
    highpass_taps = nmse.highpass_taps
    assert lowpass_taps is not None
    assert highpass_taps is not None
    lb_in = _apply_fir_filter(signal, lowpass_taps)
    hb_in = _apply_fir_filter(signal, highpass_taps)
    hb_out = nmse.forward_highband(hb_in)
    lb_out = output - hb_out

    max_diff = torch.max(torch.abs(lb_in - lb_out)).item()
    assert max_diff < 1.0e-4


def test_apply_energy_cap_limits_energy() -> None:
    """Test that energy cap enforces maximum energy per batch."""
    magnitude = torch.ones(2, 16, 10) * 10.0
    energy_cap = 100.0

    capped = apply_energy_cap(magnitude, energy_cap)
    energy = torch.mean(capped**2, dim=(-2, -1))

    assert torch.all(energy <= energy_cap + 1.0e-6)


def test_nmse_invalid_energy_cap_raises() -> None:
    """Test that invalid energy cap raises ValueError."""
    lowpass, highpass = design_band_split_filters(
        cutoff_hz=20_000.0,
        sample_rate=88_200,
        num_taps=129,
    )

    with pytest.raises(ValueError, match="finite positive value"):
        _ = NMSE(
            sample_rate=88_200,
            cutoff_hz=20_000.0,
            stft_config=STFTConfig(n_fft=256, hop_length=64, win_length=256),
            energy_cap=0.0,
            lowpass_taps=lowpass,
            highpass_taps=highpass,
        )

    with pytest.raises(ValueError, match="finite positive value"):
        _ = NMSE(
            sample_rate=88_200,
            cutoff_hz=20_000.0,
            stft_config=STFTConfig(n_fft=256, hop_length=64, win_length=256),
            energy_cap=float("nan"),
            lowpass_taps=lowpass,
            highpass_taps=highpass,
        )


def test_nmse_invalid_stft_config_raises() -> None:
    """Test that invalid STFT config raises ValueError."""
    lowpass, highpass = design_band_split_filters(
        cutoff_hz=20_000.0,
        sample_rate=88_200,
        num_taps=129,
    )

    with pytest.raises(ValueError, match="n_fft must be positive"):
        _ = NMSE(
            sample_rate=88_200,
            cutoff_hz=20_000.0,
            stft_config=STFTConfig(n_fft=0, hop_length=64, win_length=256),
            energy_cap=1.0,
            lowpass_taps=lowpass,
            highpass_taps=highpass,
        )


def test_nmse_rejects_multi_channel_mask() -> None:
    """Test that NMSE rejects masks with multiple channels."""
    lowpass, highpass = design_band_split_filters(
        cutoff_hz=20_000.0,
        sample_rate=88_200,
        num_taps=129,
    )
    nmse = NMSE(
        sample_rate=88_200,
        cutoff_hz=20_000.0,
        stft_config=STFTConfig(n_fft=256, hop_length=64, win_length=256),
        energy_cap=1.0,
        lowpass_taps=lowpass,
        highpass_taps=highpass,
        unet=UNet2D(out_channels=2, base_channels=8, num_downsamples=1),
    )

    high_band = torch.randn(1, 2048)
    with pytest.raises(ValueError, match="Mask output must have a single channel"):
        _ = nmse.forward_highband(high_band)
