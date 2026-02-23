"""Tests for Stage 1 safety constraints."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from totton_audio_de_mirroring.data.filters import design_band_split_filters
from totton_audio_de_mirroring.models.safety_constraints import (
    apply_energy_cap,
    apply_envelope_target,
    apply_highband_mask,
    apply_safety_constraints,
    build_envelope_target,
    build_highband_mask,
    enforce_highpass_dc_block,
)


def test_apply_envelope_target_preserves_lowband_bins() -> None:
    """Ensure envelope shaping never modifies low-band bins."""
    magnitude = torch.ones(1, 8, 4)
    envelope = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2])
    highband_mask = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    shaped = apply_envelope_target(magnitude, envelope, highband_mask)

    assert torch.allclose(shaped[:, :4, :], magnitude[:, :4, :])
    expected_high = envelope[4:, None].repeat(1, magnitude.shape[-1])
    assert torch.allclose(shaped[0, 4:, :], expected_high)


def test_apply_highband_mask_zeros_lowband_bins() -> None:
    """Ensure explicit high-band mask removes low-band leakage."""
    magnitude = torch.ones(1, 6, 2)
    highband_mask = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    masked = apply_highband_mask(magnitude, highband_mask)

    assert torch.count_nonzero(masked[:, :2, :]) == 0
    assert torch.allclose(masked[:, 2:, :], torch.ones(1, 4, 2))


def test_apply_safety_constraints_caps_highband_energy() -> None:
    """Ensure composed safety constraints always enforce energy cap."""
    magnitude = torch.ones(2, 16, 10) * 20.0
    envelope = torch.ones(16)
    highband_mask = torch.cat([torch.zeros(8), torch.ones(8)])
    energy_cap = 100.0

    constrained = apply_safety_constraints(
        magnitude,
        envelope_target=envelope,
        highband_mask=highband_mask,
        energy_cap=energy_cap,
    )

    energy = torch.mean(constrained[:, highband_mask.bool(), :] ** 2, dim=(-2, -1))
    assert torch.all(energy <= energy_cap + 1.0e-6)
    assert torch.count_nonzero(constrained[:, :8, :]) == 0


def test_build_envelope_target_monotonic_in_highband() -> None:
    """Ensure high-band envelope monotonically decays above cutoff."""
    sample_rate = 88_200
    envelope = build_envelope_target(
        num_freqs=129,
        sample_rate=sample_rate,
        cutoff_hz=20_000.0,
        floor=0.2,
    )
    highband_mask = build_highband_mask(
        num_freqs=129,
        sample_rate=sample_rate,
        cutoff_hz=20_000.0,
    )

    highband_values = envelope[highband_mask.bool()]
    diffs = torch.diff(highband_values)
    assert torch.all(diffs <= 1.0e-8)
    assert torch.min(highband_values) >= 0.2 - 1.0e-6


def test_enforce_highpass_dc_block_suppresses_dc_below_minus_80db() -> None:
    """Ensure post-HPF check suppresses DC component in HB output."""
    sample_rate = 88_200
    _, highpass_taps = design_band_split_filters(
        cutoff_hz=20_000.0,
        sample_rate=sample_rate,
        num_taps=1025,
    )
    taps_tensor = torch.tensor(highpass_taps, dtype=torch.float32)

    num_samples = 8192
    time = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    signal = (0.5 + 0.2 * torch.sin(2 * torch.pi * 30_000 * time)).unsqueeze(0)
    filtered = enforce_highpass_dc_block(signal, taps_tensor)

    delay = (taps_tensor.numel() - 1) // 2
    core = filtered[:, delay:-delay]
    dc = float(torch.mean(core).abs())
    dc_db = 20.0 * np.log10(dc + 1.0e-12)

    assert dc_db < -80.0


def test_apply_energy_cap_rejects_invalid_rank() -> None:
    """Ensure input validation rejects non-3D magnitude tensors."""
    magnitude = torch.ones(8, 4)
    with pytest.raises(ValueError, match="magnitude must be 3D"):
        _ = apply_energy_cap(magnitude, energy_cap=1.0)


def test_apply_energy_cap_rejects_nan_energy_cap() -> None:
    """Ensure NaN energy cap is rejected explicitly."""
    magnitude = torch.ones(1, 8, 4)
    with pytest.raises(ValueError, match="finite positive value"):
        _ = apply_energy_cap(magnitude, energy_cap=float("nan"))
