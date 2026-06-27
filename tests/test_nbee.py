"""Tests for the Neural Bandwidth Extension Engine (Stage 1b)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from totton_audio_de_mirroring.data.filters import design_band_split_filters
from totton_audio_de_mirroring.models.nbee import NBEEConfig, NeuralBandwidthExtension

SR = 88_200
CUTOFF = 20_000.0
ENERGY_CAP = 1.0e-2


def _build_model() -> NeuralBandwidthExtension:
    lowpass, highpass = design_band_split_filters(
        cutoff_hz=CUTOFF, sample_rate=SR, num_taps=1025, window="hamming"
    )
    return NeuralBandwidthExtension(
        sample_rate=SR,
        cutoff_hz=CUTOFF,
        energy_cap=ENERGY_CAP,
        envelope_floor=0.2,
        lowpass_taps=lowpass,
        highpass_taps=highpass,
        model_config=NBEEConfig(base_channels=8, num_downsamples=2),
    )


def _signal(batch: int = 2, length: int = 8192) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    t = torch.arange(length) / SR
    tone = 0.3 * torch.sin(2 * torch.pi * 5000 * t)
    noise = 0.01 * torch.randn(batch, length, generator=generator)
    return tone.unsqueeze(0) + noise


def test_nbee_forward_preserves_shape() -> None:
    model = _build_model().eval()
    signal = _signal()
    with torch.no_grad():
        out = model.forward(signal)
    assert out.shape == signal.shape
    assert torch.all(torch.isfinite(out))


def test_nbee_preserves_low_band() -> None:
    """0-20kHz is a structural bypass: output = LPF(input) + band-limited HB.

    The generated high band must carry negligible low-band energy, so the only
    low-band content in the output is the bypassed ``LPF(input)`` term.
    """
    model = _build_model().eval()
    signal = _signal()
    from totton_audio_de_mirroring.models.nmse import _apply_fir_filter

    with torch.no_grad():
        hb = model.generate_highband(signal)
    lb_of_hb = _apply_fir_filter(hb, model.lowpass_taps)
    lb_of_signal = _apply_fir_filter(signal, model.lowpass_taps)
    leak_ratio = float(lb_of_hb.abs().mean() / (lb_of_signal.abs().mean() + 1e-9))
    assert leak_ratio < 0.05


def test_nbee_respects_energy_cap() -> None:
    """Generated high band must satisfy the energy cap (IMD safety)."""
    model = _build_model().eval()
    signal = _signal()
    with torch.no_grad():
        hb = model.generate_highband(signal)
    window = torch.hann_window(1024)
    stft = torch.stft(
        hb,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=window,
        center=True,
        return_complex=True,
        normalized=True,
    )
    freqs = torch.fft.rfftfreq(1024, d=1.0 / SR)
    hb_bins = freqs >= CUTOFF
    energy = float((stft.abs()[:, hb_bins, :] ** 2).mean().item())
    assert energy <= ENERGY_CAP * 1.05


def test_nbee_generates_highband_from_low_band_only_input() -> None:
    """Generator emits HB energy even when the input HB is ~zero.

    A [0,1] suppressor multiplies the input magnitude, so a low-band-only input
    would force a zero output. NBEE predicts an absolute magnitude: with the
    head forced to a positive constant, the safety-shaped high band is nonzero
    despite the input carrying no high-band content.
    """
    model = _build_model().eval()
    # Force the linear output head to emit a positive constant magnitude.
    with torch.no_grad():
        model.unet.output_conv.weight.zero_()
        model.unet.output_conv.bias.fill_(1.0)
    t = torch.arange(8192) / SR
    lb_only = (0.5 * torch.sin(2 * torch.pi * 3000 * t)).unsqueeze(0)
    with torch.no_grad():
        hb = model.generate_highband(lb_only)
    assert torch.all(torch.isfinite(hb))
    assert float(hb.abs().max().item()) > 0.0


def test_nbee_absolute_mode_forward() -> None:
    """The legacy absolute-magnitude generation mode still runs end to end."""
    lowpass, highpass = design_band_split_filters(
        cutoff_hz=CUTOFF, sample_rate=SR, num_taps=1025, window="hamming"
    )
    model = NeuralBandwidthExtension(
        sample_rate=SR,
        cutoff_hz=CUTOFF,
        energy_cap=ENERGY_CAP,
        envelope_floor=0.2,
        lowpass_taps=lowpass,
        highpass_taps=highpass,
        model_config=NBEEConfig(
            base_channels=8, num_downsamples=2, generation_mode="absolute"
        ),
    ).eval()
    with torch.no_grad():
        out = model.forward(_signal())
    assert out.shape == _signal().shape
    assert torch.all(torch.isfinite(out))


def test_nbee_rejects_invalid_signal() -> None:
    model = _build_model()
    with pytest.raises(ValueError):
        model.forward(torch.zeros(0))


def test_nbee_config_roundtrip() -> None:
    cfg = NBEEConfig(base_channels=16, num_downsamples=3)
    restored = NBEEConfig.from_mapping(cfg.to_checkpoint_dict())
    assert restored == cfg
    assert cfg.to_checkpoint_dict()["model_type"] == "nbee"


def test_nbee_state_dict_roundtrip() -> None:
    model = _build_model()
    state = model.state_dict()
    clone = _build_model()
    clone.load_state_dict(state)
    signal = _signal()
    with torch.no_grad():
        assert torch.allclose(model.forward(signal), clone.forward(signal))


def test_design_filters_are_odd_length() -> None:
    lowpass, highpass = design_band_split_filters(
        cutoff_hz=CUTOFF, sample_rate=SR, num_taps=1025, window="hamming"
    )
    assert lowpass.shape[0] % 2 == 1
    assert np.all(np.isfinite(lowpass))
    assert np.all(np.isfinite(highpass))
