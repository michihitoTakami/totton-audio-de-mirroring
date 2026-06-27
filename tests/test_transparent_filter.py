"""Tests for the 32-bit-transparent Kaiser FIR 2x upsampler."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.filters import (
    design_transparent_upsampler_fir,
    kaiser_params_for_stopband,
    upsample_fir,
)

SRC = 44_100
RATIO = 2
TARGET = SRC * RATIO


def test_kaiser_params_match_design_formula() -> None:
    num_taps, beta = kaiser_params_for_stopband(180.0, 2050.0, TARGET)
    # Kaiser beta for 180 dB ~ 0.1102*(180-8.7) = 18.88
    assert beta == pytest.approx(18.88, abs=0.1)
    assert num_taps % 2 == 1
    assert 400 < num_taps < 800  # derived minimal taps for this transition


def test_transparent_fir_passband_and_stopband() -> None:
    taps, _ = design_transparent_upsampler_fir(source_sr=SRC, ratio=RATIO)
    w, h = sp_signal.freqz(taps / RATIO, worN=16384, fs=TARGET)
    mag_db = 20.0 * np.log10(np.abs(h) + 1e-30)
    passband_ripple = float(np.max(np.abs(mag_db[w <= 20_000.0])))
    stopband = float(np.max(mag_db[w >= 22_050.0]))
    assert passband_ripple < 1.0e-6  # far below 32-bit float floor
    assert stopband < -144.0  # below 32-bit float floor


def test_transparent_fir_roundtrip_hits_float32_floor() -> None:
    taps, _ = design_transparent_upsampler_fir(source_sr=SRC, ratio=RATIO)
    n = 8192
    t_src = np.arange(n) / SRC
    src = 0.4 * np.sin(2 * np.pi * 10_000.0 * t_src)
    up = upsample_fir(src.astype(np.float32).astype(np.float64), RATIO, taps)
    t_tgt = np.arange(n * RATIO) / TARGET
    ideal = 0.4 * np.sin(2 * np.pi * 10_000.0 * t_tgt)
    guard = taps.size
    err = float(np.max(np.abs(up[guard:-guard] - ideal[guard:-guard])))
    assert err < 1.0e-6  # near float32 epsilon, i.e. transparent


def test_transparent_fir_rejects_images_far_better_than_bessel() -> None:
    from totton_audio_de_mirroring.data.degradation import upsample_bessel_reference

    n = 8192
    t = np.arange(n) / SRC
    tone = 0.5 * np.sin(2 * np.pi * 21_000.0 * t)
    taps, _ = design_transparent_upsampler_fir(source_sr=SRC, ratio=RATIO)
    fir_up = upsample_fir(tone, RATIO, taps)
    bessel_up = upsample_bessel_reference(
        signal=tone, source_sr=SRC, target_sr=TARGET, cutoff_hz=20_000.0, order=6
    )

    def image_db(sig: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(sig * np.hanning(sig.shape[0])))
        freqs = np.fft.rfftfreq(sig.shape[0], d=1.0 / TARGET)
        fund = float(np.max(spec[freqs <= 22_050.0]) + 1e-30)
        img = float(np.max(spec[freqs >= 22_050.0]) + 1e-30)
        return 20.0 * np.log10(img / fund)

    assert image_db(fir_up) < -120.0  # images essentially gone
    assert image_db(fir_up) < image_db(bessel_up) - 60.0  # vastly better than IIR


def test_upsample_fir_length_and_validation() -> None:
    taps, _ = design_transparent_upsampler_fir(source_sr=SRC, ratio=RATIO)
    out = upsample_fir(np.zeros(1000), RATIO, taps)
    assert out.shape[0] == 2000
    with pytest.raises(ValueError):
        upsample_fir(np.zeros((2, 10)), RATIO, taps)  # 2D not allowed


def test_design_rejects_even_taps() -> None:
    with pytest.raises(ValueError, match="odd"):
        design_transparent_upsampler_fir(source_sr=SRC, ratio=RATIO, num_taps=512)
