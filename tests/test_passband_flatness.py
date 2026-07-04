"""Tests for passband flatness and gain-accuracy metrics."""

import numpy as np
import pytest
from scipy import signal as sp_signal

from totton_audio_de_mirroring.evaluation.passband_flatness import (
    compute_flatness,
    compute_lowband_gain_error_db,
)

SOURCE_SR = 44_100
TARGET_SR = 88_200


def _pink_noise(num_samples: int, sample_rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(num_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
    weights = np.ones_like(freqs)
    weights[1:] = 1.0 / np.sqrt(freqs[1:])
    weights[0] = 0.0
    pink = np.fft.irfft(spectrum * weights, n=num_samples)
    return 0.5 * pink / np.max(np.abs(pink))


@pytest.fixture(scope="module")
def source() -> np.ndarray:
    return _pink_noise(SOURCE_SR, SOURCE_SR, seed=7)


@pytest.fixture(scope="module")
def upsampled(source: np.ndarray) -> np.ndarray:
    return np.asarray(sp_signal.resample_poly(source, 2, 1), dtype=np.float64)


def test_ideal_upsampling_is_flat(source, upsampled) -> None:
    metrics = compute_flatness(source, SOURCE_SR, upsampled, TARGET_SR)
    assert metrics.max_dip_db >= -0.5
    assert metrics.max_boost_db <= 0.5


def test_notch_is_detected(source, upsampled) -> None:
    """A 13 kHz notch (the historical artifact) must register as a dip."""
    notch_b, notch_a = sp_signal.iirnotch(13_000.0, Q=8.0, fs=TARGET_SR)
    notched = sp_signal.lfilter(notch_b, notch_a, upsampled)
    metrics = compute_flatness(source, SOURCE_SR, np.asarray(notched), TARGET_SR)
    assert metrics.max_dip_db <= -2.0


def test_gain_error_tracks_scaling(source, upsampled) -> None:
    error_db = compute_lowband_gain_error_db(
        source, SOURCE_SR, upsampled * 0.5, TARGET_SR
    )
    assert error_db == pytest.approx(-6.02, abs=0.1)


def test_matched_levels_have_zero_error(source, upsampled) -> None:
    error_db = compute_lowband_gain_error_db(source, SOURCE_SR, upsampled, TARGET_SR)
    assert abs(error_db) <= 0.05


def test_invalid_inputs_raise(source) -> None:
    with pytest.raises(ValueError, match="1D"):
        compute_flatness(np.zeros((2, 2)), SOURCE_SR, source, SOURCE_SR)
    with pytest.raises(ValueError, match="positive"):
        compute_flatness(source, 0, source, SOURCE_SR)
