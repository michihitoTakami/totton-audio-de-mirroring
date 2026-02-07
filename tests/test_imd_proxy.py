"""Tests for IMD proxy evaluation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from totton_audio_de_mirroring.evaluation.imd_proxy import (
    apply_soft_clipping,
    evaluate_imd_path,
    evaluate_imd_proxy,
)

SAMPLE_RATE = 88_200
DURATION_SEC = 0.6


def _time_axis(sample_rate: int, duration_sec: float) -> np.ndarray:
    num_samples = int(round(sample_rate * duration_sec))
    return np.arange(num_samples, dtype=np.float64) / float(sample_rate)


def _sine(freq_hz: float, sample_rate: int, duration_sec: float) -> np.ndarray:
    time = _time_axis(sample_rate, duration_sec)
    return np.sin(2.0 * np.pi * freq_hz * time)


def test_apply_soft_clipping_is_bounded_and_immutable() -> None:
    """Soft clipping should not mutate input and should stay bounded."""
    signal = np.linspace(-2.0, 2.0, num=1024, dtype=np.float64)
    original = signal.copy()

    clipped = apply_soft_clipping(signal, drive=2.0)

    assert np.allclose(signal, original)
    assert clipped.shape == signal.shape
    assert float(np.max(clipped)) <= 1.0 + 1.0e-8
    assert float(np.min(clipped)) >= -1.0 - 1.0e-8


def test_evaluate_imd_proxy_shows_nmse_improvement() -> None:
    """NMSE-like high-band suppression should reduce IMD and THD+N."""
    audible = 0.2 * _sine(1_000.0, SAMPLE_RATE, DURATION_SEC)
    naive_hf = 0.9 * _sine(30_000.0, SAMPLE_RATE, DURATION_SEC) + 0.9 * _sine(
        31_000.0,
        SAMPLE_RATE,
        DURATION_SEC,
    )
    nmse_hf = 0.05 * _sine(30_000.0, SAMPLE_RATE, DURATION_SEC) + 0.05 * _sine(
        31_000.0,
        SAMPLE_RATE,
        DURATION_SEC,
    )

    naive_signal = audible + naive_hf
    nmse_signal = audible + nmse_hf

    metrics = evaluate_imd_proxy(
        naive_signal=naive_signal,
        nmse_signal=nmse_signal,
        sample_rate=SAMPLE_RATE,
        clip_drive=2.6,
        num_taps=1025,
    )

    assert metrics.nmse_has_lower_imd
    assert metrics.audible_distortion_reduction_db > 10.0
    assert metrics.thdn_improvement_db > 10.0
    assert metrics.thdn_improvement_over_10db


def test_evaluate_imd_path_returns_finite_metrics() -> None:
    """Single-path IMD metrics should remain finite for valid input."""
    signal = 0.3 * _sine(2_000.0, SAMPLE_RATE, DURATION_SEC) + 0.1 * _sine(
        28_000.0,
        SAMPLE_RATE,
        DURATION_SEC,
    )

    metrics = evaluate_imd_path(
        signal=signal,
        sample_rate=SAMPLE_RATE,
        clip_drive=1.5,
        num_taps=1025,
    )

    assert np.isfinite(metrics.audible_distortion_energy)
    assert np.isfinite(metrics.thdn_db)


def test_evaluate_imd_proxy_rejects_shape_mismatch() -> None:
    """Input/output shape mismatch should raise ValueError."""
    a = np.zeros(4096, dtype=np.float64)
    b = np.zeros(2048, dtype=np.float64)

    with pytest.raises(ValueError, match="identical shapes"):
        _ = evaluate_imd_proxy(
            naive_signal=a,
            nmse_signal=b,
            sample_rate=SAMPLE_RATE,
        )


def test_evaluate_imd_path_rejects_short_signal() -> None:
    """Signals shorter than filter warm-up should be rejected."""
    signal = np.zeros(512, dtype=np.float64)

    with pytest.raises(ValueError, match="greater than num_taps"):
        _ = evaluate_imd_path(
            signal=signal,
            sample_rate=SAMPLE_RATE,
            num_taps=1025,
        )
