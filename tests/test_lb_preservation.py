"""Tests for low-band preservation metrics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sp_signal

from totton_audio_de_mirroring.evaluation.lb_preservation import (
    evaluate_lowband_preservation,
)

SAMPLE_RATE = 88_200
DURATION_SEC = 0.25


def _time_axis(sample_rate: int, duration_sec: float) -> np.ndarray:
    num_samples = int(round(sample_rate * duration_sec))
    return np.arange(num_samples, dtype=np.float64) / float(sample_rate)


def _square_wave(freq_hz: float, sample_rate: int, duration_sec: float) -> np.ndarray:
    time = _time_axis(sample_rate, duration_sec)
    return np.sign(np.sin(2.0 * np.pi * freq_hz * time))


def test_lowband_metrics_identity_on_1khz_sine() -> None:
    """1kHz sine should pass strict low-band identity metrics."""
    time = _time_axis(SAMPLE_RATE, DURATION_SEC)
    signal = np.sin(2.0 * np.pi * 1_000.0 * time)

    metrics = evaluate_lowband_preservation(
        input_signal=signal,
        output_signal=signal.copy(),
        sample_rate=SAMPLE_RATE,
    )

    assert metrics.waveform_error_db < -100.0
    assert metrics.waveform_mse < 1.0e-10
    assert metrics.phase_error_deg < 0.1
    assert metrics.group_delay_error_ms < 0.5


def test_lowband_metrics_identity_on_chirp_signal() -> None:
    """Wideband low-frequency chirp should preserve phase and delay."""
    time = _time_axis(SAMPLE_RATE, DURATION_SEC)
    signal = sp_signal.chirp(
        time,
        f0=20.0,
        t1=float(time[-1]),
        f1=20_000.0,
        method="logarithmic",
    )

    metrics = evaluate_lowband_preservation(
        input_signal=signal,
        output_signal=signal.copy(),
        sample_rate=SAMPLE_RATE,
    )

    assert metrics.waveform_error_db < -100.0
    assert metrics.phase_error_deg < 0.1
    assert metrics.group_delay_error_ms < 0.5


def test_lowband_metrics_identity_on_impulse() -> None:
    """Impulse response should remain unchanged under identity mapping."""
    signal = np.zeros(int(round(SAMPLE_RATE * DURATION_SEC)), dtype=np.float64)
    signal[0] = 1.0

    metrics = evaluate_lowband_preservation(
        input_signal=signal,
        output_signal=signal.copy(),
        sample_rate=SAMPLE_RATE,
    )

    assert metrics.waveform_error_db < -100.0
    assert metrics.phase_error_deg < 0.1
    assert metrics.group_delay_error_ms < 0.5


def test_lowband_metrics_identity_on_square_wave() -> None:
    """Square-wave low-band components should preserve low-band identity."""
    signal = _square_wave(
        freq_hz=500.0,
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION_SEC,
    )

    metrics = evaluate_lowband_preservation(
        input_signal=signal,
        output_signal=signal.copy(),
        sample_rate=SAMPLE_RATE,
    )

    assert metrics.waveform_error_db < -100.0
    assert metrics.phase_error_deg < 0.1
    assert metrics.group_delay_error_ms < 0.5


def test_lowband_metrics_detects_degradation() -> None:
    """Amplitude attenuation in low-band should be reflected in metrics."""
    time = _time_axis(SAMPLE_RATE, DURATION_SEC)
    input_signal = np.sin(2.0 * np.pi * 1_000.0 * time)
    output_signal = 0.7 * input_signal

    metrics = evaluate_lowband_preservation(
        input_signal=input_signal,
        output_signal=output_signal,
        sample_rate=SAMPLE_RATE,
    )

    assert metrics.waveform_error_db > -20.0
    assert metrics.waveform_mse > 1.0e-3


def test_lowband_metrics_rejects_invalid_signal_shape() -> None:
    """2D signals are invalid for low-band preservation API."""
    signal = np.zeros((2, 16), dtype=np.float64)

    with pytest.raises(ValueError, match="must be 1D arrays"):
        _ = evaluate_lowband_preservation(
            input_signal=signal,
            output_signal=signal,
            sample_rate=SAMPLE_RATE,
        )
