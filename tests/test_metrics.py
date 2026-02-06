"""Tests for Stage 1 hard metrics."""

from __future__ import annotations

import numpy as np
import pytest

from totton_audio_de_mirroring.evaluation.metrics import (
    evaluate_dataset,
    evaluate_stage1_hard_metrics,
)

SAMPLE_RATE = 88_200
DURATION_SEC = 0.25


def _time_axis(sample_rate: int, duration_sec: float) -> np.ndarray:
    num_samples = int(round(sample_rate * duration_sec))
    return np.arange(num_samples, dtype=np.float64) / float(sample_rate)


def _sine(freq_hz: float, sample_rate: int, duration_sec: float) -> np.ndarray:
    time = _time_axis(sample_rate, duration_sec)
    return np.sin(2.0 * np.pi * freq_hz * time)


def test_evaluate_stage1_hard_metrics_preserves_identity_signal() -> None:
    """Identical input/output should produce near-zero low-band errors."""
    signal = _sine(1_000.0, SAMPLE_RATE, DURATION_SEC) + 0.2 * _sine(
        30_000.0, SAMPLE_RATE, DURATION_SEC
    )

    metrics = evaluate_stage1_hard_metrics(
        input_signal=signal,
        output_signal=signal.copy(),
        sample_rate=SAMPLE_RATE,
        energy_cap=1.0,
    )

    assert metrics.lb_amplitude_error_db < -120.0
    assert metrics.lb_phase_error_deg < 1.0e-4
    assert metrics.lb_group_delay_error_samples < 1.0e-3
    assert metrics.touch_metric < 1.0e-6
    assert not metrics.hb_energy_cap_violated


def test_evaluate_stage1_hard_metrics_detects_mirror_reduction() -> None:
    """Suppressing mirror pair should increase mirror reduction ratio."""
    low_freq = 21_000.0
    high_freq = 23_100.0
    keep_freq = 35_000.0
    input_signal = (
        _sine(low_freq, SAMPLE_RATE, DURATION_SEC)
        + _sine(high_freq, SAMPLE_RATE, DURATION_SEC)
        + 0.4 * _sine(keep_freq, SAMPLE_RATE, DURATION_SEC)
    )
    output_signal = (
        0.2 * _sine(low_freq, SAMPLE_RATE, DURATION_SEC)
        + 0.2 * _sine(high_freq, SAMPLE_RATE, DURATION_SEC)
        + 0.4 * _sine(keep_freq, SAMPLE_RATE, DURATION_SEC)
    )

    metrics = evaluate_stage1_hard_metrics(
        input_signal=input_signal,
        output_signal=output_signal,
        sample_rate=SAMPLE_RATE,
        energy_cap=1.0,
        n_fft=1024,
        hop_length=256,
    )

    assert metrics.mirror_reduction_ratio > 0.5
    assert metrics.touch_metric < 0.5


def test_evaluate_stage1_hard_metrics_detects_energy_cap_violation() -> None:
    """Strong high-band output should violate strict energy cap."""
    input_signal = 0.2 * _sine(30_000.0, SAMPLE_RATE, DURATION_SEC)
    output_signal = 1.2 * _sine(30_000.0, SAMPLE_RATE, DURATION_SEC)

    metrics = evaluate_stage1_hard_metrics(
        input_signal=input_signal,
        output_signal=output_signal,
        sample_rate=SAMPLE_RATE,
        energy_cap=1.0e-4,
    )

    assert metrics.hb_energy_cap_violated


def test_evaluate_dataset_aggregates_violation_rate() -> None:
    """Dataset aggregator should report cap violation rate correctly."""
    safe_input = 0.2 * _sine(30_000.0, SAMPLE_RATE, DURATION_SEC)
    safe_output = 0.2 * _sine(30_000.0, SAMPLE_RATE, DURATION_SEC)
    bad_input = 0.2 * _sine(30_000.0, SAMPLE_RATE, DURATION_SEC)
    bad_output = 1.2 * _sine(30_000.0, SAMPLE_RATE, DURATION_SEC)

    result = evaluate_dataset(
        samples=[
            ("safe", safe_input, safe_output),
            ("bad", bad_input, bad_output),
        ],
        sample_rate=SAMPLE_RATE,
        energy_cap=5.0e-2,
    )

    assert len(result.samples) == 2
    assert result.hb_energy_cap_violation_rate == pytest.approx(0.5)


def test_evaluate_stage1_hard_metrics_rejects_invalid_shape() -> None:
    """Invalid signal rank should raise ValueError."""
    signal = np.zeros((2, 128), dtype=np.float64)

    with pytest.raises(ValueError, match="must be 1D arrays"):
        _ = evaluate_stage1_hard_metrics(
            input_signal=signal,
            output_signal=signal,
            sample_rate=SAMPLE_RATE,
        )
