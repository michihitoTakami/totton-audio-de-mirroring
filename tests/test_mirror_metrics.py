"""Tests for mirror/aliasing reduction metrics and visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from totton_audio_de_mirroring.evaluation.mirror_metrics import (
    evaluate_mirror_reduction,
    export_mirror_reduction_visualization,
)

SAMPLE_RATE = 88_200
DURATION_SEC = 0.5


def _time_axis(sample_rate: int, duration_sec: float) -> np.ndarray:
    num_samples = int(round(sample_rate * duration_sec))
    return np.arange(num_samples, dtype=np.float64) / float(sample_rate)


def _sine(freq_hz: float, sample_rate: int, duration_sec: float) -> np.ndarray:
    time = _time_axis(sample_rate, duration_sec)
    return np.sin(2.0 * np.pi * freq_hz * time)


def _mirror_pair_signal(sample_rate: int, duration_sec: float) -> np.ndarray:
    mirror_center = sample_rate / 4.0
    low_freq = 21_000.0
    high_freq = 2.0 * mirror_center - low_freq
    modulation = 1.0 + 0.4 * _sine(95.0, sample_rate, duration_sec)
    mirror_pair = modulation * (
        _sine(low_freq, sample_rate, duration_sec)
        + _sine(high_freq, sample_rate, duration_sec)
    )
    keep_component = 0.35 * _sine(30_000.0, sample_rate, duration_sec)
    return mirror_pair + keep_component


def test_evaluate_mirror_reduction_exceeds_issue_target() -> None:
    """Strong suppression should exceed 70% mirror-symmetry reduction."""
    before_signal = _mirror_pair_signal(SAMPLE_RATE, DURATION_SEC)
    mirror_only = _mirror_pair_signal(SAMPLE_RATE, DURATION_SEC) - 0.35 * _sine(
        30_000.0,
        SAMPLE_RATE,
        DURATION_SEC,
    )
    keep_component = 0.35 * _sine(30_000.0, SAMPLE_RATE, DURATION_SEC)
    after_signal = 0.1 * mirror_only + keep_component

    metrics = evaluate_mirror_reduction(
        before_signal=before_signal,
        after_signal=after_signal,
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
    )

    assert metrics.symmetry_reduction_ratio > 0.70
    assert metrics.mirror_band_energy_reduction_ratio > 0.70


def test_export_mirror_reduction_visualization_writes_png(tmp_path: Path) -> None:
    """Visualization export should write non-empty PNG file."""
    before_signal = _mirror_pair_signal(SAMPLE_RATE, DURATION_SEC)
    after_signal = 0.2 * before_signal
    output_path = tmp_path / "mirror" / "sample_before_after.png"

    artifacts = export_mirror_reduction_visualization(
        before_signal=before_signal,
        after_signal=after_signal,
        sample_rate=SAMPLE_RATE,
        output_path=output_path,
        n_fft=1024,
        hop_length=256,
    )

    assert artifacts.plot_path.exists()
    assert artifacts.plot_path.stat().st_size > 0


def test_evaluate_mirror_reduction_rejects_invalid_input_shape() -> None:
    """Invalid signal rank should raise ValueError."""
    signal_2d = np.zeros((2, 256), dtype=np.float64)

    with pytest.raises(ValueError, match="must be 1D arrays"):
        _ = evaluate_mirror_reduction(
            before_signal=signal_2d,
            after_signal=signal_2d,
            sample_rate=SAMPLE_RATE,
        )
