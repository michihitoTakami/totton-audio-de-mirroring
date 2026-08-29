"""Tests for CAPB impulse-report metrics."""

from pathlib import Path

import numpy as np
import pytest
from scripts.report_capb_impulse import RateCase, _metrics, source_duration_ms


def test_impulse_metrics_use_gate_aligned_pre_window() -> None:
    sample_rate = 88_200
    center = sample_rate // 2
    signal = np.zeros(sample_rate, dtype=np.float64)
    guard = round(0.0005 * sample_rate)
    window = round(0.0035 * sample_rate)
    signal[center - guard - window : center - guard] = 0.25
    signal[center] = 0.5

    result = _metrics({"capb": signal}, center, sample_rate)["capb"]

    assert result["pre_echo_mean_square"] == pytest.approx(0.25**2)
    assert result["peak"] == pytest.approx(0.5)


def test_source_duration_converts_target_samples_to_ms() -> None:
    case = RateCase("44k1", 44_100, checkpoint=Path("unused.pt"))
    assert source_duration_ms(case, 88_200) == pytest.approx(1_000.0)
