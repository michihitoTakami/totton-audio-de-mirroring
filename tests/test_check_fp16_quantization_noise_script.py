"""Tests for FP16 quantization noise checker script."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts import check_fp16_quantization_noise as script


def test_quantize_fp16_roundtrip_rejects_non_finite() -> None:
    signal = np.array([0.0, np.nan, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite"):
        _ = script.quantize_fp16_roundtrip(signal)


def test_compute_noise_metrics_reports_positive_snr() -> None:
    sample_rate = 88_200
    time_axis = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    signal = 0.2 * np.sin(2.0 * np.pi * 30_000.0 * time_axis)

    metrics = script.compute_noise_metrics(signal)

    assert metrics.snr_db > 70.0
    assert metrics.error_rms_dbfs < -85.0
    assert metrics.error_peak > 0.0


def test_evaluate_fp16_quantization_noise_contains_default_cases() -> None:
    metrics = script.evaluate_fp16_quantization_noise(
        sample_rate=88_200,
        duration_sec=1.0,
    )

    assert set(metrics.keys()) == {
        "hb_sine_30k_amp0p5",
        "hb_sine_30k_amp0p1",
        "hb_multitone",
    }


def test_fails_thresholds_detects_regression() -> None:
    metrics = {
        "dummy": script.NoiseMetrics(
            signal_rms=0.1,
            error_rms=0.01,
            error_peak=0.02,
            snr_db=20.0,
            error_rms_dbfs=-40.0,
        )
    }

    failures = script._fails_thresholds(
        metrics,
        min_snr_db=70.0,
        max_error_rms_dbfs=-80.0,
    )

    assert len(failures) == 2


def test_main_json_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        script,
        "parse_args",
        lambda: script.argparse.Namespace(
            sample_rate=88_200,
            duration_sec=1.0,
            min_snr_db=70.0,
            max_error_rms_dbfs=-80.0,
            json=True,
        ),
    )

    script.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["failures"] == []


def test_main_exits_nonzero_when_threshold_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        script,
        "parse_args",
        lambda: script.argparse.Namespace(
            sample_rate=88_200,
            duration_sec=1.0,
            min_snr_db=120.0,
            max_error_rms_dbfs=-120.0,
            json=False,
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        script.main()

    assert int(exc_info.value.code) == 1
