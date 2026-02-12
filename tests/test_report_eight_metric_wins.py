"""Tests for Issue #109 8-metric win/loss report script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scripts.report_eight_metric_wins import main


def test_issue109_report_generates_all_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate report with microstructure + lowband metrics.

    Physical Basis:
        End-to-end generation verifies reproducible output formatting and winner
        aggregation under mixed metric directions.
    """

    metrics_root, audio_root, target_root = _prepare_fixture_tree(tmp_path)
    output_dir = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_eight_metric_wins.py",
            "--metrics-root",
            str(metrics_root),
            "--audio-root",
            str(audio_root),
            "--target-root",
            str(target_root),
            "--methods",
            "bessel_iir",
            "distillation_nn",
            "--output-dir",
            str(output_dir),
            "--report-name",
            "issue109_test",
        ],
    )

    main()

    report_md = output_dir / "issue109_test.md"
    aggregate_csv = output_dir / "issue109_test_aggregate.csv"
    per_file_csv = output_dir / "issue109_test_per_file.csv"
    report_json = output_dir / "issue109_test.json"

    assert report_md.exists()
    assert aggregate_csv.exists()
    assert per_file_csv.exists()
    assert report_json.exists()

    text = report_md.read_text(encoding="utf-8")
    assert "Issue #109 8指標 勝敗表" in text
    assert "Aggregate Winners" in text
    assert "Per-file Winner Counts" in text
    assert "distillation_nn" in text


def test_issue109_report_works_with_metrics_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate report without lowband inputs.

    Physical Basis:
        Pipeline must still provide winners for MPS/TFS/Attack/Bass when audio
        artifacts for lowband evaluation are not available.
    """

    metrics_root, _, _ = _prepare_fixture_tree(tmp_path)
    output_dir = tmp_path / "reports_only_metrics"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_eight_metric_wins.py",
            "--metrics-root",
            str(metrics_root),
            "--methods",
            "bessel_iir",
            "distillation_nn",
            "--output-dir",
            str(output_dir),
            "--report-name",
            "issue109_metrics_only",
        ],
    )

    main()

    report_md = output_dir / "issue109_metrics_only.md"
    assert report_md.exists()
    text = report_md.read_text(encoding="utf-8")
    assert "Lowband Wave Error (dB)" in text
    assert "n/a" in text


def _prepare_fixture_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create synthetic metric/audio fixture tree.

    Physical Basis:
        Controlled synthetic fixtures keep winner decisions deterministic while
        exercising both JSON-driven and waveform-driven metric paths.
    """

    metrics_root = tmp_path / "metrics"
    audio_root = tmp_path / "audio"
    target_root = tmp_path / "target"
    for path in (metrics_root, audio_root, target_root):
        path.mkdir(parents=True, exist_ok=True)

    methods = ["bessel_iir", "distillation_nn"]
    for method in methods:
        (metrics_root / method).mkdir(parents=True, exist_ok=True)
        (audio_root / method).mkdir(parents=True, exist_ok=True)

    sample_rate = 88_200
    duration = 0.1
    num_samples = int(sample_rate * duration)
    timeline = np.arange(num_samples, dtype=np.float64) / float(sample_rate)
    base_signal = (0.2 * np.sin(2.0 * np.pi * 1000.0 * timeline)).astype(np.float64)

    target_path = target_root / "thd_1khz_88200_hz_24bit_v1.wav"
    wavfile.write(target_path, sample_rate, base_signal.astype(np.float32))

    bessel_audio_path = audio_root / "bessel_iir" / "thd_1khz_88200_full.wav"
    distill_audio_path = audio_root / "distillation_nn" / "thd_1khz_88200_full.wav"
    wavfile.write(
        bessel_audio_path,
        sample_rate,
        (base_signal + 0.02 * np.sin(2.0 * np.pi * 3000.0 * timeline)).astype(
            np.float32
        ),
    )
    wavfile.write(
        distill_audio_path,
        sample_rate,
        (base_signal + 0.005 * np.sin(2.0 * np.pi * 3000.0 * timeline)).astype(
            np.float32
        ),
    )

    _write_metric_json(
        path=metrics_root / "bessel_iir" / "thd_1khz_88200_full.json",
        mps_corr=0.97,
        mps_dist=0.05,
        tfs_corr=0.90,
        attack_p95=0.4,
        bass_corr=0.70,
    )
    _write_metric_json(
        path=metrics_root / "distillation_nn" / "thd_1khz_88200_full.json",
        mps_corr=0.99,
        mps_dist=0.01,
        tfs_corr=0.95,
        attack_p95=0.1,
        bass_corr=0.85,
    )

    return metrics_root, audio_root, target_root


def _write_metric_json(
    *,
    path: Path,
    mps_corr: float,
    mps_dist: float,
    tfs_corr: float,
    attack_p95: float,
    bass_corr: float,
) -> None:
    """Write synthetic microstructure report JSON.

    Physical Basis:
        Script integration tests depend on JSON schema compatibility with
        microstructure report outputs.
    """

    payload = {
        "metrics": {
            "ch0": {
                "mps": {
                    "mps_correlation": mps_corr,
                    "mps_distance": mps_dist,
                },
                "tfs": {
                    "mean_correlation": tfs_corr,
                },
                "transient": {
                    "attack_time_delta_p95_ms": attack_p95,
                },
                "bass": {
                    "cycle_shape_corr_mean": bass_corr,
                },
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
