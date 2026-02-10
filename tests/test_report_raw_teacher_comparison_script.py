"""Tests for raw88-vs-bessel Stage1 comparison report script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from scripts.report_raw_teacher_comparison import main


def test_report_raw_teacher_comparison_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = _make_run_dir(tmp_path=tmp_path, teacher_tag="raw88", seed=1234)
    bessel_dir = _make_run_dir(tmp_path=tmp_path, teacher_tag="bessel", seed=1234)
    output_md = tmp_path / "comparison.md"
    output_csv = tmp_path / "comparison.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_raw_teacher_comparison.py",
            "--raw-run-dir",
            str(raw_dir),
            "--bessel-run-dir",
            str(bessel_dir),
            "--output-md",
            str(output_md),
            "--output-csv",
            str(output_csv),
        ],
    )
    main()

    md = output_md.read_text(encoding="utf-8")
    csv_text = output_csv.read_text(encoding="utf-8")
    assert "Stage1 Raw88 vs Bessel Comparison" in md
    assert "Metric Table" in md
    assert "metric,better,raw88,bessel,delta_raw_minus_bessel,winner" in csv_text


def test_report_raw_teacher_comparison_rejects_unmatched_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = _make_run_dir(tmp_path=tmp_path, teacher_tag="raw88", seed=1234)
    bessel_dir = _make_run_dir(tmp_path=tmp_path, teacher_tag="bessel", seed=777)
    output_md = tmp_path / "comparison.md"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_raw_teacher_comparison.py",
            "--raw-run-dir",
            str(raw_dir),
            "--bessel-run-dir",
            str(bessel_dir),
            "--output-md",
            str(output_md),
        ],
    )
    with pytest.raises(RuntimeError, match="matched conditions"):
        main()


def _make_run_dir(*, tmp_path: Path, teacher_tag: str, seed: int) -> Path:
    run_dir = tmp_path / teacher_tag / f"stage1_{teacher_tag}_nmse_20260210_s{seed}"
    selected_dir = run_dir / "selected"
    selected_dir.mkdir(parents=True)

    manifest = {
        "teacher_tag": teacher_tag,
        "run_id": f"stage1_{teacher_tag}_nmse_20260210_s{seed}",
        "training_config": {"seed": seed},
        "train_config_sha256": "same_train_hash",
        "gate_thresholds": {"mirror_target": 0.70},
        "args": {
            "eval_input_dir": "tests/fixtures/golden_samples/stage1/input",
            "imd_naive_dir": "tests/fixtures/golden_samples/imd/naive",
            "eval_glob": "*.npy",
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    selected_checkpoint = str(run_dir / "stage1_best.pt")
    candidate = {
        "checkpoint_path": selected_checkpoint,
        "hard_summary": {
            "hb_energy_cap_violation_rate": 0.0,
            "lb_amplitude_error_db": -30.0,
            "lb_phase_error_deg": 2.0,
            "lb_group_delay_error_samples": 100.0,
        },
        "mirror_summary": {"symmetry_reduction_ratio": 0.9},
        "imd_summary": {"mean_thdn_improvement_db": 1.2},
        "ringing_summary": {
            "mean_plateau_ripple_rms_ratio": 1.01,
            "mean_ringing_ratio_delta": -0.01,
        },
    }
    selection = {
        "selected_checkpoint": selected_checkpoint,
        "candidates": [candidate],
    }
    (selected_dir / "selection_report.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    return run_dir
