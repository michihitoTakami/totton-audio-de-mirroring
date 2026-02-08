"""Tests for Stage 1 hard metrics CLI script."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scripts.evaluate_stage1 import main


@pytest.fixture
def paired_npy_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create small paired input/output directories for CLI tests."""
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()

    sample_rate = 88_200
    duration_sec = 0.1
    time = np.arange(int(sample_rate * duration_sec), dtype=np.float64) / sample_rate

    a_in = np.sin(2.0 * np.pi * 1_000.0 * time)
    a_out = a_in.copy()
    b_in = 0.2 * np.sin(2.0 * np.pi * 30_000.0 * time)
    b_out = 1.2 * np.sin(2.0 * np.pi * 30_000.0 * time)

    np.save(input_dir / "a.npy", a_in)
    np.save(output_dir / "a.npy", a_out)
    np.save(input_dir / "b.npy", b_in)
    np.save(output_dir / "b.npy", b_out)

    return input_dir, output_dir


@pytest.fixture
def ringing_regression_npy_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create paired square-wave signals with intentional ringing regression."""
    input_dir = tmp_path / "ring_inputs"
    output_dir = tmp_path / "ring_outputs"
    input_dir.mkdir()
    output_dir.mkdir()

    sample_rate = 88_200
    num_samples = int(sample_rate * 0.05)
    half = num_samples // 2
    square = np.concatenate(
        [
            np.full(half, 0.5, dtype=np.float64),
            np.full(num_samples - half, -0.5, dtype=np.float64),
        ]
    )
    ring_kernel = np.asarray([1.0, -0.9, 0.8, -0.7, 0.6, -0.5], dtype=np.float64)
    ringing_square = np.convolve(square, ring_kernel, mode="same")

    np.save(input_dir / "square.npy", square)
    np.save(output_dir / "square.npy", ringing_square)
    return input_dir, output_dir


def test_cli_writes_json_and_csv(
    paired_npy_dirs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should emit parseable JSON and CSV reports."""
    input_dir, output_dir = paired_npy_dirs
    json_path = tmp_path / "report" / "metrics.json"
    csv_path = tmp_path / "report" / "metrics.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_stage1.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--json",
            str(json_path),
            "--csv",
            str(csv_path),
            "--print-json",
            "--energy-cap",
            "1e-4",
        ],
    )

    main()

    payload = json.loads(json_path.read_text())
    assert payload["summary"]["num_samples"] == 2
    assert "lb_phase_error_deg" in payload["summary"]
    assert "mirror_metrics" in payload
    assert "symmetry_reduction_ratio" in payload["mirror_metrics"]["summary"]
    assert "ringing_metrics" in payload
    assert "mean_plateau_ripple_rms_ratio" in payload["ringing_metrics"]["summary"]
    assert "gates" in payload
    assert payload["gates"]["stage1_acceptance_pass"] is False
    assert payload["gates"]["energy_cap"]["passed"] is False

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert "touch_metric" in rows[0]
    assert "plateau_ripple_rms_ratio" in rows[0]
    assert "gate_stage1_acceptance_pass" in rows[0]
    assert "gate_ringing_regression_pass" in rows[0]


def test_cli_strict_energy_cap_returns_exit_code_2(
    paired_npy_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should fail with exit code 2 when cap violation exists."""
    input_dir, output_dir = paired_npy_dirs
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_stage1.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--energy-cap",
            "1e-4",
            "--strict-energy-cap",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2


def test_cli_raises_when_output_pair_is_missing(
    paired_npy_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should raise FileNotFoundError for missing output pair."""
    input_dir, output_dir = paired_npy_dirs
    (output_dir / "a.npy").unlink()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_stage1.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(FileNotFoundError, match="Missing output pair"):
        main()


def test_cli_exports_mirror_visualizations(
    paired_npy_dirs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should export mirror visualization images when requested."""
    input_dir, output_dir = paired_npy_dirs
    json_path = tmp_path / "report" / "metrics.json"
    visual_dir = tmp_path / "report" / "mirror_visuals"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_stage1.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--json",
            str(json_path),
            "--mirror-visual-dir",
            str(visual_dir),
            "--mirror-visual-limit",
            "1",
        ],
    )

    main()

    payload = json.loads(json_path.read_text())
    exported = payload["mirror_visualizations"]
    assert len(exported) == 1
    assert Path(exported[0]).exists()


def test_cli_strict_mirror_reduction_returns_exit_code_3(
    paired_npy_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should fail with exit code 3 when mirror reduction target is unmet."""
    input_dir, output_dir = paired_npy_dirs
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_stage1.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--strict-mirror-reduction",
            "--mirror-target-reduction",
            "0.70",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 3


def test_cli_writes_ringing_json_and_csv(
    paired_npy_dirs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should emit dedicated ringing JSON/CSV reports when requested."""
    input_dir, output_dir = paired_npy_dirs
    ringing_json = tmp_path / "report" / "ringing.json"
    ringing_csv = tmp_path / "report" / "ringing.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_stage1.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--ringing-json",
            str(ringing_json),
            "--ringing-csv",
            str(ringing_csv),
        ],
    )

    main()

    ringing_payload = json.loads(ringing_json.read_text(encoding="utf-8"))
    assert ringing_payload["summary"]["num_samples"] == 2
    assert len(ringing_payload["samples"]) == 2
    assert "plateau_ripple_p2p_ratio" in ringing_payload["samples"][0]

    with ringing_csv.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert "overshoot_abs_delta" in rows[0]


def test_cli_strict_ringing_regression_returns_exit_code_4(
    ringing_regression_npy_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should fail with exit code 4 when ringing-regression gate is unmet."""
    input_dir, output_dir = ringing_regression_npy_dirs
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_stage1.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--strict-ringing-regression",
            "--max-plateau-ripple-rms-ratio",
            "1.0",
            "--max-plateau-ripple-p2p-ratio",
            "1.0",
            "--max-overshoot-abs-increase",
            "0.0",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 4


def test_cli_multiple_strict_failures_return_exit_code_5(
    paired_npy_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should use combined strict exit code when multiple gates fail."""
    input_dir, output_dir = paired_npy_dirs
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_stage1.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--energy-cap",
            "1e-4",
            "--strict-energy-cap",
            "--strict-mirror-reduction",
            "--mirror-target-reduction",
            "0.70",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 5
