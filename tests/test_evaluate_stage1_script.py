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

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert "touch_metric" in rows[0]


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
