"""Tests for Stage 1 hard metrics CLI script."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scripts.evaluate_stage1 import (
    RingingDatasetSummary,
    Stage1GateConfig,
    _evaluate_stage1_gates,
    main,
)

from totton_audio_de_mirroring.evaluation.metrics import (
    DatasetEvaluationResult,
    SampleEvaluationResult,
    Stage1HardMetrics,
)
from totton_audio_de_mirroring.evaluation.mirror_metrics import (
    MirrorDatasetEvaluationResult,
    MirrorReductionMetrics,
    MirrorSampleEvaluationResult,
)


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


def _build_gate_inputs(
    *,
    energy_violation_rate: float,
    symmetry_reduction_ratio: float,
    plateau_rms_ratio: float,
    plateau_p2p_ratio: float,
    overshoot_abs_delta: float,
    ringing_ratio_delta: float,
) -> tuple[
    DatasetEvaluationResult, MirrorDatasetEvaluationResult, RingingDatasetSummary
]:
    hard_metrics = Stage1HardMetrics(
        lb_amplitude_error_db=-30.0,
        lb_phase_error_deg=1.0,
        lb_group_delay_error_samples=1.0,
        mirror_reduction_ratio=0.8,
        hb_energy=1.0e-4,
        hb_energy_cap=1.0e-3,
        hb_energy_cap_violated=False,
        touch_metric=0.1,
    )
    hard_result = DatasetEvaluationResult(
        samples=(SampleEvaluationResult(sample_id="s0", metrics=hard_metrics),),
        mean_metrics=hard_metrics,
        hb_energy_cap_violation_rate=energy_violation_rate,
    )

    mirror_metrics = MirrorReductionMetrics(
        symmetry_score_before=1.0,
        symmetry_score_after=0.2,
        symmetry_reduction_ratio=symmetry_reduction_ratio,
        mirror_band_energy_before=1.0,
        mirror_band_energy_after=0.2,
        mirror_band_energy_reduction_ratio=0.8,
        stripe_score_before=1.0,
        stripe_score_after=0.2,
        stripe_reduction_ratio=0.8,
    )
    mirror_result = MirrorDatasetEvaluationResult(
        samples=(MirrorSampleEvaluationResult(sample_id="s0", metrics=mirror_metrics),),
        mean_metrics=mirror_metrics,
        target_reduction_ratio=0.70,
        symmetry_target_pass_rate=1.0,
    )
    ringing_summary = RingingDatasetSummary(
        num_samples=1,
        mean_plateau_ripple_rms_before=0.01,
        mean_plateau_ripple_rms_after=0.01,
        mean_plateau_ripple_rms_ratio=plateau_rms_ratio,
        mean_plateau_ripple_p2p_before=0.02,
        mean_plateau_ripple_p2p_after=0.02,
        mean_plateau_ripple_p2p_ratio=plateau_p2p_ratio,
        mean_overshoot_abs_before=0.0,
        mean_overshoot_abs_after=overshoot_abs_delta,
        mean_overshoot_abs_delta=overshoot_abs_delta,
        mean_ringing_ratio_before=1.0,
        mean_ringing_ratio_after=1.0 + ringing_ratio_delta,
        mean_ringing_ratio_delta=ringing_ratio_delta,
    )
    return hard_result, mirror_result, ringing_summary


def test_evaluate_stage1_gates_accepts_values_on_threshold() -> None:
    """Threshold-equal values should pass all Stage1 gates."""
    result, mirror_result, ringing_summary = _build_gate_inputs(
        energy_violation_rate=0.0,
        symmetry_reduction_ratio=0.70,
        plateau_rms_ratio=1.10,
        plateau_p2p_ratio=1.10,
        overshoot_abs_delta=5.0e-3,
        ringing_ratio_delta=0.0,
    )
    gate_config = Stage1GateConfig(
        mirror_target_reduction=0.70,
        max_plateau_ripple_rms_ratio=1.10,
        max_plateau_ripple_p2p_ratio=1.10,
        max_overshoot_abs_increase=5.0e-3,
        require_nonpositive_ringing_ratio_delta=True,
    )

    gate_status = _evaluate_stage1_gates(
        result=result,
        mirror_result=mirror_result,
        ringing_summary=ringing_summary,
        gate_config=gate_config,
    )

    assert gate_status["stage1_acceptance_pass"] is True
    assert gate_status["energy_cap"]["passed"] is True
    assert gate_status["mirror_reduction"]["passed"] is True
    assert gate_status["ringing_regression"]["passed"] is True


def test_evaluate_stage1_gates_rejects_ringing_ratio_delta_when_strict() -> None:
    """Positive ringing-ratio delta should fail when strict mode is enabled."""
    result, mirror_result, ringing_summary = _build_gate_inputs(
        energy_violation_rate=0.0,
        symmetry_reduction_ratio=0.80,
        plateau_rms_ratio=1.0,
        plateau_p2p_ratio=1.0,
        overshoot_abs_delta=0.0,
        ringing_ratio_delta=1.0e-6,
    )
    strict_gate_config = Stage1GateConfig(
        mirror_target_reduction=0.70,
        max_plateau_ripple_rms_ratio=1.10,
        max_plateau_ripple_p2p_ratio=1.10,
        max_overshoot_abs_increase=5.0e-3,
        require_nonpositive_ringing_ratio_delta=True,
    )
    relaxed_gate_config = Stage1GateConfig(
        mirror_target_reduction=0.70,
        max_plateau_ripple_rms_ratio=1.10,
        max_plateau_ripple_p2p_ratio=1.10,
        max_overshoot_abs_increase=5.0e-3,
        require_nonpositive_ringing_ratio_delta=False,
    )

    strict_status = _evaluate_stage1_gates(
        result=result,
        mirror_result=mirror_result,
        ringing_summary=ringing_summary,
        gate_config=strict_gate_config,
    )
    relaxed_status = _evaluate_stage1_gates(
        result=result,
        mirror_result=mirror_result,
        ringing_summary=ringing_summary,
        gate_config=relaxed_gate_config,
    )

    assert strict_status["ringing_regression"]["passed"] is False
    assert strict_status["stage1_acceptance_pass"] is False
    assert relaxed_status["ringing_regression"]["passed"] is True
    assert relaxed_status["stage1_acceptance_pass"] is True
