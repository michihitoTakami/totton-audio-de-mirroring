"""Regression test suite for Stage 1 quality metrics using golden samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from totton_audio_de_mirroring.evaluation.imd_proxy import evaluate_imd_proxy
from totton_audio_de_mirroring.evaluation.metrics import (
    evaluate_dataset,
    sample_result_to_flat_dict,
)
from totton_audio_de_mirroring.evaluation.mirror_metrics import (
    evaluate_mirror_reduction_dataset,
    mirror_dataset_result_to_payload,
)

_GOLDEN_ROOT = Path("tests/fixtures/golden_samples")
_BASELINE_PATH = _GOLDEN_ROOT / "regression_baseline.json"
_STAGE1_INPUT_DIR = _GOLDEN_ROOT / "stage1" / "input"
_STAGE1_OUTPUT_DIR = _GOLDEN_ROOT / "stage1" / "output"
_IMD_NAIVE_DIR = _GOLDEN_ROOT / "imd" / "naive"
_IMD_NMSE_DIR = _GOLDEN_ROOT / "imd" / "nmse"

_DEFAULT_REL_TOL = 1.0e-2
_DEFAULT_ABS_TOL = 1.0e-7
_KEY_TOLERANCES: dict[str, tuple[float, float]] = {
    "lb_phase_error_deg": (2.0e-2, 1.0e-6),
    "lb_group_delay_error_samples": (2.0e-2, 1.0e-4),
    "touch_metric": (1.0e-2, 1.0e-6),
}


def _load_baseline() -> dict[str, Any]:
    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


def _load_npy_pairs(
    input_dir: Path,
    output_dir: Path,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    input_paths = sorted(input_dir.glob("*.npy"))
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for input_path in input_paths:
        output_path = output_dir / input_path.name
        if not output_path.exists():
            raise FileNotFoundError(f"Missing golden output: {output_path}")

        input_signal = np.asarray(np.load(input_path), dtype=np.float64)
        output_signal = np.asarray(np.load(output_path), dtype=np.float64)
        pairs.append((input_path.stem, input_signal, output_signal))

    if len(pairs) == 0:
        raise ValueError(f"No golden samples found in: {input_dir}")
    return pairs


def _resolve_tolerance(key: str) -> tuple[float, float]:
    for suffix, tolerance in _KEY_TOLERANCES.items():
        if key.endswith(suffix):
            return tolerance
    return _DEFAULT_REL_TOL, _DEFAULT_ABS_TOL


def _assert_close(expected: float, actual: float, key: str) -> None:
    rel_tol, abs_tol = _resolve_tolerance(key)
    if not np.isclose(actual, expected, rtol=rel_tol, atol=abs_tol):
        raise AssertionError(
            f"Metric mismatch for {key}: expected={expected}, actual={actual}"
        )


def _assert_mapping_close(
    expected: dict[str, Any],
    actual: dict[str, Any],
    prefix: str,
) -> None:
    for key, expected_value in expected.items():
        full_key = f"{prefix}.{key}"
        if key not in actual:
            raise AssertionError(f"Missing key in actual payload: {full_key}")
        actual_value = actual[key]
        if isinstance(expected_value, bool):
            assert isinstance(actual_value, bool)
            assert actual_value is expected_value, full_key
            continue
        if isinstance(expected_value, int):
            assert int(actual_value) == expected_value, full_key
            continue
        if isinstance(expected_value, float):
            _assert_close(expected_value, float(actual_value), full_key)
            continue
        if isinstance(expected_value, str):
            assert str(actual_value) == expected_value, full_key
            continue
        raise TypeError(f"Unsupported type for {full_key}: {type(expected_value)}")


def test_stage1_hard_metrics_regression_against_golden_samples() -> None:
    """Hard metrics should stay within golden baseline tolerance.

    Physical Basis:
        Fixed golden input/output pairs capture expected Stage 1 behavior for
        low-band preservation, mirror suppression, and HB energy-cap safety.
    """
    baseline = _load_baseline()
    sample_rate = int(baseline["sample_rate"])
    pairs = _load_npy_pairs(_STAGE1_INPUT_DIR, _STAGE1_OUTPUT_DIR)

    result = evaluate_dataset(samples=pairs, sample_rate=sample_rate, energy_cap=1.0e-3)
    expected_summary = baseline["stage1_hard_metrics"]["summary"]
    # Build explicit summary mapping with baseline keys to avoid accidental key drift.
    summary_payload = {
        "num_samples": len(result.samples),
        "hb_energy_cap_violation_rate": result.hb_energy_cap_violation_rate,
        "lb_amplitude_error_db": result.mean_metrics.lb_amplitude_error_db,
        "lb_phase_error_deg": result.mean_metrics.lb_phase_error_deg,
        "lb_group_delay_error_samples": result.mean_metrics.lb_group_delay_error_samples,
        "mirror_reduction_ratio": result.mean_metrics.mirror_reduction_ratio,
        "hb_energy": result.mean_metrics.hb_energy,
        "hb_energy_cap": result.mean_metrics.hb_energy_cap,
        "hb_energy_cap_violated": result.mean_metrics.hb_energy_cap_violated,
        "touch_metric": result.mean_metrics.touch_metric,
    }
    _assert_mapping_close(expected_summary, summary_payload, "stage1_hard.summary")

    expected_samples = {
        sample["sample_id"]: sample
        for sample in baseline["stage1_hard_metrics"]["samples"]
    }
    actual_samples = {
        sample.sample_id: sample_result_to_flat_dict(sample)
        for sample in result.samples
    }
    assert set(actual_samples) == set(expected_samples)
    for sample_id, expected in expected_samples.items():
        _assert_mapping_close(
            expected,
            actual_samples[sample_id],
            f"stage1_hard.samples.{sample_id}",
        )


def test_mirror_metrics_regression_against_golden_samples() -> None:
    """Mirror-metric summary should remain stable on the golden dataset.

    Physical Basis:
        STFT symmetry and mirror-band energy reduction are direct indicators of
        alias/mirror artifact suppression quality.
    """
    baseline = _load_baseline()
    sample_rate = int(baseline["sample_rate"])
    pairs = _load_npy_pairs(_STAGE1_INPUT_DIR, _STAGE1_OUTPUT_DIR)

    result = evaluate_mirror_reduction_dataset(
        samples=pairs,
        sample_rate=sample_rate,
        target_reduction_ratio=0.70,
    )
    payload = mirror_dataset_result_to_payload(result)

    _assert_mapping_close(
        baseline["mirror_metrics"]["summary"],
        payload["summary"],
        "mirror.summary",
    )

    expected_samples = {
        sample["sample_id"]: sample for sample in baseline["mirror_metrics"]["samples"]
    }
    actual_samples = {sample["sample_id"]: sample for sample in payload["samples"]}
    assert set(actual_samples) == set(expected_samples)
    for sample_id, expected in expected_samples.items():
        _assert_mapping_close(
            expected,
            actual_samples[sample_id],
            f"mirror.samples.{sample_id}",
        )


def test_imd_proxy_regression_against_golden_samples() -> None:
    """IMD proxy metrics should match the golden baseline for each sample.

    Physical Basis:
        The IMD proxy verifies that HB suppression lowers audible-band
        distortion after mild nonlinearity.
    """
    baseline = _load_baseline()
    sample_rate = int(baseline["sample_rate"])
    clip_drive = float(baseline["clip_drive"])
    pairs = _load_npy_pairs(_IMD_NAIVE_DIR, _IMD_NMSE_DIR)

    for sample_id, naive_signal, nmse_signal in pairs:
        metrics = evaluate_imd_proxy(
            naive_signal=naive_signal,
            nmse_signal=nmse_signal,
            sample_rate=sample_rate,
            clip_drive=clip_drive,
            num_taps=1025,
        )
        actual = {
            "audible_distortion_reduction_db": metrics.audible_distortion_reduction_db,
            "thdn_improvement_db": metrics.thdn_improvement_db,
            "nmse_has_lower_imd": metrics.nmse_has_lower_imd,
            "thdn_improvement_over_10db": metrics.thdn_improvement_over_10db,
        }
        _assert_mapping_close(
            baseline["imd_proxy"][sample_id],
            actual,
            f"imd.{sample_id}",
        )


def test_regression_gate_detects_degraded_quality() -> None:
    """Regression gate should fail when mirror artifacts are re-injected.

    Physical Basis:
        Reintroducing mirror-like high-band tones increases symmetry and energy,
        which should be captured by hard and mirror metrics.
    """
    baseline = _load_baseline()
    sample_rate = int(baseline["sample_rate"])
    pairs = _load_npy_pairs(_STAGE1_INPUT_DIR, _STAGE1_OUTPUT_DIR)

    degraded_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sample_id, input_signal, output_signal in pairs:
        time = np.arange(output_signal.size, dtype=np.float64) / float(sample_rate)
        reinjected = 0.16 * np.sin(2.0 * np.pi * 20_500.0 * time) + 0.16 * np.sin(
            2.0 * np.pi * 23_600.0 * time
        )
        degraded_output = np.asarray(output_signal + reinjected, dtype=np.float64)
        degraded_pairs.append((sample_id, input_signal, degraded_output))

    degraded_hard = evaluate_dataset(
        samples=degraded_pairs,
        sample_rate=sample_rate,
        energy_cap=1.0e-3,
    )
    degraded_mirror = evaluate_mirror_reduction_dataset(
        samples=degraded_pairs,
        sample_rate=sample_rate,
        target_reduction_ratio=0.70,
    )

    base_hard = baseline["stage1_hard_metrics"]["summary"]
    base_mirror = baseline["mirror_metrics"]["summary"]

    assert (
        degraded_hard.mean_metrics.mirror_reduction_ratio
        < float(base_hard["mirror_reduction_ratio"]) - 0.20
    )
    assert (
        degraded_mirror.mean_metrics.symmetry_reduction_ratio
        < float(base_mirror["symmetry_reduction_ratio"]) - 0.20
    )
    assert degraded_hard.hb_energy_cap_violation_rate > float(
        base_hard["hb_energy_cap_violation_rate"]
    )
