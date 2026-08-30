"""Cross-rate CAPB release-quality checks beyond the frozen probe gates."""

from __future__ import annotations

from typing import Any

_DISTORTION_METRICS = (
    "thd_1khz_20khz_db",
    "smpte_imd_db",
    "ccif_imd_db",
    "added_am_sideband_db",
)


def normalized_prototype_position(
    value: float,
    gentle: float,
    sharp: float,
) -> float:
    """Normalize a transient metric along its rate-local prototype span.

    Physical Basis:
        The 44.1 and 48 kHz prototype banks have different absolute impulse
        energies. Their dimensionless position between the validated gentle
        and sharp endpoints is therefore the meaningful cross-rate measure.
    """
    span = sharp - gentle
    if span <= 0.0:
        raise ValueError("sharp metric must be greater than gentle metric.")
    return (value - gentle) / span


def evaluate_release_quality(
    distortion: dict[str, Any],
    impulse: dict[str, Any],
    gate_44k1: dict[str, Any],
    gate_48k: dict[str, Any],
    *,
    distortion_tolerance_db: float = 1.0,
    transient_position_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Evaluate 48 kHz quality against 44.1 kHz and local FIR floors.

    Args:
        distortion: Strict-FP32 distortion report for both rate families.
        impulse: Impulse metrics for both rate families.
        gate_44k1: Strict frozen-gate report for 44.1 kHz.
        gate_48k: Strict frozen-gate report for 48 kHz.
        distortion_tolerance_db: Allowed margin above the applicable dB floor.
        transient_position_tolerance: Allowed normalized transient-position
            increase against the 44.1 kHz checkpoint.

    Returns:
        Serializable release decision and individual checks.

    Raises:
        ValueError: If tolerances or precision metadata are invalid.

    Physical Basis:
        Raw cross-rate distortion is bounded by different fixed-FIR float32
        accumulation floors. The controller is accepted only when it stays
        within the worse of the 44.1 kHz result and its own rate-local floor,
        while transient response is compared on the prototype continuum.
    """
    if distortion_tolerance_db < 0.0 or transient_position_tolerance < 0.0:
        raise ValueError("Release-quality tolerances must be non-negative.")
    _require_strict_precision(distortion, "distortion")
    _require_strict_precision(gate_44k1, "44.1 kHz gate")
    _require_strict_precision(gate_48k, "48 kHz gate")

    checks: list[dict[str, Any]] = [
        _boolean_check("44k1_frozen_gates", bool(gate_44k1["all_passed"])),
        _boolean_check("48k_frozen_gates", bool(gate_48k["all_passed"])),
    ]
    metrics_44 = distortion["44k1"]["distortion"]["capb"]
    metrics_48 = distortion["48k"]["distortion"]["capb"]
    fixed_48 = distortion["48k"]["distortion"]["torch_sharp"]
    for metric in _DISTORTION_METRICS:
        threshold = max(metrics_44[metric], fixed_48[metric]) + distortion_tolerance_db
        checks.append(_numeric_check(f"48k_{metric}", metrics_48[metric], threshold))

    impulse_44 = impulse["44k1"]["metrics"]
    impulse_48 = impulse["48k"]["metrics"]
    for metric in ("local_energy", "peak"):
        position_44 = normalized_prototype_position(
            impulse_44["capb"][metric],
            impulse_44["gentle"][metric],
            impulse_44["sharp"][metric],
        )
        position_48 = normalized_prototype_position(
            impulse_48["capb"][metric],
            impulse_48["gentle"][metric],
            impulse_48["sharp"][metric],
        )
        checks.append(
            _numeric_check(
                f"48k_normalized_{metric}",
                position_48,
                position_44 + transient_position_tolerance,
            )
        )

    checks.append(
        _numeric_check(
            "48k_pre_echo_mean_square",
            impulse_48["capb"]["pre_echo_mean_square"],
            impulse_44["capb"]["pre_echo_mean_square"],
        )
    )
    checks.append(
        _numeric_check(
            "48k_impulse_train_gain_error_db",
            _gate_row_value(gate_48k, "G5_gain", "impulse_train_10ms"),
            _gate_row_value(gate_44k1, "G5_gain", "impulse_train_10ms"),
        )
    )
    return {
        "all_passed": all(bool(check["passed"]) for check in checks),
        "distortion_tolerance_db": distortion_tolerance_db,
        "transient_position_tolerance": transient_position_tolerance,
        "checks": checks,
    }


def _require_strict_precision(report: dict[str, Any], label: str) -> None:
    execution = report.get("execution")
    if not isinstance(execution, dict):
        raise ValueError(f"{label} report is missing execution metadata.")
    if execution.get("precision_mode") != "strict_fp32":
        raise ValueError(f"{label} report must use strict_fp32.")


def _gate_row_value(report: dict[str, Any], gate_id: str, probe_id: str) -> float:
    for gate in report.get("gates", []):
        if gate.get("gate_id") != gate_id:
            continue
        for row in gate.get("rows", []):
            if row.get("probe_id") == probe_id:
                return float(row["value"])
    raise ValueError(f"Missing {gate_id}/{probe_id} gate row.")


def _boolean_check(check_id: str, passed: bool) -> dict[str, Any]:
    return {"check_id": check_id, "passed": passed}


def _numeric_check(check_id: str, value: float, threshold: float) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "value": float(value),
        "threshold": float(threshold),
        "passed": value <= threshold,
    }
