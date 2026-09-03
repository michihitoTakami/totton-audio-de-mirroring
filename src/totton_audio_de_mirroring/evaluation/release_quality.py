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
            increase against the 44.1 kHz checkpoint, on top of any documented
            gap between the two families' ``focused_gentle_fraction``.

    Returns:
        Serializable release decision and individual checks.

    Raises:
        ValueError: If tolerances or precision metadata are invalid.

    Physical Basis:
        Raw cross-rate distortion is bounded by different fixed-FIR float32
        accumulation floors. The controller is accepted only when it stays
        within the worse of the 44.1 kHz result and its own rate-local floor,
        while transient response is compared on the prototype continuum.
        The 48 kHz gentle endpoint loses more impulse gain than the 44.1 kHz
        one, so a checkpoint pair may deliberately keep a larger middle share
        at 48 kHz through its routing prior; that documented fraction gap is
        added to the position allowance, and impulse-train gain is judged
        against the frozen G5 gate for both families instead of demanding
        48 kHz to beat 44.1 kHz, which would force 48 kHz back toward the
        ringing sharp endpoint.
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
    fraction_gap = max(
        0.0,
        _gentle_fraction(impulse["44k1"]) - _gentle_fraction(impulse["48k"]),
    )
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
                position_44 + transient_position_tolerance + fraction_gap,
            )
        )

    checks.append(
        _numeric_check(
            "48k_pre_echo_mean_square",
            impulse_48["capb"]["pre_echo_mean_square"],
            impulse_44["capb"]["pre_echo_mean_square"],
        )
    )
    for label, report in (("44k1", gate_44k1), ("48k", gate_48k)):
        row = _gate_row(report, "G5_gain", "impulse_train_10ms")
        checks.append(
            _numeric_check(
                f"{label}_impulse_train_gain_error_db",
                float(row["value"]),
                float(row["threshold"]),
            )
        )
    return {
        "all_passed": all(bool(check["passed"]) for check in checks),
        "distortion_tolerance_db": distortion_tolerance_db,
        "transient_position_tolerance": transient_position_tolerance,
        "focused_gentle_fraction_gap": fraction_gap,
        "checks": checks,
    }


def _gentle_fraction(family_report: dict[str, Any]) -> float:
    """Return the checkpoint's documented gentle share of impulse routing."""
    prior = family_report.get("routing_prior")
    if not isinstance(prior, dict):
        return 0.0
    fraction = float(prior.get("focused_gentle_fraction", 0.0))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("focused_gentle_fraction must lie in [0, 1].")
    return fraction


def _require_strict_precision(report: dict[str, Any], label: str) -> None:
    execution = report.get("execution")
    if not isinstance(execution, dict):
        raise ValueError(f"{label} report is missing execution metadata.")
    if execution.get("precision_mode") != "strict_fp32":
        raise ValueError(f"{label} report must use strict_fp32.")


def _gate_row(report: dict[str, Any], gate_id: str, probe_id: str) -> dict[str, Any]:
    for gate in report.get("gates", []):
        if gate.get("gate_id") != gate_id:
            continue
        for row in gate.get("rows", []):
            if row.get("probe_id") == probe_id:
                if "threshold" not in row:
                    raise ValueError(f"{gate_id}/{probe_id} row lacks a threshold.")
                return dict(row)
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
