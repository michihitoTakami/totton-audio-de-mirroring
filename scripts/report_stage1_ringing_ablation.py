"""Generate an ablation report for Stage 1 ringing auxiliary losses."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class EvaluationSnapshot:
    """Key Stage 1 evaluation metrics used in Issue #81 ablation.

    Args:
        symmetry_reduction_ratio: Mirror suppression ratio from mirror metrics.
        hb_energy_cap_violation_rate: Energy cap violation rate.
        lb_phase_error_deg: Low-band phase error in degrees.
        lb_group_delay_error_samples: Low-band group-delay error in samples.
        mean_ringing_ratio_delta: Mean after-before ringing ratio delta.
        mean_overshoot_abs_delta: Mean overshoot increase.
        mean_plateau_ripple_rms_ratio: Mean plateau RMS ripple ratio.
        mean_plateau_ripple_p2p_ratio: Mean plateau P2P ripple ratio.

    Physical Basis:
        These metrics jointly verify mirror suppression, low-band identity,
        and no-ringing-regression constraints for Stage 1 objectives.
    """

    symmetry_reduction_ratio: float
    hb_energy_cap_violation_rate: float
    lb_phase_error_deg: float
    lb_group_delay_error_samples: float
    mean_ringing_ratio_delta: float
    mean_overshoot_abs_delta: float
    mean_plateau_ripple_rms_ratio: float
    mean_plateau_ripple_p2p_ratio: float


@dataclass(frozen=True)
class LossContributionSnapshot:
    """Loss contribution snapshot from checkpoint history.

    Args:
        contrib_mask: Weighted contribution ratio for mask loss.
        contrib_stft: Weighted contribution ratio for STFT loss.
        contrib_preserve: Weighted contribution ratio for preserve loss.
        contrib_energy: Weighted contribution ratio for energy loss.
        contrib_edge: Weighted contribution ratio for edge ringing loss.
        contrib_step: Weighted contribution ratio for step ringing loss.

    Physical Basis:
        Contribution ratios make it explicit whether ringing auxiliaries remain
        secondary while mirror suppression terms stay dominant.
    """

    contrib_mask: float
    contrib_stft: float
    contrib_preserve: float
    contrib_energy: float
    contrib_edge: float
    contrib_step: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Issue #81 ablation reporting.

    Physical Basis:
        Comparing two fixed evaluation snapshots provides reproducible evidence
        of ringing loss impact without changing metric definitions.
    """
    parser = argparse.ArgumentParser(description="Report Stage1 ringing ablation")
    parser.add_argument("--baseline-eval-json", type=Path, required=True)
    parser.add_argument("--ringing-eval-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--ringing-checkpoint", type=Path, default=None)
    parser.add_argument("--max-mirror-drop", type=float, default=0.02)
    parser.add_argument("--max-lb-phase-error-deg", type=float, default=15.0)
    parser.add_argument("--max-lb-group-delay-error-samples", type=float, default=600.0)
    return parser.parse_args()


def main() -> None:
    """Generate markdown report for Stage 1 ringing-loss ablation.

    Raises:
        FileNotFoundError: If required files are missing.
        RuntimeError: If input JSON is malformed or report writing fails.

    Physical Basis:
        A fixed markdown report captures whether ringing-focused training
        improves edge behavior without violating mirror and low-band constraints.
    """
    args = parse_args()
    baseline_eval_payload = _load_json(args.baseline_eval_json)
    ringing_eval_payload = _load_json(args.ringing_eval_json)

    baseline = _extract_evaluation_snapshot(baseline_eval_payload)
    ringing = _extract_evaluation_snapshot(ringing_eval_payload)

    baseline_contrib = _extract_loss_contributions(args.baseline_checkpoint)
    ringing_contrib = _extract_loss_contributions(args.ringing_checkpoint)

    report_text = _render_report(
        baseline=baseline,
        ringing=ringing,
        baseline_contrib=baseline_contrib,
        ringing_contrib=ringing_contrib,
        max_mirror_drop=float(args.max_mirror_drop),
        max_lb_phase_error_deg=float(args.max_lb_phase_error_deg),
        max_lb_group_delay_error_samples=float(args.max_lb_group_delay_error_samples),
    )

    try:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report_text, encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to write report markdown: {exc}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON payload from disk."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON root must be object: {path}")
    return data


def _extract_evaluation_snapshot(payload: dict[str, Any]) -> EvaluationSnapshot:
    """Extract evaluation summary fields from evaluate_stage1 payload."""
    summary = _require_mapping(payload, "summary")
    mirror_metrics = _require_mapping(payload, "mirror_metrics")
    mirror_summary = _require_mapping(mirror_metrics, "summary")
    ringing_metrics = _require_mapping(payload, "ringing_metrics")
    ringing_summary = _require_mapping(ringing_metrics, "summary")
    return EvaluationSnapshot(
        symmetry_reduction_ratio=_require_float(
            mirror_summary, "symmetry_reduction_ratio"
        ),
        hb_energy_cap_violation_rate=_require_float(
            summary, "hb_energy_cap_violation_rate"
        ),
        lb_phase_error_deg=_require_float(summary, "lb_phase_error_deg"),
        lb_group_delay_error_samples=_require_float(
            summary, "lb_group_delay_error_samples"
        ),
        mean_ringing_ratio_delta=_require_float(
            ringing_summary, "mean_ringing_ratio_delta"
        ),
        mean_overshoot_abs_delta=_require_float(
            ringing_summary, "mean_overshoot_abs_delta"
        ),
        mean_plateau_ripple_rms_ratio=_require_float(
            ringing_summary, "mean_plateau_ripple_rms_ratio"
        ),
        mean_plateau_ripple_p2p_ratio=_require_float(
            ringing_summary, "mean_plateau_ripple_p2p_ratio"
        ),
    )


def _extract_loss_contributions(
    checkpoint_path: Path | None,
) -> LossContributionSnapshot | None:
    """Extract final-epoch contribution ratios from a Stage 1 checkpoint."""
    if checkpoint_path is None:
        return None
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {checkpoint_path}: {exc}"
        ) from exc

    train_history = checkpoint.get("train_history")
    if not isinstance(train_history, list) or len(train_history) == 0:
        raise RuntimeError(f"Checkpoint missing train_history: {checkpoint_path}")
    last_metrics = train_history[-1]
    if not isinstance(last_metrics, dict):
        raise RuntimeError(f"Malformed train_history entry: {checkpoint_path}")

    return LossContributionSnapshot(
        contrib_mask=float(last_metrics.get("contrib_mask", 0.0)),
        contrib_stft=float(last_metrics.get("contrib_stft", 0.0)),
        contrib_preserve=float(last_metrics.get("contrib_preserve", 0.0)),
        contrib_energy=float(last_metrics.get("contrib_energy", 0.0)),
        contrib_edge=float(last_metrics.get("contrib_edge", 0.0)),
        contrib_step=float(last_metrics.get("contrib_step", 0.0)),
    )


def _render_report(
    *,
    baseline: EvaluationSnapshot,
    ringing: EvaluationSnapshot,
    baseline_contrib: LossContributionSnapshot | None,
    ringing_contrib: LossContributionSnapshot | None,
    max_mirror_drop: float,
    max_lb_phase_error_deg: float,
    max_lb_group_delay_error_samples: float,
) -> str:
    """Render markdown report text."""
    mirror_ok = ringing.symmetry_reduction_ratio >= (
        baseline.symmetry_reduction_ratio - max_mirror_drop
    )
    ringing_ok = ringing.mean_ringing_ratio_delta <= baseline.mean_ringing_ratio_delta
    lowband_ok = (
        ringing.lb_phase_error_deg <= max_lb_phase_error_deg
        and ringing.lb_group_delay_error_samples <= max_lb_group_delay_error_samples
    )
    energy_ok = ringing.hb_energy_cap_violation_rate == 0.0

    lines: list[str] = []
    lines.append("# Issue #81 Stage1 Ringing-Loss Ablation Report")
    lines.append("")
    lines.append("## Verdict")
    lines.append(
        f"- mirror maintained: {'PASS' if mirror_ok else 'FAIL'} "
        f"(baseline={baseline.symmetry_reduction_ratio:.6f}, "
        f"ringing={ringing.symmetry_reduction_ratio:.6f})"
    )
    lines.append(
        f"- ringing improved: {'PASS' if ringing_ok else 'FAIL'} "
        f"(baseline_delta={baseline.mean_ringing_ratio_delta:.6f}, "
        f"ringing_delta={ringing.mean_ringing_ratio_delta:.6f})"
    )
    lines.append(
        f"- low-band non-interference gate: {'PASS' if lowband_ok else 'FAIL'} "
        f"(phase={ringing.lb_phase_error_deg:.6f}, "
        f"group_delay={ringing.lb_group_delay_error_samples:.6f})"
    )
    lines.append(
        f"- energy cap non-violation: {'PASS' if energy_ok else 'FAIL'} "
        f"(violation_rate={ringing.hb_energy_cap_violation_rate:.6f})"
    )
    lines.append("")
    lines.append("## Metric Comparison")
    lines.append("| Metric | Baseline | Ringing-Loss | Delta (Ringing-Baseline) |")
    lines.append("|---|---:|---:|---:|")
    lines.extend(
        [
            _metric_row(
                "symmetry_reduction_ratio",
                baseline.symmetry_reduction_ratio,
                ringing.symmetry_reduction_ratio,
            ),
            _metric_row(
                "mean_ringing_ratio_delta",
                baseline.mean_ringing_ratio_delta,
                ringing.mean_ringing_ratio_delta,
            ),
            _metric_row(
                "mean_overshoot_abs_delta",
                baseline.mean_overshoot_abs_delta,
                ringing.mean_overshoot_abs_delta,
            ),
            _metric_row(
                "mean_plateau_ripple_rms_ratio",
                baseline.mean_plateau_ripple_rms_ratio,
                ringing.mean_plateau_ripple_rms_ratio,
            ),
            _metric_row(
                "mean_plateau_ripple_p2p_ratio",
                baseline.mean_plateau_ripple_p2p_ratio,
                ringing.mean_plateau_ripple_p2p_ratio,
            ),
            _metric_row(
                "lb_phase_error_deg",
                baseline.lb_phase_error_deg,
                ringing.lb_phase_error_deg,
            ),
            _metric_row(
                "lb_group_delay_error_samples",
                baseline.lb_group_delay_error_samples,
                ringing.lb_group_delay_error_samples,
            ),
            _metric_row(
                "hb_energy_cap_violation_rate",
                baseline.hb_energy_cap_violation_rate,
                ringing.hb_energy_cap_violation_rate,
            ),
        ]
    )
    if baseline_contrib is not None and ringing_contrib is not None:
        lines.append("")
        lines.append("## Loss Contribution Comparison")
        lines.append("| Contribution | Baseline | Ringing-Loss | Delta |")
        lines.append("|---|---:|---:|---:|")
        lines.extend(
            [
                _contrib_row(
                    "mask", baseline_contrib.contrib_mask, ringing_contrib.contrib_mask
                ),
                _contrib_row(
                    "stft", baseline_contrib.contrib_stft, ringing_contrib.contrib_stft
                ),
                _contrib_row(
                    "preserve",
                    baseline_contrib.contrib_preserve,
                    ringing_contrib.contrib_preserve,
                ),
                _contrib_row(
                    "energy",
                    baseline_contrib.contrib_energy,
                    ringing_contrib.contrib_energy,
                ),
                _contrib_row(
                    "edge", baseline_contrib.contrib_edge, ringing_contrib.contrib_edge
                ),
                _contrib_row(
                    "step", baseline_contrib.contrib_step, ringing_contrib.contrib_step
                ),
            ]
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "- Stage1 low-band identity is structurally guaranteed by band-split bypass; "
        "this report additionally checks phase/group-delay metrics for non-interference."
    )
    lines.append(
        "- Ringing-loss terms are auxiliary objectives. Mirror suppression and energy "
        "safety remain hard constraints and must not regress."
    )
    return "\n".join(lines) + "\n"


def _metric_row(name: str, baseline: float, ringing: float) -> str:
    delta = ringing - baseline
    return f"| {name} | {baseline:.6f} | {ringing:.6f} | {delta:.6f} |"


def _contrib_row(name: str, baseline: float, ringing: float) -> str:
    delta = ringing - baseline
    return f"| {name} | {baseline:.4f} | {ringing:.4f} | {delta:.4f} |"


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected mapping at key '{key}'.")
    return value


def _require_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise RuntimeError(f"Expected numeric value at key '{key}'.")
    return float(value)


if __name__ == "__main__":
    main()
