"""Generate a matched-condition comparison report for raw88 vs bessel Stage1 runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunBundle:
    """Loaded artifacts for one Stage1 run.

    Args:
        teacher_tag: Teacher label (`raw88` or `bessel`).
        run_dir: Stage1 run report directory.
        run_manifest: Parsed run manifest payload.
        selected_candidate: Selected candidate payload from selection report.

    Physical Basis:
        The selected candidate captures gate-compliant behavior for one
        teacher policy under a fixed data/seed/gate configuration.
    """

    teacher_tag: str
    run_dir: Path
    run_manifest: dict[str, Any]
    selected_candidate: dict[str, Any]


@dataclass(frozen=True)
class MetricRule:
    """Comparison rule for one metric entry.

    Args:
        name: Metric key used in output tables.
        better: Optimization direction (`higher` or `lower`).

    Physical Basis:
        Direction-aware rules avoid ambiguous winner decisions when metrics
        represent suppression gain versus error/violation magnitudes.
    """

    name: str
    better: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for raw88-vs-bessel comparison reporting.

    Physical Basis:
        The report must compare runs created under matched seeds and gate
        settings to isolate teacher policy impact.
    """
    parser = argparse.ArgumentParser(description="Report raw88 vs bessel Stage1 runs")
    parser.add_argument("--raw-run-dir", type=Path, required=True)
    parser.add_argument("--bessel-run-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--allow-unmatched", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate a markdown/CSV teacher comparison report.

    Raises:
        FileNotFoundError: If required artifacts are missing.
        RuntimeError: If payload shape or matched-condition checks fail.

    Physical Basis:
        Side-by-side summary of hard/mirror/ringing/IMD metrics helps detect
        policy regressions before promoting raw88 as production default.
    """
    args = parse_args()
    raw = _load_run_bundle(run_dir=args.raw_run_dir, expected_teacher="raw88")
    bessel = _load_run_bundle(run_dir=args.bessel_run_dir, expected_teacher="bessel")
    if not args.allow_unmatched:
        _validate_matched_conditions(raw=raw, bessel=bessel)

    rows = _build_metric_rows(raw=raw, bessel=bessel)
    markdown = _render_markdown(raw=raw, bessel=bessel, rows=rows)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")

    if args.output_csv is not None:
        _write_csv(rows=rows, path=args.output_csv)


def _load_run_bundle(*, run_dir: Path, expected_teacher: str) -> RunBundle:
    """Load run manifest and selected candidate payload from a run directory."""
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    manifest = _load_json(run_dir / "run_manifest.json")
    teacher_tag = str(manifest.get("teacher_tag", "")).strip()
    if teacher_tag != expected_teacher:
        raise RuntimeError(
            f"teacher_tag mismatch for {run_dir}: expected {expected_teacher!r}, got {teacher_tag!r}."
        )

    selection = _load_json(run_dir / "selected" / "selection_report.json")
    selected_checkpoint = str(selection.get("selected_checkpoint", "")).strip()
    candidates = selection.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("selection_report missing candidates list.")

    selected = _extract_selected_candidate(
        candidates=candidates,
        selected_checkpoint=selected_checkpoint,
    )
    return RunBundle(
        teacher_tag=teacher_tag,
        run_dir=run_dir,
        run_manifest=manifest,
        selected_candidate=selected,
    )


def _load_json(path: Path) -> dict[str, Any]:
    """Load object-shaped JSON payload."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be object: {path}")
    return payload


def _extract_selected_candidate(
    *, candidates: list[Any], selected_checkpoint: str
) -> dict[str, Any]:
    """Return selected candidate payload from candidate list."""
    if len(selected_checkpoint) == 0:
        raise RuntimeError("selection_report missing selected_checkpoint.")

    selected_name = Path(selected_checkpoint).name
    for item in candidates:
        if not isinstance(item, dict):
            continue
        checkpoint_path = str(item.get("checkpoint_path", ""))
        if checkpoint_path == selected_checkpoint:
            return dict(item)
        if Path(checkpoint_path).name == selected_name:
            return dict(item)
    raise RuntimeError("Selected checkpoint entry not found in candidates list.")


def _validate_matched_conditions(*, raw: RunBundle, bessel: RunBundle) -> None:
    """Validate both runs use identical non-teacher comparison conditions."""
    mismatches: list[str] = []

    raw_manifest = raw.run_manifest
    bessel_manifest = bessel.run_manifest

    _compare_field(
        mismatches,
        field_name="training_config.seed",
        left=_read_nested(raw_manifest, "training_config", "seed"),
        right=_read_nested(bessel_manifest, "training_config", "seed"),
    )
    _compare_field(
        mismatches,
        field_name="train_config_sha256",
        left=raw_manifest.get("train_config_sha256"),
        right=bessel_manifest.get("train_config_sha256"),
    )
    _compare_field(
        mismatches,
        field_name="gate_thresholds",
        left=raw_manifest.get("gate_thresholds"),
        right=bessel_manifest.get("gate_thresholds"),
    )
    _compare_field(
        mismatches,
        field_name="args.eval_input_dir",
        left=_read_nested(raw_manifest, "args", "eval_input_dir"),
        right=_read_nested(bessel_manifest, "args", "eval_input_dir"),
    )
    _compare_field(
        mismatches,
        field_name="args.imd_naive_dir",
        left=_read_nested(raw_manifest, "args", "imd_naive_dir"),
        right=_read_nested(bessel_manifest, "args", "imd_naive_dir"),
    )
    _compare_field(
        mismatches,
        field_name="args.eval_glob",
        left=_read_nested(raw_manifest, "args", "eval_glob"),
        right=_read_nested(bessel_manifest, "args", "eval_glob"),
    )

    if len(mismatches) > 0:
        mismatch_text = "\n".join(f"- {item}" for item in mismatches)
        raise RuntimeError(
            "Runs are not under matched conditions. Use --allow-unmatched to bypass.\n"
            f"{mismatch_text}"
        )


def _compare_field(
    mismatches: list[str],
    *,
    field_name: str,
    left: Any,
    right: Any,
) -> None:
    if left != right:
        mismatches.append(f"{field_name}: raw={left!r}, bessel={right!r}")


def _read_nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _build_metric_rows(*, raw: RunBundle, bessel: RunBundle) -> list[dict[str, Any]]:
    """Build per-metric comparison rows and winner labels."""
    rules = [
        MetricRule(name="symmetry_reduction_ratio", better="higher"),
        MetricRule(name="hb_energy_cap_violation_rate", better="lower"),
        MetricRule(name="lb_amplitude_error_db", better="lower"),
        MetricRule(name="lb_phase_error_deg", better="lower"),
        MetricRule(name="lb_group_delay_error_samples", better="lower"),
        MetricRule(name="mean_thdn_improvement_db", better="higher"),
        MetricRule(name="mean_plateau_ripple_rms_ratio", better="lower"),
        MetricRule(name="mean_ringing_ratio_delta", better="lower"),
    ]

    raw_metrics = _extract_metric_map(raw.selected_candidate)
    bessel_metrics = _extract_metric_map(bessel.selected_candidate)

    rows: list[dict[str, Any]] = []
    for rule in rules:
        raw_value = float(raw_metrics[rule.name])
        bessel_value = float(bessel_metrics[rule.name])
        winner = _winner(rule=rule, raw_value=raw_value, bessel_value=bessel_value)
        rows.append(
            {
                "metric": rule.name,
                "better": rule.better,
                "raw88": raw_value,
                "bessel": bessel_value,
                "delta_raw_minus_bessel": raw_value - bessel_value,
                "winner": winner,
            }
        )
    return rows


def _extract_metric_map(candidate: dict[str, Any]) -> dict[str, float]:
    """Extract normalized metric map from candidate payload."""
    hard = _require_mapping(candidate, "hard_summary")
    mirror = _require_mapping(candidate, "mirror_summary")
    imd = _require_mapping(candidate, "imd_summary")
    ringing = _require_mapping(candidate, "ringing_summary")
    return {
        "symmetry_reduction_ratio": _require_float(mirror, "symmetry_reduction_ratio"),
        "hb_energy_cap_violation_rate": _require_float(
            hard, "hb_energy_cap_violation_rate"
        ),
        "lb_amplitude_error_db": _require_float(hard, "lb_amplitude_error_db"),
        "lb_phase_error_deg": _require_float(hard, "lb_phase_error_deg"),
        "lb_group_delay_error_samples": _require_float(
            hard, "lb_group_delay_error_samples"
        ),
        "mean_thdn_improvement_db": _require_float(imd, "mean_thdn_improvement_db"),
        "mean_plateau_ripple_rms_ratio": _require_float(
            ringing, "mean_plateau_ripple_rms_ratio"
        ),
        "mean_ringing_ratio_delta": _require_float(ringing, "mean_ringing_ratio_delta"),
    }


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"Missing object field: {key}")
    return value


def _require_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise RuntimeError(f"Missing numeric field: {key}")
    return float(value)


def _winner(*, rule: MetricRule, raw_value: float, bessel_value: float) -> str:
    if raw_value == bessel_value:
        return "tie"
    if rule.better == "higher":
        return "raw88" if raw_value > bessel_value else "bessel"
    if rule.better == "lower":
        return "raw88" if raw_value < bessel_value else "bessel"
    raise RuntimeError(f"Unsupported rule direction: {rule.better!r}")


def _render_markdown(
    *, raw: RunBundle, bessel: RunBundle, rows: list[dict[str, Any]]
) -> str:
    """Render markdown report body."""
    raw_wins = sum(1 for row in rows if row["winner"] == "raw88")
    bessel_wins = sum(1 for row in rows if row["winner"] == "bessel")
    ties = len(rows) - raw_wins - bessel_wins

    lines = [
        "# Stage1 Raw88 vs Bessel Comparison",
        "",
        "## Run Metadata",
        f"- raw88 run_id: {raw.run_manifest.get('run_id', 'unknown')}",
        f"- bessel run_id: {bessel.run_manifest.get('run_id', 'unknown')}",
        f"- raw88 run_dir: {raw.run_dir}",
        f"- bessel run_dir: {bessel.run_dir}",
        f"- seed(raw): {_read_nested(raw.run_manifest, 'training_config', 'seed')}",
        f"- seed(bessel): {_read_nested(bessel.run_manifest, 'training_config', 'seed')}",
        "",
        "## Win/Loss Summary",
        f"- raw88 wins: {raw_wins}",
        f"- bessel wins: {bessel_wins}",
        f"- ties: {ties}",
        "",
        "## Metric Table",
        "| Metric | Better | raw88 | bessel | raw88-bessel | Winner |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {better} | {raw88:.6f} | {bessel:.6f} | {delta_raw_minus_bessel:.6f} | {winner} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_csv(*, rows: list[dict[str, Any]], path: Path) -> None:
    """Write metric comparison rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "better",
        "raw88",
        "bessel",
        "delta_raw_minus_bessel",
        "winner",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
