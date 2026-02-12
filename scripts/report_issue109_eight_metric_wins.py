"""Generate Issue #109 win/loss tables for 8 microstructure metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

from totton_audio_de_mirroring.evaluation.lb_preservation import (
    evaluate_lowband_preservation,
)


@dataclass(frozen=True)
class MetricSpec:
    """Metadata and extraction rule for one metric.

    Args:
        key: Stable metric identifier for CSV/JSON outputs.
        label: Human-readable label for markdown reports.
        better: Optimization direction (`high`, `low`, or `abs_low`).
        source: Metric source (`microstructure` or `lowband`).
        json_path: Nested key path inside one microstructure metric JSON.

    Physical Basis:
        Winner decisions must be direction-aware because microstructure quality
        uses mixed objectives: correlation is maximized while distance/error
        terms are minimized.
    """

    key: str
    label: str
    better: str
    source: str
    json_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class MethodSignalMetric:
    """One metric value for one method and one signal.

    Args:
        method: Method name such as `distillation_nn`.
        signal_id: Canonical signal ID such as `thd_1khz`.
        metric_key: Metric identifier.
        value: Numeric metric value.

    Physical Basis:
        Per-file values preserve outlier behavior that can be hidden by only
        looking at aggregate means.
    """

    method: str
    signal_id: str
    metric_key: str
    value: float


DEFAULT_METHODS = [
    "bessel_iir",
    "bessel_fir",
    "fir_10k",
    "baseline_nn",
    "distillation_nn",
]

METRIC_SPECS = [
    MetricSpec(
        key="mps_corr",
        label="MPS Corr",
        better="high",
        source="microstructure",
        json_path=("metrics", "ch0", "mps", "mps_correlation"),
    ),
    MetricSpec(
        key="mps_dist",
        label="MPS Dist",
        better="low",
        source="microstructure",
        json_path=("metrics", "ch0", "mps", "mps_distance"),
    ),
    MetricSpec(
        key="tfs_corr",
        label="TFS Corr",
        better="high",
        source="microstructure",
        json_path=("metrics", "ch0", "tfs", "mean_correlation"),
    ),
    MetricSpec(
        key="attack_p95_abs_ms",
        label="Attack P95 (ms)",
        better="abs_low",
        source="microstructure",
        json_path=("metrics", "ch0", "transient", "attack_time_delta_p95_ms"),
    ),
    MetricSpec(
        key="bass_cycle_corr",
        label="Bass Cycle Corr",
        better="high",
        source="microstructure",
        json_path=("metrics", "ch0", "bass", "cycle_shape_corr_mean"),
    ),
    MetricSpec(
        key="lowband_wave_error_db",
        label="Lowband Wave Error (dB)",
        better="low",
        source="lowband",
    ),
    MetricSpec(
        key="lowband_phase_error_deg",
        label="Lowband Phase Error (deg)",
        better="low",
        source="lowband",
    ),
    MetricSpec(
        key="lowband_group_delay_error_ms",
        label="Lowband Group Delay Error (ms)",
        better="low",
        source="lowband",
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Physical Basis:
        A deterministic command interface ensures reproducible comparisons
        across teacher policies and implementation methods.
    """

    parser = argparse.ArgumentParser(
        description="Generate Issue #109 8-metric win/loss report."
    )
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--target-root", type=Path, default=None)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_METHODS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/issue109/eight_metrics"),
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="win_table",
    )
    return parser.parse_args()


def main() -> None:
    """Run Issue #109 reporting pipeline.

    Raises:
        FileNotFoundError: If required input directories are missing.
        RuntimeError: If no metric values are available.

    Physical Basis:
        The report separates per-file and aggregate winners to ensure
        method selection reflects both consistency and overall trend.
    """

    args = parse_args()
    _validate_inputs(metrics_root=args.metrics_root, methods=args.methods)
    method_names = list(dict.fromkeys(args.methods))
    all_values = _collect_metric_values(
        metrics_root=args.metrics_root,
        audio_root=args.audio_root,
        target_root=args.target_root,
        methods=method_names,
    )
    if len(all_values) == 0:
        raise RuntimeError("No metric values were collected.")

    aggregate_rows = _build_aggregate_rows(all_values=all_values, methods=method_names)
    per_file_rows = _build_per_file_rows(all_values=all_values, methods=method_names)
    point_rows = _build_point_rows(aggregate_rows=aggregate_rows, methods=method_names)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_md = output_dir / f"{args.report_name}.md"
    output_aggregate_csv = output_dir / f"{args.report_name}_aggregate.csv"
    output_per_file_csv = output_dir / f"{args.report_name}_per_file.csv"
    output_json = output_dir / f"{args.report_name}.json"

    markdown = _render_markdown(
        method_names=method_names,
        aggregate_rows=aggregate_rows,
        per_file_rows=per_file_rows,
        point_rows=point_rows,
        metrics_root=args.metrics_root,
        audio_root=args.audio_root,
        target_root=args.target_root,
    )
    output_md.write_text(markdown, encoding="utf-8")
    _write_csv(path=output_aggregate_csv, rows=aggregate_rows)
    _write_csv(path=output_per_file_csv, rows=per_file_rows)
    payload = {
        "methods": method_names,
        "metrics_root": str(args.metrics_root),
        "audio_root": str(args.audio_root) if args.audio_root else None,
        "target_root": str(args.target_root) if args.target_root else None,
        "aggregate_rows": aggregate_rows,
        "per_file_rows": per_file_rows,
        "points": point_rows,
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")


def _validate_inputs(*, metrics_root: Path, methods: list[str]) -> None:
    """Validate required CLI inputs.

    Physical Basis:
        Explicit input validation prevents accidental method omission and
        keeps comparison conditions reproducible.
    """

    if not metrics_root.exists() or not metrics_root.is_dir():
        raise FileNotFoundError(f"metrics_root not found: {metrics_root}")
    if len(methods) == 0:
        raise ValueError("methods must not be empty.")


def _collect_metric_values(
    *,
    metrics_root: Path,
    audio_root: Path | None,
    target_root: Path | None,
    methods: list[str],
) -> list[MethodSignalMetric]:
    """Collect all metric values from JSON and optional lowband waveform eval.

    Physical Basis:
        Microstructure metrics capture modulation/phase/transient behavior,
        while lowband metrics enforce the 0-20kHz preservation requirement.
    """

    values: list[MethodSignalMetric] = []
    values.extend(
        _collect_microstructure_values(metrics_root=metrics_root, methods=methods)
    )
    if audio_root is not None and target_root is not None:
        values.extend(
            _collect_lowband_values(
                audio_root=audio_root,
                target_root=target_root,
                methods=methods,
            )
        )
    return values


def _collect_microstructure_values(
    *,
    metrics_root: Path,
    methods: list[str],
) -> list[MethodSignalMetric]:
    """Collect MPS/TFS/Attack/Bass metric values from report JSON files.

    Physical Basis:
        These metrics quantify texture, fine-structure phase, transient edge
        timing, and low-frequency cycle fidelity from shared stimuli.
    """

    values: list[MethodSignalMetric] = []
    for method in methods:
        method_dir = metrics_root / method
        if not method_dir.exists() or not method_dir.is_dir():
            continue
        for json_path in sorted(method_dir.glob("*.json")):
            payload = _load_json(json_path)
            signal_id = _canonical_signal_id(json_path.stem)
            for spec in METRIC_SPECS:
                if spec.source != "microstructure":
                    continue
                metric_value = _read_metric_value(payload=payload, spec=spec)
                if metric_value is None:
                    continue
                values.append(
                    MethodSignalMetric(
                        method=method,
                        signal_id=signal_id,
                        metric_key=spec.key,
                        value=metric_value,
                    )
                )
    return values


def _collect_lowband_values(
    *,
    audio_root: Path,
    target_root: Path,
    methods: list[str],
) -> list[MethodSignalMetric]:
    """Collect lowband metrics by comparing method WAV to target WAV.

    Physical Basis:
        Hard Requirement #1 enforces preservation of waveform/phase/group-delay
        in the low-band; this must be evaluated against target 88.2kHz signals.
    """

    values: list[MethodSignalMetric] = []
    target_map = _build_target_map(target_root=target_root)
    for method in methods:
        method_dir = audio_root / method
        if not method_dir.exists() or not method_dir.is_dir():
            continue
        for wav_path in sorted(method_dir.glob("*.wav")):
            signal_id = _canonical_signal_id(wav_path.stem)
            target_path = target_map.get(signal_id)
            if target_path is None:
                continue
            lowband = _evaluate_lowband_pair(
                reference_path=target_path, output_path=wav_path
            )
            values.extend(
                [
                    MethodSignalMetric(
                        method=method,
                        signal_id=signal_id,
                        metric_key="lowband_wave_error_db",
                        value=lowband["lowband_wave_error_db"],
                    ),
                    MethodSignalMetric(
                        method=method,
                        signal_id=signal_id,
                        metric_key="lowband_phase_error_deg",
                        value=lowband["lowband_phase_error_deg"],
                    ),
                    MethodSignalMetric(
                        method=method,
                        signal_id=signal_id,
                        metric_key="lowband_group_delay_error_ms",
                        value=lowband["lowband_group_delay_error_ms"],
                    ),
                ]
            )
    return values


def _load_json(path: Path) -> dict[str, Any]:
    """Load object-shaped JSON payload from file.

    Physical Basis:
        Deterministic parsing avoids hidden schema drift when aggregating
        evaluation artifacts from multiple methods.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be object: {path}")
    return payload


def _read_metric_value(*, payload: dict[str, Any], spec: MetricSpec) -> float | None:
    """Read one scalar metric value from a report payload.

    Physical Basis:
        Scalar extraction isolates comparable quality indicators across methods
        without depending on optional nested plots or diagnostics.
    """

    value: Any = payload
    for key in spec.json_path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if not isinstance(value, int | float):
        return None
    numeric = float(value)
    if spec.better == "abs_low":
        return float(abs(numeric))
    return numeric


def _canonical_signal_id(stem: str) -> str:
    """Normalize filename stem to signal ID for cross-artifact joins.

    Physical Basis:
        Stable signal IDs are required to compare per-file winners across
        metrics generated by different tools and naming conventions.
    """

    cleaned = stem
    cleaned = re.sub(r"_\d{5,6}_full$", "", cleaned)
    cleaned = re.sub(r"_\d{5,6}_hz_\d+bit_v\d+$", "", cleaned)
    return cleaned


def _build_target_map(*, target_root: Path) -> dict[str, Path]:
    """Build mapping from canonical signal ID to target WAV path.

    Physical Basis:
        A one-to-one signal map keeps lowband preservation comparisons aligned
        to the exact original 88.2kHz stimulus.
    """

    target_map: dict[str, Path] = {}
    for wav_path in sorted(target_root.glob("*.wav")):
        signal_id = _canonical_signal_id(wav_path.stem)
        if signal_id in target_map:
            existing = target_map[signal_id]
            existing_is_88k = "_88200_" in existing.stem
            incoming_is_88k = "_88200_" in wav_path.stem
            if existing_is_88k and not incoming_is_88k:
                continue
            if incoming_is_88k and not existing_is_88k:
                target_map[signal_id] = wav_path
            continue
        target_map[signal_id] = wav_path
    return target_map


def _evaluate_lowband_pair(
    *, reference_path: Path, output_path: Path
) -> dict[str, float]:
    """Evaluate lowband metrics between reference and output WAV files.

    Raises:
        RuntimeError: If sample rates are mismatched.

    Physical Basis:
        Lowband identity must be checked in waveform, phase, and group delay
        to detect audible-band regressions.
    """

    ref_rate, ref_signal = wavfile.read(reference_path)
    out_rate, out_signal = wavfile.read(output_path)
    if int(ref_rate) != int(out_rate):
        raise RuntimeError(
            f"Sample-rate mismatch: {reference_path}={ref_rate}, {output_path}={out_rate}"
        )
    ref_float = _to_mono_float(ref_signal)
    out_float = _to_mono_float(out_signal)
    length = min(ref_float.shape[0], out_float.shape[0])
    if length <= 0:
        raise RuntimeError(f"Empty signals: {reference_path}, {output_path}")
    metrics = evaluate_lowband_preservation(
        input_signal=ref_float[:length],
        output_signal=out_float[:length],
        sample_rate=int(ref_rate),
    )
    return {
        "lowband_wave_error_db": float(metrics.waveform_error_db),
        "lowband_phase_error_deg": float(metrics.phase_error_deg),
        "lowband_group_delay_error_ms": float(metrics.group_delay_error_ms),
    }


def _to_mono_float(signal: np.ndarray) -> np.ndarray:
    """Convert input waveform to mono float64 in [-1, 1] scale.

    Raises:
        ValueError: If waveform rank is unsupported.

    Physical Basis:
        Lowband preservation metrics expect time-domain arrays with consistent
        normalization and channel handling.
    """

    if signal.ndim == 1:
        mono = signal
    elif signal.ndim == 2:
        mono = np.mean(signal, axis=1)
    else:
        raise ValueError(f"Unsupported WAV rank: {signal.ndim}")
    if np.issubdtype(mono.dtype, np.integer):
        info = np.iinfo(mono.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        return np.asarray(mono, dtype=np.float64) / scale
    return np.asarray(mono, dtype=np.float64)


def _build_aggregate_rows(
    *,
    all_values: list[MethodSignalMetric],
    methods: list[str],
) -> list[dict[str, Any]]:
    """Build aggregate metric rows with method means and winners.

    Physical Basis:
        Aggregate winners summarize method behavior over the full stimulus set
        for objective comparison and decision making.
    """

    grouped = _group_values_by_metric_and_method(all_values=all_values)
    rows: list[dict[str, Any]] = []
    for spec in METRIC_SPECS:
        per_method = grouped.get(spec.key, {})
        method_means: dict[str, float | None] = {}
        for method in methods:
            values = per_method.get(method, [])
            method_means[method] = (
                float(np.mean(np.asarray(values, dtype=np.float64)))
                if len(values) > 0
                else None
            )
        winner = _pick_winner(spec=spec, method_values=method_means)
        row: dict[str, Any] = {
            "metric_key": spec.key,
            "metric_label": spec.label,
            "better": spec.better,
            "winner": winner,
            "coverage_methods": sum(
                1 for method in methods if method_means[method] is not None
            ),
        }
        for method in methods:
            row[method] = method_means[method]
        rows.append(row)
    return rows


def _build_per_file_rows(
    *,
    all_values: list[MethodSignalMetric],
    methods: list[str],
) -> list[dict[str, Any]]:
    """Build per-file winner rows for each metric.

    Physical Basis:
        Per-file winners expose localized degradations that can be masked by
        aggregate means, improving reliability of method selection.
    """

    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for item in all_values:
        if item.metric_key not in grouped:
            grouped[item.metric_key] = {}
        if item.signal_id not in grouped[item.metric_key]:
            grouped[item.metric_key][item.signal_id] = {}
        grouped[item.metric_key][item.signal_id][item.method] = item.value

    rows: list[dict[str, Any]] = []
    for spec in METRIC_SPECS:
        signal_map = grouped.get(spec.key, {})
        for signal_id in sorted(signal_map.keys()):
            values_by_method = signal_map[signal_id]
            method_values = {method: values_by_method.get(method) for method in methods}
            winner = _pick_winner(spec=spec, method_values=method_values)
            row: dict[str, Any] = {
                "metric_key": spec.key,
                "metric_label": spec.label,
                "signal_id": signal_id,
                "winner": winner,
                "coverage_methods": sum(
                    1 for method in methods if method_values.get(method) is not None
                ),
            }
            for method in methods:
                row[method] = method_values.get(method)
            rows.append(row)
    return rows


def _group_values_by_metric_and_method(
    *,
    all_values: list[MethodSignalMetric],
) -> dict[str, dict[str, list[float]]]:
    """Group metric values by metric key and method.

    Physical Basis:
        Grouped structure enables deterministic aggregation and winner
        calculations across comparable samples.
    """

    grouped: dict[str, dict[str, list[float]]] = {}
    for value in all_values:
        if value.metric_key not in grouped:
            grouped[value.metric_key] = {}
        if value.method not in grouped[value.metric_key]:
            grouped[value.metric_key][value.method] = []
        grouped[value.metric_key][value.method].append(value.value)
    return grouped


def _pick_winner(*, spec: MetricSpec, method_values: dict[str, float | None]) -> str:
    """Pick winner or tie from method values using metric direction.

    Physical Basis:
        Direction-specific optimization avoids invalid comparisons between
        correlation-style and error-style quality metrics.
    """

    available = [
        (method, value) for method, value in method_values.items() if value is not None
    ]
    if len(available) == 0:
        return "n/a"
    if spec.better == "high":
        best_value = max(value for _, value in available)
        winners = [method for method, value in available if value == best_value]
    elif spec.better in {"low", "abs_low"}:
        best_value = min(value for _, value in available)
        winners = [method for method, value in available if value == best_value]
    else:
        raise RuntimeError(f"Unsupported metric direction: {spec.better}")
    if len(winners) == 1:
        return winners[0]
    return "tie:" + ",".join(sorted(winners))


def _build_point_rows(
    *,
    aggregate_rows: list[dict[str, Any]],
    methods: list[str],
) -> list[dict[str, float]]:
    """Build aggregate point table from metric winners.

    Physical Basis:
        Point aggregation provides a compact overall score while preserving
        equal credit for ties.
    """

    points = dict.fromkeys(methods, 0.0)
    for row in aggregate_rows:
        winner = str(row.get("winner", "n/a"))
        if winner == "n/a":
            continue
        if winner.startswith("tie:"):
            tie_methods = [name for name in winner[4:].split(",") if name in points]
            if len(tie_methods) == 0:
                continue
            share = 1.0 / float(len(tie_methods))
            for method in tie_methods:
                points[method] += share
            continue
        if winner in points:
            points[winner] += 1.0
    return [
        {"method": method, "points": points[method]}
        for method in sorted(methods, key=lambda name: (-points[name], name))
    ]


def _write_csv(*, path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dict rows to CSV.

    Physical Basis:
        Flat CSV outputs improve auditability and support downstream analysis
        in notebooks or spreadsheets.
    """

    if len(rows) == 0:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown(
    *,
    method_names: list[str],
    aggregate_rows: list[dict[str, Any]],
    per_file_rows: list[dict[str, Any]],
    point_rows: list[dict[str, float]],
    metrics_root: Path,
    audio_root: Path | None,
    target_root: Path | None,
) -> str:
    """Render markdown report text.

    Physical Basis:
        Human-readable reports are required for reproducible experiment review
        and explicit winner tracking per metric.
    """

    lines: list[str] = []
    lines.append("# Issue #109 8指標 勝敗表")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- metrics_root: `{metrics_root}`")
    lines.append(f"- audio_root: `{audio_root}`")
    lines.append(f"- target_root: `{target_root}`")
    lines.append(f"- methods: {', '.join(method_names)}")
    lines.append("")
    lines.append("## Aggregate Winners")
    lines.append("")
    header = ["Metric", "Better"] + method_names + ["Winner", "Coverage"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in aggregate_rows:
        cells = [
            str(row["metric_label"]),
            str(row["better"]),
        ]
        for method in method_names:
            value = row.get(method)
            cells.append("n/a" if value is None else f"{float(value):.6f}")
        cells.append(str(row["winner"]))
        cells.append(str(row["coverage_methods"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Points")
    lines.append("")
    lines.append("| method | points |")
    lines.append("|---|---:|")
    for item in point_rows:
        lines.append(f"| {item['method']} | {item['points']:.2f} |")
    lines.append("")
    lines.append("## Per-file Winner Counts")
    lines.append("")
    winner_counts = _count_per_file_wins(
        per_file_rows=per_file_rows, methods=method_names
    )
    lines.append("| Metric | " + " | ".join(method_names) + " | tie | n/a |")
    lines.append("|" + "|".join(["---"] * (len(method_names) + 3)) + "|")
    for metric_key, per_method in winner_counts.items():
        label = next(
            (spec.label for spec in METRIC_SPECS if spec.key == metric_key),
            metric_key,
        )
        tie_count = int(per_method.get("tie", 0))
        na_count = int(per_method.get("n/a", 0))
        method_cells = [str(int(per_method.get(method, 0))) for method in method_names]
        lines.append(
            "| "
            + " | ".join([label, *method_cells, str(tie_count), str(na_count)])
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _count_per_file_wins(
    *,
    per_file_rows: list[dict[str, Any]],
    methods: list[str],
) -> dict[str, dict[str, int]]:
    """Count winners per metric over all signals.

    Physical Basis:
        Winner-frequency summaries quantify consistency of each method across
        the shared signal set.
    """

    counts: dict[str, dict[str, int]] = {}
    for row in per_file_rows:
        metric_key = str(row["metric_key"])
        if metric_key not in counts:
            counts[metric_key] = dict.fromkeys(methods, 0)
            counts[metric_key]["tie"] = 0
            counts[metric_key]["n/a"] = 0
        winner = str(row.get("winner", "n/a"))
        if winner == "n/a":
            counts[metric_key]["n/a"] += 1
        elif winner.startswith("tie:"):
            counts[metric_key]["tie"] += 1
        elif winner in counts[metric_key]:
            counts[metric_key][winner] += 1
    return counts


if __name__ == "__main__":
    main()
