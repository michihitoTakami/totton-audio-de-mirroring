"""CLI for Stage 1 hard metrics evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from totton_audio_de_mirroring.evaluation.imd_proxy import evaluate_imd_proxy
from totton_audio_de_mirroring.evaluation.metrics import (
    DatasetEvaluationResult,
    evaluate_dataset,
    sample_result_to_flat_dict,
)
from totton_audio_de_mirroring.evaluation.mirror_metrics import (
    MirrorDatasetEvaluationResult,
    evaluate_mirror_reduction_dataset,
    export_mirror_reduction_visualization,
    mirror_dataset_result_to_payload,
)
from totton_audio_de_mirroring.evaluation.time_domain_visualization import (
    RingingComparisonMetrics,
    compare_edge_aligned_ringing,
)


@dataclass(frozen=True)
class RingingDatasetSummary:
    """Dataset-level summary for edge-aligned ringing comparisons.

    Args:
        num_samples: Number of evaluated sample pairs.
        mean_plateau_ripple_rms_before: Mean RMS ripple before processing.
        mean_plateau_ripple_rms_after: Mean RMS ripple after processing.
        mean_plateau_ripple_rms_ratio: Mean after/before RMS ripple ratio.
        mean_plateau_ripple_p2p_before: Mean P2P ripple before processing.
        mean_plateau_ripple_p2p_after: Mean P2P ripple after processing.
        mean_plateau_ripple_p2p_ratio: Mean after/before P2P ripple ratio.
        mean_overshoot_abs_before: Mean absolute overshoot before processing.
        mean_overshoot_abs_after: Mean absolute overshoot after processing.
        mean_overshoot_abs_delta: Mean after-before overshoot increase.
        mean_ringing_ratio_before: Mean post/pre ringing ratio before processing.
        mean_ringing_ratio_after: Mean post/pre ringing ratio after processing.
        mean_ringing_ratio_delta: Mean after-before ringing-ratio increase.

    Physical Basis:
        Aggregating edge-aligned metrics across square probes quantifies
        whether a checkpoint preserves time response compared with baseline SRC.
    """

    num_samples: int
    mean_plateau_ripple_rms_before: float
    mean_plateau_ripple_rms_after: float
    mean_plateau_ripple_rms_ratio: float
    mean_plateau_ripple_p2p_before: float
    mean_plateau_ripple_p2p_after: float
    mean_plateau_ripple_p2p_ratio: float
    mean_overshoot_abs_before: float
    mean_overshoot_abs_after: float
    mean_overshoot_abs_delta: float
    mean_ringing_ratio_before: float
    mean_ringing_ratio_after: float
    mean_ringing_ratio_delta: float


@dataclass(frozen=True)
class Stage1GateConfig:
    """Threshold configuration for Stage1 acceptance gates.

    Args:
        mirror_target_reduction: Minimum mirror symmetry-reduction ratio.
        max_plateau_ripple_rms_ratio: Max allowed after/before ripple RMS ratio.
        max_plateau_ripple_p2p_ratio: Max allowed after/before ripple P2P ratio.
        max_overshoot_abs_increase: Max allowed overshoot absolute increase.
        require_nonpositive_ringing_ratio_delta: Require no ringing-ratio increase.

    Physical Basis:
        Stage1 acceptance combines mirror suppression, high-band safety,
        and ringing non-regression into explicit reproducible thresholds.
    """

    mirror_target_reduction: float
    max_plateau_ripple_rms_ratio: float
    max_plateau_ripple_p2p_ratio: float
    max_overshoot_abs_increase: float
    require_nonpositive_ringing_ratio_delta: bool
    max_lb_phase_error_deg: float
    max_lb_group_delay_error_samples: float
    max_lb_amplitude_error_db: float
    require_positive_thdn_improvement: bool


_STRICT_EXIT_ENERGY_CAP = 2
_STRICT_EXIT_MIRROR_REDUCTION = 3
_STRICT_EXIT_RINGING_REGRESSION = 4
_STRICT_EXIT_MULTIPLE_GATES = 5
_STRICT_EXIT_LOWBAND_PRESERVATION = 6
_STRICT_EXIT_IMD_PROXY = 7


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Stage 1 evaluation.

    Physical Basis:
        Fixed evaluation parameters make hard-metric comparisons
        reproducible across model checkpoints and datasets.
    """
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 hard metrics.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument(
        "--evaluation-target",
        type=str,
        default="input",
        choices=("input", "raw88.2", "bessel88.2"),
    )
    parser.add_argument("--glob", type=str, default="*.npy")
    parser.add_argument("--sample-rate", type=int, default=88_200)
    parser.add_argument("--cutoff-hz", type=float, default=20_000.0)
    parser.add_argument("--energy-cap", type=float, default=1.0e-3)
    parser.add_argument("--num-taps", type=int, default=1025)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument(
        "--mirror-band-hz",
        type=float,
        nargs=2,
        default=(20_000.0, 22_050.0),
        metavar=("LOWER", "UPPER"),
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--strict-energy-cap", action="store_true")
    parser.add_argument("--strict-mirror-reduction", action="store_true")
    parser.add_argument("--strict-ringing-regression", action="store_true")
    parser.add_argument("--strict-lowband-preservation", action="store_true")
    parser.add_argument("--strict-imd-proxy", action="store_true")
    parser.add_argument("--mirror-target-reduction", type=float, default=0.70)
    parser.add_argument("--max-lb-phase-error-deg", type=float, default=15.0)
    parser.add_argument("--max-lb-group-delay-error-samples", type=float, default=600.0)
    parser.add_argument("--max-lb-amplitude-error-db", type=float, default=-20.0)
    parser.add_argument("--max-plateau-ripple-rms-ratio", type=float, default=1.10)
    parser.add_argument("--max-plateau-ripple-p2p-ratio", type=float, default=1.10)
    parser.add_argument("--max-overshoot-abs-increase", type=float, default=5.0e-3)
    parser.add_argument("--allow-ringing-ratio-increase", action="store_true")
    parser.add_argument("--mirror-visual-dir", type=Path, default=None)
    parser.add_argument("--mirror-visual-limit", type=int, default=16)
    parser.add_argument("--ringing-json", type=Path, default=None)
    parser.add_argument("--ringing-csv", type=Path, default=None)
    parser.add_argument("--ringing-plateau-start-ms", type=float, default=0.1)
    parser.add_argument("--ringing-plateau-end-ms", type=float, default=0.8)
    parser.add_argument("--ringing-window-ms", type=float, default=0.8)
    parser.add_argument("--imd-naive-dir", type=Path, default=None)
    parser.add_argument("--allow-nonpositive-thdn-improvement", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run Stage 1 hard-metrics evaluation.

    Raises:
        FileNotFoundError: If paired input/output files are missing.
        RuntimeError: If output serialization fails.

    Physical Basis:
        Batch evaluation is needed to verify low-band identity and
        mirror suppression behavior under varied signal conditions.
    """
    args = parse_args()
    _validate_args(args)
    _prepare_integrated_report_outputs(args)
    pairs = _load_pairs(args.input_dir, args.output_dir, args.glob)
    eval_pairs = _resolve_evaluation_pairs(
        pairs=pairs,
        target_dir=args.target_dir,
        evaluation_target=args.evaluation_target,
    )
    gate_config = _build_stage1_gate_config(args)
    result = evaluate_dataset(
        samples=eval_pairs,
        sample_rate=args.sample_rate,
        cutoff_hz=args.cutoff_hz,
        energy_cap=args.energy_cap,
        num_taps=args.num_taps,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        mirror_band_hz=(args.mirror_band_hz[0], args.mirror_band_hz[1]),
    )
    mirror_result = evaluate_mirror_reduction_dataset(
        samples=pairs,
        sample_rate=args.sample_rate,
        mirror_band_hz=(args.mirror_band_hz[0], args.mirror_band_hz[1]),
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        target_reduction_ratio=args.mirror_target_reduction,
    )
    ringing_metrics = _evaluate_ringing_dataset(
        pairs=eval_pairs,
        sample_rate=args.sample_rate,
        plateau_start_ms=args.ringing_plateau_start_ms,
        plateau_end_ms=args.ringing_plateau_end_ms,
        ringing_window_ms=args.ringing_window_ms,
    )
    ringing_summary = _summarize_ringing_metrics(ringing_metrics)
    imd_summary = _evaluate_imd_dataset(
        naive_dir=args.imd_naive_dir,
        nmse_dir=args.output_dir,
        sample_rate=args.sample_rate,
    )
    gate_status = _evaluate_stage1_gates(
        result=result,
        mirror_result=mirror_result,
        ringing_summary=ringing_summary,
        imd_summary=imd_summary,
        gate_config=gate_config,
    )

    visual_exports: list[str] = []
    if args.mirror_visual_dir is not None:
        visual_exports = _export_mirror_visualizations(
            pairs=pairs,
            output_dir=args.mirror_visual_dir,
            sample_rate=args.sample_rate,
            mirror_band_hz=(args.mirror_band_hz[0], args.mirror_band_hz[1]),
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            max_exports=args.mirror_visual_limit,
        )

    if args.csv is not None:
        _write_csv(
            result,
            args.csv,
            ringing_metrics=ringing_metrics,
            gate_status=gate_status,
        )
    if args.ringing_csv is not None:
        _write_ringing_csv(ringing_metrics, args.ringing_csv)
    if args.json is not None:
        _write_json(
            result,
            args.json,
            mirror_result=mirror_result,
            visual_exports=visual_exports,
            ringing_metrics=ringing_metrics,
            ringing_summary=ringing_summary,
            imd_summary=imd_summary,
            gate_status=gate_status,
            evaluation_target=args.evaluation_target,
        )
    if args.ringing_json is not None:
        _write_ringing_json(
            ringing_metrics=ringing_metrics,
            ringing_summary=ringing_summary,
            path=args.ringing_json,
        )

    if args.print_json:
        print(
            json.dumps(
                _to_payload(
                    result,
                    mirror_result=mirror_result,
                    visual_exports=visual_exports,
                    ringing_metrics=ringing_metrics,
                    ringing_summary=ringing_summary,
                    imd_summary=imd_summary,
                    gate_status=gate_status,
                    evaluation_target=args.evaluation_target,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_summary(result, mirror_result, evaluation_target=args.evaluation_target)
        if len(visual_exports) > 0:
            print(f"Mirror visualizations exported: {len(visual_exports)}")
    if args.markdown is not None:
        _write_markdown_report(
            path=args.markdown,
            result=result,
            mirror_result=mirror_result,
            ringing_summary=ringing_summary,
            gate_status=gate_status,
            visual_exports=visual_exports,
            imd_summary=imd_summary,
            evaluation_target=args.evaluation_target,
        )

    _raise_for_strict_gate_failures(args=args, gate_status=gate_status)


def _load_pairs(
    input_dir: Path,
    output_dir: Path,
    pattern: str,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load paired input/output `.npy` signals by filename.

    Args:
        input_dir: Directory containing reference inputs.
        output_dir: Directory containing model outputs.
        pattern: Glob pattern used in `input_dir`.

    Returns:
        List of (sample_id, input_signal, output_signal).

    Raises:
        FileNotFoundError: If directories/files are missing.
        ValueError: If arrays are not valid 1D signal pairs.

    Physical Basis:
        Filename-based pairing enables deterministic evaluation over
        reproducible test sets without hidden sampling randomness.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not output_dir.exists() or not output_dir.is_dir():
        raise FileNotFoundError(f"output_dir not found: {output_dir}")

    input_paths = sorted(input_dir.glob(pattern))
    if len(input_paths) == 0:
        raise FileNotFoundError(
            f"No input files matched pattern '{pattern}' in {input_dir}."
        )

    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for input_path in input_paths:
        output_path = output_dir / input_path.name
        if not output_path.exists():
            raise FileNotFoundError(f"Missing output pair for {input_path.name}")

        try:
            input_signal = np.asarray(np.load(input_path), dtype=np.float64)
            output_signal = np.asarray(np.load(output_path), dtype=np.float64)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pair '{input_path.name}': {exc}"
            ) from exc

        sample_id = input_path.stem
        pairs.append((sample_id, input_signal, output_signal))

    return pairs


def _build_stage1_gate_config(args: argparse.Namespace) -> Stage1GateConfig:
    """Build immutable gate-threshold config from CLI args."""
    return Stage1GateConfig(
        mirror_target_reduction=float(args.mirror_target_reduction),
        max_plateau_ripple_rms_ratio=float(args.max_plateau_ripple_rms_ratio),
        max_plateau_ripple_p2p_ratio=float(args.max_plateau_ripple_p2p_ratio),
        max_overshoot_abs_increase=float(args.max_overshoot_abs_increase),
        require_nonpositive_ringing_ratio_delta=(
            not bool(args.allow_ringing_ratio_increase)
        ),
        max_lb_phase_error_deg=float(args.max_lb_phase_error_deg),
        max_lb_group_delay_error_samples=float(args.max_lb_group_delay_error_samples),
        max_lb_amplitude_error_db=float(args.max_lb_amplitude_error_db),
        require_positive_thdn_improvement=(
            not bool(args.allow_nonpositive_thdn_improvement)
        ),
    )


def _to_payload(
    result: DatasetEvaluationResult,
    mirror_result: MirrorDatasetEvaluationResult | None = None,
    visual_exports: list[str] | None = None,
    ringing_metrics: list[dict[str, Any]] | None = None,
    ringing_summary: RingingDatasetSummary | None = None,
    imd_summary: dict[str, Any] | None = None,
    gate_status: dict[str, Any] | None = None,
    evaluation_target: str = "input",
) -> dict[str, Any]:
    """Convert dataset evaluation to JSON-serializable payload.

    Physical Basis:
        Structured payloads preserve metric definitions for downstream
        plotting and regression checks.
    """
    payload: dict[str, Any] = {
        "evaluation_target": evaluation_target,
        "summary": asdict(result.mean_metrics)
        | {
            "hb_energy_cap_violation_rate": result.hb_energy_cap_violation_rate,
            "num_samples": len(result.samples),
        },
        "samples": [sample_result_to_flat_dict(sample) for sample in result.samples],
    }
    if mirror_result is not None:
        payload["mirror_metrics"] = mirror_dataset_result_to_payload(mirror_result)
    if visual_exports is not None:
        payload["mirror_visualizations"] = visual_exports
    if ringing_metrics is not None and ringing_summary is not None:
        payload["ringing_metrics"] = {
            "summary": asdict(ringing_summary),
            "samples": ringing_metrics,
        }
    if imd_summary is not None:
        payload["imd_proxy"] = imd_summary
    if gate_status is not None:
        payload["gates"] = gate_status
    return payload


def _write_json(
    result: DatasetEvaluationResult,
    path: Path,
    mirror_result: MirrorDatasetEvaluationResult | None = None,
    visual_exports: list[str] | None = None,
    ringing_metrics: list[dict[str, Any]] | None = None,
    ringing_summary: RingingDatasetSummary | None = None,
    imd_summary: dict[str, Any] | None = None,
    gate_status: dict[str, Any] | None = None,
    evaluation_target: str = "input",
) -> None:
    """Write evaluation payload to JSON file.

    Args:
        result: Dataset evaluation result.
        path: Output JSON path.

    Raises:
        RuntimeError: If write operation fails.

    Physical Basis:
        Persisted evaluation snapshots are required for reproducible
        model selection and regression tracking.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                _to_payload(
                    result,
                    mirror_result=mirror_result,
                    visual_exports=visual_exports,
                    ringing_metrics=ringing_metrics,
                    ringing_summary=ringing_summary,
                    imd_summary=imd_summary,
                    gate_status=gate_status,
                    evaluation_target=evaluation_target,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to write JSON report: {exc}") from exc


def _export_mirror_visualizations(
    pairs: list[tuple[str, np.ndarray, np.ndarray]],
    output_dir: Path,
    sample_rate: int,
    mirror_band_hz: tuple[float, float],
    n_fft: int,
    hop_length: int,
    max_exports: int,
) -> list[str]:
    """Export before/after mirror visualizations for selected samples.

    Args:
        pairs: Paired sample tuples (sample_id, input_signal, output_signal).
        output_dir: Root directory for plot exports.
        sample_rate: Signal sample rate in Hz.
        mirror_band_hz: Band used for mirror analysis in Hz.
        n_fft: STFT FFT size.
        hop_length: STFT hop size.
        max_exports: Maximum number of samples to export.

    Returns:
        List of exported visualization paths.

    Raises:
        ValueError: If max_exports is negative.
        RuntimeError: If export fails.

    Physical Basis:
        Visual confirmation of mirror-band suppression supports objective
        metrics and listening-test interpretation.
    """
    if max_exports < 0:
        raise ValueError(f"mirror_visual_limit must be >= 0, got {max_exports}.")

    exported_paths: list[str] = []
    for sample_id, input_signal, output_signal in pairs[:max_exports]:
        output_path = output_dir / f"{sample_id}_mirror_before_after.png"
        try:
            artifact = export_mirror_reduction_visualization(
                before_signal=input_signal,
                after_signal=output_signal,
                sample_rate=sample_rate,
                output_path=output_path,
                mirror_band_hz=mirror_band_hz,
                n_fft=n_fft,
                hop_length=hop_length,
                title=f"Mirror Comparison: {sample_id}",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to export mirror visualization for sample '{sample_id}': {exc}"
            ) from exc
        exported_paths.append(str(artifact.plot_path))
    return exported_paths


def _write_csv(
    result: DatasetEvaluationResult,
    path: Path,
    ringing_metrics: list[dict[str, Any]] | None = None,
    gate_status: dict[str, Any] | None = None,
) -> None:
    """Write per-sample metrics to CSV file.

    Args:
        result: Dataset evaluation result.
        path: Output CSV path.

    Raises:
        RuntimeError: If write operation fails.

    Physical Basis:
        Flat tabular metrics simplify signal-level error analysis in
        standard tooling.
    """
    rows = [sample_result_to_flat_dict(sample) for sample in result.samples]
    ringing_by_id = (
        {row["sample_id"]: row for row in ringing_metrics}
        if ringing_metrics is not None
        else {}
    )
    merged_rows: list[dict[str, Any]] = []
    gate_columns = _gate_status_to_csv_columns(gate_status)
    for row in rows:
        merged = dict(row)
        ringing_row = ringing_by_id.get(str(row["sample_id"]))
        if ringing_row is not None:
            for key, value in ringing_row.items():
                if key == "sample_id":
                    continue
                merged[key] = value
        merged.update(gate_columns)
        merged_rows.append(merged)
    fieldnames = list(merged_rows[0].keys())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)
    except Exception as exc:
        raise RuntimeError(f"Failed to write CSV report: {exc}") from exc


def _evaluate_ringing_dataset(
    *,
    pairs: list[tuple[str, np.ndarray, np.ndarray]],
    sample_rate: int,
    plateau_start_ms: float,
    plateau_end_ms: float,
    ringing_window_ms: float,
) -> list[dict[str, Any]]:
    """Evaluate edge-aligned ringing metrics on paired signals.

    Physical Basis:
        Comparing edge-aligned metrics against the reference SRC output catches
        transient regressions that mirror-only metrics can miss.
    """
    if len(pairs) == 0:
        raise ValueError("pairs cannot be empty")

    outputs: list[dict[str, Any]] = []
    for sample_id, input_signal, output_signal in pairs:
        comparison = compare_edge_aligned_ringing(
            before_signal=input_signal,
            after_signal=output_signal,
            sample_rate=sample_rate,
            plateau_start_ms=plateau_start_ms,
            plateau_end_ms=plateau_end_ms,
            ringing_window_ms=ringing_window_ms,
        )
        outputs.append(_ringing_comparison_to_payload(sample_id, comparison))
    return outputs


def _ringing_comparison_to_payload(
    sample_id: str,
    comparison: RingingComparisonMetrics,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "edge_index_before": comparison.before.edge_index,
        "edge_index_after": comparison.after.edge_index,
        "plateau_ripple_rms_before": comparison.before.plateau_ripple_rms,
        "plateau_ripple_rms_after": comparison.after.plateau_ripple_rms,
        "plateau_ripple_rms_ratio": comparison.plateau_ripple_rms_ratio,
        "plateau_ripple_p2p_before": comparison.before.plateau_ripple_p2p,
        "plateau_ripple_p2p_after": comparison.after.plateau_ripple_p2p,
        "plateau_ripple_p2p_ratio": comparison.plateau_ripple_p2p_ratio,
        "overshoot_abs_before": comparison.before.overshoot_abs,
        "overshoot_abs_after": comparison.after.overshoot_abs,
        "overshoot_abs_delta": comparison.overshoot_abs_delta,
        "ringing_ratio_before": comparison.before.post_to_pre_ringing_energy_ratio,
        "ringing_ratio_after": comparison.after.post_to_pre_ringing_energy_ratio,
        "ringing_ratio_delta": comparison.ringing_ratio_delta,
    }


def _summarize_ringing_metrics(
    ringing_metrics: list[dict[str, Any]],
) -> RingingDatasetSummary:
    """Aggregate per-sample ringing metrics into dataset summary."""
    if len(ringing_metrics) == 0:
        raise ValueError("ringing_metrics cannot be empty")

    def _mean(field_name: str) -> float:
        return float(np.mean([float(item[field_name]) for item in ringing_metrics]))

    return RingingDatasetSummary(
        num_samples=len(ringing_metrics),
        mean_plateau_ripple_rms_before=_mean("plateau_ripple_rms_before"),
        mean_plateau_ripple_rms_after=_mean("plateau_ripple_rms_after"),
        mean_plateau_ripple_rms_ratio=_mean("plateau_ripple_rms_ratio"),
        mean_plateau_ripple_p2p_before=_mean("plateau_ripple_p2p_before"),
        mean_plateau_ripple_p2p_after=_mean("plateau_ripple_p2p_after"),
        mean_plateau_ripple_p2p_ratio=_mean("plateau_ripple_p2p_ratio"),
        mean_overshoot_abs_before=_mean("overshoot_abs_before"),
        mean_overshoot_abs_after=_mean("overshoot_abs_after"),
        mean_overshoot_abs_delta=_mean("overshoot_abs_delta"),
        mean_ringing_ratio_before=_mean("ringing_ratio_before"),
        mean_ringing_ratio_after=_mean("ringing_ratio_after"),
        mean_ringing_ratio_delta=_mean("ringing_ratio_delta"),
    )


def _write_ringing_csv(ringing_metrics: list[dict[str, Any]], path: Path) -> None:
    """Write per-sample ringing metrics to CSV."""
    if len(ringing_metrics) == 0:
        raise ValueError("ringing_metrics cannot be empty")
    fieldnames = list(ringing_metrics[0].keys())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ringing_metrics)
    except Exception as exc:
        raise RuntimeError(f"Failed to write ringing CSV report: {exc}") from exc


def _write_ringing_json(
    *,
    ringing_metrics: list[dict[str, Any]],
    ringing_summary: RingingDatasetSummary,
    path: Path,
) -> None:
    """Write ringing summary + per-sample metrics to JSON."""
    payload = {
        "summary": asdict(ringing_summary),
        "samples": ringing_metrics,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to write ringing JSON report: {exc}") from exc


def _evaluate_stage1_gates(
    *,
    result: DatasetEvaluationResult,
    mirror_result: MirrorDatasetEvaluationResult,
    ringing_summary: RingingDatasetSummary,
    imd_summary: dict[str, Any] | None,
    gate_config: Stage1GateConfig,
) -> dict[str, Any]:
    """Evaluate Stage1 acceptance gates and include thresholds/observations.

    Physical Basis:
        Explicit gate evidence ensures acceptance/failure is reproducible in
        CI and offline analysis, not inferred from opaque summaries.
    """
    energy_observed = float(result.hb_energy_cap_violation_rate)
    mirror_observed = float(mirror_result.mean_metrics.symmetry_reduction_ratio)
    lb_phase_observed = float(result.mean_metrics.lb_phase_error_deg)
    lb_group_delay_observed = float(result.mean_metrics.lb_group_delay_error_samples)
    lb_amplitude_observed = float(result.mean_metrics.lb_amplitude_error_db)
    ringing_ratio_delta = float(ringing_summary.mean_ringing_ratio_delta)
    ringing_ratio_threshold = (
        0.0 if gate_config.require_nonpositive_ringing_ratio_delta else float("inf")
    )

    energy_pass = energy_observed <= 0.0
    mirror_pass = mirror_observed >= gate_config.mirror_target_reduction
    lowband_pass = bool(
        lb_phase_observed <= gate_config.max_lb_phase_error_deg
        and lb_group_delay_observed <= gate_config.max_lb_group_delay_error_samples
        and lb_amplitude_observed <= gate_config.max_lb_amplitude_error_db
    )
    ringing_rms_observed = float(ringing_summary.mean_plateau_ripple_rms_ratio)
    ringing_p2p_observed = float(ringing_summary.mean_plateau_ripple_p2p_ratio)
    overshoot_observed = float(ringing_summary.mean_overshoot_abs_delta)
    ringing_ratio_pass = (
        ringing_ratio_delta <= 0.0
        if gate_config.require_nonpositive_ringing_ratio_delta
        else True
    )
    ringing_pass = bool(
        ringing_rms_observed <= gate_config.max_plateau_ripple_rms_ratio
        and ringing_p2p_observed <= gate_config.max_plateau_ripple_p2p_ratio
        and overshoot_observed <= gate_config.max_overshoot_abs_increase
        and ringing_ratio_pass
    )
    imd_gate_payload: dict[str, Any]
    imd_pass = True
    if imd_summary is not None:
        thdn_observed = float(imd_summary["mean_thdn_improvement_db"])
        lower_imd_observed = bool(imd_summary["all_nmse_has_lower_imd"])
        imd_pass = bool(
            lower_imd_observed
            and (
                (thdn_observed > 0.0)
                if gate_config.require_positive_thdn_improvement
                else True
            )
        )
        imd_gate_payload = {
            "strict_selected": True,
            "passed": imd_pass,
            "threshold": {
                "require_nmse_has_lower_imd": True,
                "min_mean_thdn_improvement_db": (
                    0.0
                    if gate_config.require_positive_thdn_improvement
                    else float("-inf")
                ),
            },
            "observed": {
                "all_nmse_has_lower_imd": lower_imd_observed,
                "mean_thdn_improvement_db": thdn_observed,
            },
        }
    else:
        imd_gate_payload = {
            "strict_selected": False,
            "passed": True,
            "threshold": {},
            "observed": {},
            "note": "imd-naive-dir not provided; IMD gate skipped.",
        }

    stage1_acceptance_pass = bool(
        energy_pass and mirror_pass and ringing_pass and lowband_pass and imd_pass
    )
    failed_gates = [
        name
        for name, passed in (
            ("energy_cap", energy_pass),
            ("mirror_reduction", mirror_pass),
            ("lowband_preservation", lowband_pass),
            ("ringing_regression", ringing_pass),
            ("imd_proxy", imd_pass),
        )
        if not passed
    ]
    return {
        "stage1_acceptance_pass": stage1_acceptance_pass,
        "failed_gates": failed_gates,
        "energy_cap": {
            "strict_selected": True,
            "passed": energy_pass,
            "threshold": {"max_hb_energy_cap_violation_rate": 0.0},
            "observed": {"hb_energy_cap_violation_rate": energy_observed},
        },
        "mirror_reduction": {
            "strict_selected": True,
            "passed": mirror_pass,
            "threshold": {
                "min_symmetry_reduction_ratio": gate_config.mirror_target_reduction
            },
            "observed": {"symmetry_reduction_ratio": mirror_observed},
        },
        "lowband_preservation": {
            "strict_selected": True,
            "passed": lowband_pass,
            "threshold": {
                "max_lb_phase_error_deg": gate_config.max_lb_phase_error_deg,
                "max_lb_group_delay_error_samples": gate_config.max_lb_group_delay_error_samples,
                "max_lb_amplitude_error_db": gate_config.max_lb_amplitude_error_db,
            },
            "observed": {
                "lb_phase_error_deg": lb_phase_observed,
                "lb_group_delay_error_samples": lb_group_delay_observed,
                "lb_amplitude_error_db": lb_amplitude_observed,
            },
        },
        "ringing_regression": {
            "strict_selected": True,
            "passed": ringing_pass,
            "threshold": {
                "max_plateau_ripple_rms_ratio": gate_config.max_plateau_ripple_rms_ratio,
                "max_plateau_ripple_p2p_ratio": gate_config.max_plateau_ripple_p2p_ratio,
                "max_overshoot_abs_increase": gate_config.max_overshoot_abs_increase,
                "max_ringing_ratio_delta": ringing_ratio_threshold,
                "allow_ringing_ratio_increase": (
                    not gate_config.require_nonpositive_ringing_ratio_delta
                ),
            },
            "observed": {
                "mean_plateau_ripple_rms_ratio": ringing_rms_observed,
                "mean_plateau_ripple_p2p_ratio": ringing_p2p_observed,
                "mean_overshoot_abs_delta": overshoot_observed,
                "mean_ringing_ratio_delta": ringing_ratio_delta,
            },
        },
        "imd_proxy": imd_gate_payload,
    }


def _gate_status_to_csv_columns(gate_status: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten gate status payload into CSV columns."""
    if gate_status is None:
        return {}

    def _gate(key: str) -> dict[str, Any]:
        value = gate_status.get(key)
        return value if isinstance(value, dict) else {}

    energy = _gate("energy_cap")
    mirror = _gate("mirror_reduction")
    ringing = _gate("ringing_regression")
    lowband = _gate("lowband_preservation")
    imd = _gate("imd_proxy")
    energy_observed = energy.get("observed", {})
    mirror_observed = mirror.get("observed", {})
    ringing_observed = ringing.get("observed", {})
    lowband_observed = lowband.get("observed", {})
    imd_observed = imd.get("observed", {})
    return {
        "gate_stage1_acceptance_pass": gate_status.get("stage1_acceptance_pass"),
        "gate_failed_gates": ",".join(
            str(item) for item in gate_status.get("failed_gates", [])
        ),
        "gate_energy_cap_pass": energy.get("passed"),
        "gate_energy_cap_violation_rate_observed": energy_observed.get(
            "hb_energy_cap_violation_rate"
        ),
        "gate_mirror_reduction_pass": mirror.get("passed"),
        "gate_mirror_symmetry_reduction_ratio_observed": mirror_observed.get(
            "symmetry_reduction_ratio"
        ),
        "gate_lowband_preservation_pass": lowband.get("passed"),
        "gate_lowband_phase_error_deg_observed": lowband_observed.get(
            "lb_phase_error_deg"
        ),
        "gate_lowband_group_delay_error_samples_observed": lowband_observed.get(
            "lb_group_delay_error_samples"
        ),
        "gate_lowband_amplitude_error_db_observed": lowband_observed.get(
            "lb_amplitude_error_db"
        ),
        "gate_ringing_regression_pass": ringing.get("passed"),
        "gate_ringing_plateau_rms_ratio_observed": ringing_observed.get(
            "mean_plateau_ripple_rms_ratio"
        ),
        "gate_ringing_plateau_p2p_ratio_observed": ringing_observed.get(
            "mean_plateau_ripple_p2p_ratio"
        ),
        "gate_ringing_overshoot_delta_observed": ringing_observed.get(
            "mean_overshoot_abs_delta"
        ),
        "gate_ringing_ratio_delta_observed": ringing_observed.get(
            "mean_ringing_ratio_delta"
        ),
        "gate_imd_proxy_pass": imd.get("passed"),
        "gate_imd_mean_thdn_improvement_db_observed": imd_observed.get(
            "mean_thdn_improvement_db"
        ),
        "gate_imd_all_nmse_has_lower_imd_observed": imd_observed.get(
            "all_nmse_has_lower_imd"
        ),
    }


def _raise_for_strict_gate_failures(
    *, args: argparse.Namespace, gate_status: dict[str, Any]
) -> None:
    """Raise `SystemExit` with fixed code mapping for strict gate failures."""
    failures: list[int] = []

    def _strict_failed(strict_flag: bool, gate_key: str, exit_code: int) -> None:
        if not strict_flag:
            return
        gate_raw = gate_status.get(gate_key)
        if not isinstance(gate_raw, dict):
            raise RuntimeError(f"gate_status missing gate '{gate_key}'")
        if not bool(gate_raw.get("passed")):
            failures.append(exit_code)

    _strict_failed(args.strict_energy_cap, "energy_cap", _STRICT_EXIT_ENERGY_CAP)
    _strict_failed(
        args.strict_mirror_reduction,
        "mirror_reduction",
        _STRICT_EXIT_MIRROR_REDUCTION,
    )
    _strict_failed(
        args.strict_ringing_regression,
        "ringing_regression",
        _STRICT_EXIT_RINGING_REGRESSION,
    )
    _strict_failed(
        args.strict_lowband_preservation,
        "lowband_preservation",
        _STRICT_EXIT_LOWBAND_PRESERVATION,
    )
    _strict_failed(
        args.strict_imd_proxy,
        "imd_proxy",
        _STRICT_EXIT_IMD_PROXY,
    )

    if len(failures) == 0:
        return
    if len(failures) > 1:
        raise SystemExit(_STRICT_EXIT_MULTIPLE_GATES)
    raise SystemExit(failures[0])


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before metric evaluation."""
    if args.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if args.cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be positive")
    if args.energy_cap < 0.0:
        raise ValueError("energy_cap must be non-negative")
    if args.num_taps <= 0:
        raise ValueError("num_taps must be positive")
    if args.n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if args.hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if args.target_dir is not None and (
        not args.target_dir.exists() or not args.target_dir.is_dir()
    ):
        raise FileNotFoundError(f"target_dir not found: {args.target_dir}")
    if args.evaluation_target != "input" and args.target_dir is None:
        raise ValueError("target_dir is required when evaluation_target is not 'input'")
    if args.strict_imd_proxy and args.imd_naive_dir is None:
        raise ValueError("imd_naive_dir is required when strict_imd_proxy is enabled")
    mirror_lower_hz = float(args.mirror_band_hz[0])
    mirror_upper_hz = float(args.mirror_band_hz[1])
    if mirror_lower_hz < 0.0:
        raise ValueError("mirror-band lower bound must be non-negative")
    if mirror_upper_hz <= mirror_lower_hz:
        raise ValueError("mirror-band upper bound must be greater than lower bound")
    if args.mirror_target_reduction < 0.0 or args.mirror_target_reduction > 1.0:
        raise ValueError("mirror_target_reduction must be in [0.0, 1.0]")
    if args.max_plateau_ripple_rms_ratio <= 0.0:
        raise ValueError("max_plateau_ripple_rms_ratio must be positive")
    if args.max_plateau_ripple_p2p_ratio <= 0.0:
        raise ValueError("max_plateau_ripple_p2p_ratio must be positive")
    if args.max_overshoot_abs_increase < 0.0:
        raise ValueError("max_overshoot_abs_increase must be non-negative")
    if args.max_lb_phase_error_deg <= 0.0:
        raise ValueError("max_lb_phase_error_deg must be positive")
    if args.max_lb_group_delay_error_samples <= 0.0:
        raise ValueError("max_lb_group_delay_error_samples must be positive")
    if args.max_lb_amplitude_error_db > 0.0:
        raise ValueError("max_lb_amplitude_error_db must be <= 0.0")
    if args.ringing_plateau_start_ms < 0.0:
        raise ValueError("ringing_plateau_start_ms must be non-negative")
    if args.ringing_plateau_end_ms <= args.ringing_plateau_start_ms:
        raise ValueError("ringing_plateau_end_ms must be greater than start")
    if args.ringing_window_ms <= 0.0:
        raise ValueError("ringing_window_ms must be positive")
    if args.mirror_visual_limit < 0:
        raise ValueError("mirror_visual_limit must be non-negative")


def _print_summary(
    result: DatasetEvaluationResult,
    mirror_result: MirrorDatasetEvaluationResult,
    *,
    evaluation_target: str,
) -> None:
    """Print concise summary report to stdout.

    Physical Basis:
        Fast terminal summaries support quick iterative checks before
        deeper JSON/CSV analysis.
    """
    summary = result.mean_metrics
    print(f"Evaluation target: {evaluation_target}")
    print(f"Samples: {len(result.samples)}")
    print(f"LB amplitude error (dB): {summary.lb_amplitude_error_db:.6f}")
    print(f"LB phase error (deg): {summary.lb_phase_error_deg:.6f}")
    print(f"LB group delay error (samples): {summary.lb_group_delay_error_samples:.6f}")
    print(f"Mirror reduction ratio: {summary.mirror_reduction_ratio:.6f}")
    print(
        "Mirror symmetry reduction ratio (mean): "
        f"{mirror_result.mean_metrics.symmetry_reduction_ratio:.6f}"
    )
    print(
        "Mirror target pass rate: "
        f"{mirror_result.symmetry_target_pass_rate:.6f} "
        f"(target={mirror_result.target_reduction_ratio:.2f})"
    )
    print(f"HB energy: {summary.hb_energy:.8e}")
    print(f"Touch metric: {summary.touch_metric:.6f}")
    print(f"HB energy cap violation rate: {result.hb_energy_cap_violation_rate:.6f}")


def _resolve_evaluation_pairs(
    *,
    pairs: list[tuple[str, np.ndarray, np.ndarray]],
    target_dir: Path | None,
    evaluation_target: str,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Resolve (reference, output) pairs used for low-band/ringing evaluation."""
    if evaluation_target == "input":
        return pairs
    if target_dir is None:
        raise ValueError("target_dir is required when evaluation_target is not 'input'")

    resolved: list[tuple[str, np.ndarray, np.ndarray]] = []
    output_map = {sample_id: output for sample_id, _, output in pairs}
    for sample_id in sorted(output_map.keys()):
        target_path = target_dir / f"{sample_id}.npy"
        if not target_path.exists():
            raise FileNotFoundError(f"Missing evaluation target pair: {target_path}")
        try:
            target_signal = np.asarray(np.load(target_path), dtype=np.float64)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load evaluation target '{target_path.name}': {exc}"
            ) from exc
        resolved.append((sample_id, target_signal, output_map[sample_id]))
    return resolved


def _evaluate_imd_dataset(
    *,
    naive_dir: Path | None,
    nmse_dir: Path,
    sample_rate: int,
) -> dict[str, Any] | None:
    """Evaluate IMD proxy over shared sample IDs when naive references exist."""
    if naive_dir is None:
        return None
    if not naive_dir.exists() or not naive_dir.is_dir():
        raise FileNotFoundError(f"imd_naive_dir not found: {naive_dir}")
    if not nmse_dir.exists() or not nmse_dir.is_dir():
        raise FileNotFoundError(f"nmse_dir not found: {nmse_dir}")

    sample_ids = sorted(path.stem for path in naive_dir.glob("*.npy"))
    if len(sample_ids) == 0:
        raise FileNotFoundError(f"No IMD naive files found in {naive_dir}")

    per_sample: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        naive_path = naive_dir / f"{sample_id}.npy"
        nmse_path = nmse_dir / f"{sample_id}.npy"
        if not nmse_path.exists():
            raise FileNotFoundError(f"Missing NMSE IMD pair: {nmse_path}")
        naive_signal = np.asarray(np.load(naive_path), dtype=np.float64)
        nmse_signal = np.asarray(np.load(nmse_path), dtype=np.float64)
        metrics = evaluate_imd_proxy(
            naive_signal=naive_signal,
            nmse_signal=nmse_signal,
            sample_rate=sample_rate,
            num_taps=1025,
        )
        per_sample.append(
            {
                "sample_id": sample_id,
                "audible_distortion_reduction_db": metrics.audible_distortion_reduction_db,
                "thdn_improvement_db": metrics.thdn_improvement_db,
                "nmse_has_lower_imd": metrics.nmse_has_lower_imd,
                "thdn_improvement_over_10db": metrics.thdn_improvement_over_10db,
            }
        )

    return {
        "num_samples": len(per_sample),
        "mean_audible_distortion_reduction_db": float(
            np.mean(
                [float(item["audible_distortion_reduction_db"]) for item in per_sample]
            )
        ),
        "mean_thdn_improvement_db": float(
            np.mean([float(item["thdn_improvement_db"]) for item in per_sample])
        ),
        "all_nmse_has_lower_imd": bool(
            all(bool(item["nmse_has_lower_imd"]) for item in per_sample)
        ),
        "all_thdn_improvement_over_10db": bool(
            all(bool(item["thdn_improvement_over_10db"]) for item in per_sample)
        ),
        "samples": per_sample,
    }


def _prepare_integrated_report_outputs(args: argparse.Namespace) -> None:
    """Populate default artifact paths when report-dir is specified."""
    report_dir = args.report_dir
    if report_dir is None:
        return
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.json is None:
        args.json = report_dir / "stage1_integrated_evaluation.json"
    if args.markdown is None:
        args.markdown = report_dir / "stage1_integrated_evaluation.md"
    if args.mirror_visual_dir is None:
        args.mirror_visual_dir = report_dir / "figures" / "mirror"
    if args.ringing_json is None:
        args.ringing_json = report_dir / "ringing_metrics.json"
    if args.ringing_csv is None:
        args.ringing_csv = report_dir / "ringing_metrics.csv"


def _write_markdown_report(
    *,
    path: Path,
    result: DatasetEvaluationResult,
    mirror_result: MirrorDatasetEvaluationResult,
    ringing_summary: RingingDatasetSummary,
    gate_status: dict[str, Any],
    visual_exports: list[str],
    imd_summary: dict[str, Any] | None,
    evaluation_target: str,
) -> None:
    """Write integrated Stage1 markdown report with explicit gate outcomes."""
    summary = result.mean_metrics
    failed_gates = gate_status.get("failed_gates", [])
    failed_text = (
        ", ".join(str(item) for item in failed_gates) if failed_gates else "none"
    )
    lines = [
        "# Stage1 Integrated Evaluation Report",
        "",
        f"- evaluation_target: `{evaluation_target}`",
        f"- num_samples: `{len(result.samples)}`",
        f"- stage1_acceptance_pass: `{gate_status.get('stage1_acceptance_pass')}`",
        f"- failed_gates: `{failed_text}`",
        "",
        "## Lowband Preservation",
        "",
        f"- lb_amplitude_error_db: `{summary.lb_amplitude_error_db:.6f}`",
        f"- lb_phase_error_deg: `{summary.lb_phase_error_deg:.6f}`",
        f"- lb_group_delay_error_samples: `{summary.lb_group_delay_error_samples:.6f}`",
        "",
        "## Mirror / HB Safety",
        "",
        f"- symmetry_reduction_ratio: `{mirror_result.mean_metrics.symmetry_reduction_ratio:.6f}`",
        f"- hb_energy_cap_violation_rate: `{result.hb_energy_cap_violation_rate:.6f}`",
        "",
        "## Ringing Regression",
        "",
        f"- mean_plateau_ripple_rms_ratio: `{ringing_summary.mean_plateau_ripple_rms_ratio:.6f}`",
        f"- mean_plateau_ripple_p2p_ratio: `{ringing_summary.mean_plateau_ripple_p2p_ratio:.6f}`",
        f"- mean_overshoot_abs_delta: `{ringing_summary.mean_overshoot_abs_delta:.6f}`",
        f"- mean_ringing_ratio_delta: `{ringing_summary.mean_ringing_ratio_delta:.6f}`",
        "",
        "## IMD Proxy",
        "",
    ]
    if imd_summary is None:
        lines.append("- skipped (`--imd-naive-dir` not provided)")
    else:
        lines.extend(
            [
                f"- mean_thdn_improvement_db: `{float(imd_summary['mean_thdn_improvement_db']):.6f}`",
                f"- all_nmse_has_lower_imd: `{bool(imd_summary['all_nmse_has_lower_imd'])}`",
                f"- mean_audible_distortion_reduction_db: `{float(imd_summary['mean_audible_distortion_reduction_db']):.6f}`",
            ]
        )
    lines.extend(["", "## Figures", ""])
    if len(visual_exports) == 0:
        lines.append("- none")
    else:
        lines.extend([f"- `{item}`" for item in visual_exports])

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to write markdown report: {exc}") from exc


if __name__ == "__main__":
    main()
