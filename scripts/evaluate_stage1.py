"""CLI for Stage 1 hard metrics evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from totton_audio_de_mirroring.evaluation.metrics import (
    DatasetEvaluationResult,
    evaluate_dataset,
    sample_result_to_flat_dict,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Stage 1 evaluation.

    Physical Basis:
        Fixed evaluation parameters make hard-metric comparisons
        reproducible across model checkpoints and datasets.
    """
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 hard metrics.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--strict-energy-cap", action="store_true")
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
    pairs = _load_pairs(args.input_dir, args.output_dir, args.glob)
    result = evaluate_dataset(
        samples=pairs,
        sample_rate=args.sample_rate,
        cutoff_hz=args.cutoff_hz,
        energy_cap=args.energy_cap,
        num_taps=args.num_taps,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        mirror_band_hz=(args.mirror_band_hz[0], args.mirror_band_hz[1]),
    )

    if args.csv is not None:
        _write_csv(result, args.csv)
    if args.json is not None:
        _write_json(result, args.json)

    if args.print_json:
        print(json.dumps(_to_payload(result), indent=2, sort_keys=True))
    else:
        _print_summary(result)

    if args.strict_energy_cap and result.hb_energy_cap_violation_rate > 0.0:
        raise SystemExit(2)


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


def _to_payload(result: DatasetEvaluationResult) -> dict[str, Any]:
    """Convert dataset evaluation to JSON-serializable payload.

    Physical Basis:
        Structured payloads preserve metric definitions for downstream
        plotting and regression checks.
    """
    return {
        "summary": asdict(result.mean_metrics)
        | {
            "hb_energy_cap_violation_rate": result.hb_energy_cap_violation_rate,
            "num_samples": len(result.samples),
        },
        "samples": [sample_result_to_flat_dict(sample) for sample in result.samples],
    }


def _write_json(result: DatasetEvaluationResult, path: Path) -> None:
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
        path.write_text(json.dumps(_to_payload(result), indent=2, sort_keys=True))
    except Exception as exc:
        raise RuntimeError(f"Failed to write JSON report: {exc}") from exc


def _write_csv(result: DatasetEvaluationResult, path: Path) -> None:
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
    fieldnames = list(rows[0].keys())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        raise RuntimeError(f"Failed to write CSV report: {exc}") from exc


def _print_summary(result: DatasetEvaluationResult) -> None:
    """Print concise summary report to stdout.

    Physical Basis:
        Fast terminal summaries support quick iterative checks before
        deeper JSON/CSV analysis.
    """
    summary = result.mean_metrics
    print(f"Samples: {len(result.samples)}")
    print(f"LB amplitude error (dB): {summary.lb_amplitude_error_db:.6f}")
    print(f"LB phase error (deg): {summary.lb_phase_error_deg:.6f}")
    print(f"LB group delay error (samples): {summary.lb_group_delay_error_samples:.6f}")
    print(f"Mirror reduction ratio: {summary.mirror_reduction_ratio:.6f}")
    print(f"HB energy: {summary.hb_energy:.8e}")
    print(f"Touch metric: {summary.touch_metric:.6f}")
    print(f"HB energy cap violation rate: {result.hb_energy_cap_violation_rate:.6f}")


if __name__ == "__main__":
    main()
