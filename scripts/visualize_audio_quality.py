"""Comprehensive audio quality visualization script.

Generate frequency response and THD+N spectrum visualizations for
Stage 1 (88.2kHz) and Stage 2 (705.6kHz) evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from totton_audio_de_mirroring.evaluation.frequency_response import (
    evaluate_frequency_response_pair,
)
from totton_audio_de_mirroring.evaluation.thdn_visualization import (
    evaluate_thdn_spectrum_pair,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for audio quality visualization.

    Physical Basis:
        Fixed visualization parameters ensure reproducible quality
        assessment across different model checkpoints and datasets.
    """
    parser = argparse.ArgumentParser(
        description="Visualize audio quality: frequency response and THD+N."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input (before) .npy files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing output (after) .npy files",
    )
    parser.add_argument(
        "--visual-dir",
        type=Path,
        required=True,
        help="Directory to save visualization PNG files",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.npy",
        help="File glob pattern for .npy files",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=88_200,
        help="Sample rate in Hz (default: 88200 for Stage 1)",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=8192,
        help="FFT size for frequency analysis",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=20_000.0,
        help="Audible band cutoff in Hz",
    )
    parser.add_argument(
        "--num-taps",
        type=int,
        default=1025,
        help="FIR filter taps for band splitting",
    )
    parser.add_argument(
        "--clip-drive",
        type=float,
        default=2.0,
        help="Soft-clipping drive for THD+N simulation",
    )
    parser.add_argument(
        "--max-freq-khz",
        type=float,
        default=None,
        help="Maximum frequency to display in plots (kHz), default: Nyquist",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path to save summary metrics JSON",
    )
    return parser.parse_args()


def load_sample_pairs(
    input_dir: Path,
    output_dir: Path,
    glob_pattern: str = "*.npy",
    limit: int | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load paired input/output samples.

    Args:
        input_dir: Directory containing input .npy files.
        output_dir: Directory containing output .npy files.
        glob_pattern: File glob pattern.
        limit: Maximum number of samples to load.

    Returns:
        List of (sample_id, before_signal, after_signal) tuples.

    Raises:
        FileNotFoundError: If paired files are missing.

    Physical Basis:
        Paired comparison enables assessment of spectral changes
        introduced by NMSE processing while maintaining signal identity.
    """
    input_files = sorted(input_dir.glob(glob_pattern))
    if limit is not None:
        input_files = input_files[:limit]

    pairs = []
    for input_path in input_files:
        output_path = output_dir / input_path.name

        if not output_path.exists():
            print(f"Warning: Output file not found for {input_path.name}, skipping")
            continue

        sample_id = input_path.stem
        before_signal = np.load(input_path)
        after_signal = np.load(output_path)

        if before_signal.ndim != 1 or after_signal.ndim != 1:
            print(f"Warning: Skipping non-1D signal: {sample_id}")
            continue

        if before_signal.shape != after_signal.shape:
            print(
                f"Warning: Shape mismatch for {sample_id}: "
                f"{before_signal.shape} vs {after_signal.shape}, skipping"
            )
            continue

        pairs.append((sample_id, before_signal, after_signal))

    return pairs


def main() -> None:
    """Run comprehensive audio quality visualization.

    Raises:
        FileNotFoundError: If input directories don't exist.
        RuntimeError: If visualization generation fails.

    Physical Basis:
        Comprehensive visualization reveals both frequency-domain behavior
        (anti-aliasing, imaging artifacts) and distortion characteristics
        (THD+N, nonlinear products) to validate NMSE quality.
    """
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {args.output_dir}")

    args.visual_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading sample pairs from {args.input_dir} and {args.output_dir}...")
    pairs = load_sample_pairs(args.input_dir, args.output_dir, args.glob, args.limit)

    if not pairs:
        raise RuntimeError("No valid sample pairs found")

    print(f"Found {len(pairs)} sample pairs")

    freq_metrics_all = []
    thdn_metrics_all = []

    for sample_id, before_signal, after_signal in pairs:
        print(f"\nProcessing {sample_id}...")

        # Frequency response visualization
        freq_output_path = args.visual_dir / f"{sample_id}_frequency_response.png"
        print(f"  - Generating frequency response: {freq_output_path.name}")

        try:
            freq_before, freq_after = evaluate_frequency_response_pair(
                before_signal=before_signal,
                after_signal=after_signal,
                sample_rate=args.sample_rate,
                output_path=freq_output_path,
                n_fft=args.n_fft,
                title=f"Frequency Response: {sample_id}",
                max_freq_khz=args.max_freq_khz,
            )
            freq_metrics_all.append(
                {
                    "sample_id": sample_id,
                    "before_attenuation_44khz_db": freq_before.attenuation_44khz_db,
                    "after_attenuation_44khz_db": freq_after.attenuation_44khz_db,
                    "before_imaging_100khz_plus": freq_before.imaging_energy_100khz_plus,
                    "after_imaging_100khz_plus": freq_after.imaging_energy_100khz_plus,
                }
            )
        except Exception as e:
            print(f"  ! Frequency response failed: {e}")

        # THD+N spectrum visualization
        thdn_output_path = args.visual_dir / f"{sample_id}_thdn_spectrum.png"
        print(f"  - Generating THD+N spectrum: {thdn_output_path.name}")

        try:
            thdn_before, thdn_after = evaluate_thdn_spectrum_pair(
                before_signal=before_signal,
                after_signal=after_signal,
                sample_rate=args.sample_rate,
                output_path=thdn_output_path,
                cutoff_hz=args.cutoff_hz,
                num_taps=args.num_taps,
                clip_drive=args.clip_drive,
                n_fft=args.n_fft,
                title=f"THD+N Spectrum: {sample_id}",
            )
            thdn_metrics_all.append(
                {
                    "sample_id": sample_id,
                    "before_thdn_db": thdn_before.thdn_db,
                    "after_thdn_db": thdn_after.thdn_db,
                    "thdn_improvement_db": thdn_before.thdn_db - thdn_after.thdn_db,
                    "before_max_harmonic_db": thdn_before.max_harmonic_db,
                    "after_max_harmonic_db": thdn_after.max_harmonic_db,
                }
            )
        except Exception as e:
            print(f"  ! THD+N spectrum failed: {e}")

    # Compute summary statistics
    if freq_metrics_all:
        freq_summary = {
            "num_samples": len(freq_metrics_all),
            "avg_before_attenuation_44khz_db": float(
                np.mean([m["before_attenuation_44khz_db"] for m in freq_metrics_all])
            ),
            "avg_after_attenuation_44khz_db": float(
                np.mean([m["after_attenuation_44khz_db"] for m in freq_metrics_all])
            ),
            "avg_before_imaging_100khz_plus": float(
                np.mean([m["before_imaging_100khz_plus"] for m in freq_metrics_all])
            ),
            "avg_after_imaging_100khz_plus": float(
                np.mean([m["after_imaging_100khz_plus"] for m in freq_metrics_all])
            ),
        }
    else:
        freq_summary = {}

    if thdn_metrics_all:
        thdn_summary = {
            "num_samples": len(thdn_metrics_all),
            "avg_before_thdn_db": float(
                np.mean([m["before_thdn_db"] for m in thdn_metrics_all])
            ),
            "avg_after_thdn_db": float(
                np.mean([m["after_thdn_db"] for m in thdn_metrics_all])
            ),
            "avg_thdn_improvement_db": float(
                np.mean([m["thdn_improvement_db"] for m in thdn_metrics_all])
            ),
            "avg_before_max_harmonic_db": float(
                np.mean([m["before_max_harmonic_db"] for m in thdn_metrics_all])
            ),
            "avg_after_max_harmonic_db": float(
                np.mean([m["after_max_harmonic_db"] for m in thdn_metrics_all])
            ),
        }
    else:
        thdn_summary = {}

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if freq_summary:
        print("\nFrequency Response:")
        print(f"  Samples: {freq_summary['num_samples']}")
        print(
            f"  Attenuation @ 44.1kHz: "
            f"{freq_summary['avg_before_attenuation_44khz_db']:.1f} dB (before) -> "
            f"{freq_summary['avg_after_attenuation_44khz_db']:.1f} dB (after)"
        )
        print(
            f"  Imaging >100kHz: "
            f"{freq_summary['avg_before_imaging_100khz_plus']:.2e} (before) -> "
            f"{freq_summary['avg_after_imaging_100khz_plus']:.2e} (after)"
        )

    if thdn_summary:
        print("\nTHD+N:")
        print(f"  Samples: {thdn_summary['num_samples']}")
        print(
            f"  THD+N: "
            f"{thdn_summary['avg_before_thdn_db']:.1f} dB (before) -> "
            f"{thdn_summary['avg_after_thdn_db']:.1f} dB (after)"
        )
        print(f"  Improvement: {thdn_summary['avg_thdn_improvement_db']:.1f} dB")
        print(
            f"  Max Harmonic: "
            f"{thdn_summary['avg_before_max_harmonic_db']:.1f} dB (before) -> "
            f"{thdn_summary['avg_after_max_harmonic_db']:.1f} dB (after)"
        )

    # Save summary JSON
    if args.summary_json is not None:
        summary = {
            "frequency_response": {
                "summary": freq_summary,
                "samples": freq_metrics_all,
            },
            "thdn": {
                "summary": thdn_summary,
                "samples": thdn_metrics_all,
            },
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {args.summary_json}")

    print(f"\nVisualizations saved to: {args.visual_dir}")


if __name__ == "__main__":
    main()
