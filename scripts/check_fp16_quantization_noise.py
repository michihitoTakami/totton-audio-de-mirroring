"""Evaluate FP16 quantization noise for high-band Stage1 signals."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

_EPSILON = 1.0e-15


@dataclass(frozen=True)
class NoiseMetrics:
    """Quantization-noise metrics for one waveform.

    Args:
        signal_rms: RMS amplitude of original signal.
        error_rms: RMS amplitude of quantization error.
        error_peak: Peak absolute quantization error.
        snr_db: Signal-to-quantization-noise ratio in dB.
        error_rms_dbfs: Error RMS level in dBFS.

    Physical Basis:
        FP16 quantization error in Stage1 HB path must remain low enough to
        avoid audible regression while preserving suppression behavior.
    """

    signal_rms: float
    error_rms: float
    error_peak: float
    snr_db: float
    error_rms_dbfs: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Physical Basis:
        Fixed thresholds make quality regression checks reproducible across
        environments and model variants.
    """
    parser = argparse.ArgumentParser(
        description="Check FP16 quantization noise level for Stage1 HB signals."
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=88_200,
        help="Sample rate used to synthesize high-band test signals.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=1.0,
        help="Duration of each test signal.",
    )
    parser.add_argument(
        "--min-snr-db",
        type=float,
        default=70.0,
        help="Minimum acceptable SNR for each case.",
    )
    parser.add_argument(
        "--max-error-rms-dbfs",
        type=float,
        default=-80.0,
        help="Maximum acceptable error RMS level in dBFS (must be negative).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON payload only.",
    )
    return parser.parse_args()


def quantize_fp16_roundtrip(signal: np.ndarray) -> np.ndarray:
    """Quantize signal via float16 round-trip.

    Args:
        signal: Input waveform.

    Returns:
        Signal after float16 -> float32 round-trip.

    Raises:
        ValueError: If signal is not finite 1D data.

    Physical Basis:
        This approximates precision loss introduced by FP16 arithmetic in
        TensorRT kernels used for Stage1 HB model execution.
    """
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"signal must be 1D, got ndim={array.ndim}.")
    if array.size == 0:
        raise ValueError("signal must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError("signal contains non-finite values.")
    return np.asarray(array.astype(np.float16).astype(np.float32), dtype=np.float64)


def compute_noise_metrics(signal: np.ndarray) -> NoiseMetrics:
    """Compute quantization-noise metrics for one signal.

    Args:
        signal: Input waveform in [-1.0, 1.0] range.

    Returns:
        Quantization metrics.

    Physical Basis:
        SNR and dBFS error summarize numerical noise energy relevant to
        audible artifacts and Stage1 suppression stability.
    """
    original = np.asarray(signal, dtype=np.float64)
    quantized = quantize_fp16_roundtrip(original)
    error = quantized - original

    signal_rms = float(np.sqrt(np.mean(np.square(original))))
    error_rms = float(np.sqrt(np.mean(np.square(error))))
    error_peak = float(np.max(np.abs(error)))
    snr_db = float(20.0 * np.log10((signal_rms + _EPSILON) / (error_rms + _EPSILON)))
    error_rms_dbfs = float(20.0 * np.log10(error_rms + _EPSILON))
    return NoiseMetrics(
        signal_rms=signal_rms,
        error_rms=error_rms,
        error_peak=error_peak,
        snr_db=snr_db,
        error_rms_dbfs=error_rms_dbfs,
    )


def generate_highband_cases(
    *,
    sample_rate: int,
    duration_sec: float,
) -> dict[str, np.ndarray]:
    """Generate representative high-band waveforms.

    Args:
        sample_rate: Signal sample rate in Hz.
        duration_sec: Signal duration in seconds.

    Returns:
        Mapping from case name to synthesized waveform.

    Raises:
        ValueError: If sample rate or duration is invalid.

    Physical Basis:
        Cases focus on 20-44kHz where Stage1 suppression and mixed-precision
        acceleration are applied.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if duration_sec <= 0.0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}.")

    num_samples = int(round(float(sample_rate) * float(duration_sec)))
    if num_samples <= 0:
        raise ValueError("duration_sec results in zero samples.")

    time_axis = np.arange(num_samples, dtype=np.float64) / float(sample_rate)
    return {
        "hb_sine_30k_amp0p5": np.asarray(
            0.5 * np.sin(2.0 * np.pi * 30_000.0 * time_axis),
            dtype=np.float64,
        ),
        "hb_sine_30k_amp0p1": np.asarray(
            0.1 * np.sin(2.0 * np.pi * 30_000.0 * time_axis),
            dtype=np.float64,
        ),
        "hb_multitone": np.asarray(
            0.2 * np.sin(2.0 * np.pi * 22_000.0 * time_axis)
            + 0.2 * np.sin(2.0 * np.pi * 30_000.0 * time_axis)
            + 0.1 * np.sin(2.0 * np.pi * 40_000.0 * time_axis),
            dtype=np.float64,
        ),
    }


def evaluate_fp16_quantization_noise(
    *,
    sample_rate: int,
    duration_sec: float,
) -> dict[str, NoiseMetrics]:
    """Evaluate all default high-band quantization test cases.

    Args:
        sample_rate: Signal sample rate in Hz.
        duration_sec: Duration for synthesized cases.

    Returns:
        Case metrics keyed by case name.

    Physical Basis:
        Evaluating multiple HB envelopes catches worst-case and practical
        quantization noise bounds for mixed-precision deployment.
    """
    cases = generate_highband_cases(sample_rate=sample_rate, duration_sec=duration_sec)
    return {name: compute_noise_metrics(signal) for name, signal in cases.items()}


def _fails_thresholds(
    metrics: dict[str, NoiseMetrics],
    *,
    min_snr_db: float,
    max_error_rms_dbfs: float,
) -> list[str]:
    failures: list[str] = []
    for name, item in metrics.items():
        if item.snr_db < min_snr_db:
            failures.append(
                f"{name}: snr_db={item.snr_db:.2f} < min_snr_db={min_snr_db:.2f}"
            )
        if item.error_rms_dbfs > max_error_rms_dbfs:
            failures.append(
                f"{name}: error_rms_dbfs={item.error_rms_dbfs:.2f} > "
                f"max_error_rms_dbfs={max_error_rms_dbfs:.2f}"
            )
    return failures


def main() -> None:
    """Run quantization-noise evaluation CLI."""
    args = parse_args()
    if args.max_error_rms_dbfs >= 0.0:
        raise ValueError("max_error_rms_dbfs must be negative dBFS value.")

    metrics = evaluate_fp16_quantization_noise(
        sample_rate=int(args.sample_rate),
        duration_sec=float(args.duration_sec),
    )
    failures = _fails_thresholds(
        metrics,
        min_snr_db=float(args.min_snr_db),
        max_error_rms_dbfs=float(args.max_error_rms_dbfs),
    )

    payload = {
        "sample_rate": int(args.sample_rate),
        "duration_sec": float(args.duration_sec),
        "thresholds": {
            "min_snr_db": float(args.min_snr_db),
            "max_error_rms_dbfs": float(args.max_error_rms_dbfs),
        },
        "metrics": {name: asdict(item) for name, item in metrics.items()},
        "failures": failures,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for name, item in metrics.items():
            print(
                f"{name}: snr_db={item.snr_db:.2f}, "
                f"error_rms_dbfs={item.error_rms_dbfs:.2f}, "
                f"error_peak={item.error_peak:.6e}"
            )
        if failures:
            print("Threshold failures:")
            for line in failures:
                print(f"- {line}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
