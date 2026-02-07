"""IMD proxy evaluation utilities for Stage 1 outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from totton_audio_de_mirroring.data.filters import band_split, design_band_split_filters

DEFAULT_CUTOFF_HZ = 20_000.0
DEFAULT_NUM_TAPS = 1025
DEFAULT_CLIP_DRIVE = 2.0
EPSILON = 1.0e-12


@dataclass(frozen=True)
class IMDPathMetrics:
    """Per-signal IMD proxy metrics after nonlinear simulation.

    Args:
        audible_distortion_energy: Mean squared low-band distortion energy.
        thdn_db: THD+N-like ratio in dB for low-band component.

    Physical Basis:
        A light nonlinear analog stage maps high-band energy into audible
        intermodulation products. Distortion is quantified in the 0-20kHz band.
    """

    audible_distortion_energy: float
    thdn_db: float


@dataclass(frozen=True)
class IMDProxyMetrics:
    """Comparison metrics between naive and NMSE outputs.

    Args:
        naive: IMD proxy metrics for naive/baseline signal.
        nmse: IMD proxy metrics for NMSE output signal.
        audible_distortion_reduction_db: Low-band distortion-energy reduction.
        thdn_improvement_db: THD+N improvement in dB (positive is better).
        nmse_has_lower_imd: Whether NMSE has lower low-band distortion energy.
        thdn_improvement_over_10db: Whether THD+N improved by at least 10dB.

    Physical Basis:
        IMD risk is reduced when high-band artifacts are suppressed before
        nonlinear playback paths, lowering audible-band distortion products.
    """

    naive: IMDPathMetrics
    nmse: IMDPathMetrics
    audible_distortion_reduction_db: float
    thdn_improvement_db: float
    nmse_has_lower_imd: bool
    thdn_improvement_over_10db: bool


def apply_soft_clipping(
    signal: np.ndarray, drive: float = DEFAULT_CLIP_DRIVE
) -> np.ndarray:
    """Apply smooth saturation to simulate mild analog nonlinearity.

    Args:
        signal: Input 1D signal.
        drive: Saturation drive (>0). Larger values increase nonlinearity.

    Returns:
        Soft-clipped signal in approximately [-1, 1].

    Raises:
        ValueError: If arguments are invalid.

    Physical Basis:
        Mild analog transfer nonlinearities can be approximated with tanh,
        producing harmonics and intermodulation from high-band components.
    """
    _validate_signal(signal, "signal")
    _validate_positive_float(drive, "drive")

    clipped = np.tanh(drive * np.asarray(signal, dtype=np.float64))
    return np.asarray(clipped, dtype=np.float64)


def evaluate_imd_path(
    signal: np.ndarray,
    sample_rate: int,
    clip_drive: float = DEFAULT_CLIP_DRIVE,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    num_taps: int = DEFAULT_NUM_TAPS,
) -> IMDPathMetrics:
    """Evaluate IMD proxy metrics for a single signal path.

    Args:
        signal: Candidate signal in Stage 1 domain.
        sample_rate: Sample rate in Hz.
        clip_drive: Soft-clipping drive for nonlinear simulation.
        cutoff_hz: Low/high split cutoff frequency in Hz.
        num_taps: FIR taps used for low-band extraction.

    Returns:
        IMDPathMetrics for the given signal.

    Raises:
        ValueError: If arguments are invalid.

    Physical Basis:
        Comparing low-band content before/after nonlinearity estimates how much
        high-band energy folds into the audible region through IMD.
    """
    _validate_signal(signal, "signal")
    _validate_sample_rate(sample_rate)
    _validate_positive_float(cutoff_hz, "cutoff_hz")
    _validate_positive_int(num_taps, "num_taps")
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd.")
    if cutoff_hz >= sample_rate / 2.0:
        raise ValueError("cutoff_hz must be below Nyquist.")
    if signal.size <= num_taps:
        raise ValueError(
            f"signal length must be greater than num_taps ({num_taps}), got {signal.size}."
        )

    clipped = apply_soft_clipping(signal, drive=clip_drive)
    lowpass_taps, highpass_taps = design_band_split_filters(
        cutoff_hz=cutoff_hz,
        sample_rate=sample_rate,
        num_taps=num_taps,
    )
    low_band_input, _ = band_split(signal, lowpass_taps, highpass_taps)
    low_band_clipped, _ = band_split(clipped, lowpass_taps, highpass_taps)

    trimmed_input = _trim_filter_warmup(low_band_input, num_taps)
    trimmed_clipped = _trim_filter_warmup(low_band_clipped, num_taps)

    distortion = _compute_linear_residual(trimmed_input, trimmed_clipped)
    distortion_energy = float(np.mean(np.square(distortion)))
    thdn_db = _compute_thdn_like_db(trimmed_input, trimmed_clipped)

    return IMDPathMetrics(
        audible_distortion_energy=distortion_energy,
        thdn_db=thdn_db,
    )


def evaluate_imd_proxy(
    naive_signal: np.ndarray,
    nmse_signal: np.ndarray,
    sample_rate: int,
    clip_drive: float = DEFAULT_CLIP_DRIVE,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    num_taps: int = DEFAULT_NUM_TAPS,
) -> IMDProxyMetrics:
    """Compare IMD proxy risk between naive and NMSE outputs.

    Args:
        naive_signal: Baseline signal (e.g., naive upsampling output).
        nmse_signal: NMSE output signal.
        sample_rate: Sample rate in Hz.
        clip_drive: Soft-clipping drive for nonlinear simulation.
        cutoff_hz: Low/high split cutoff frequency in Hz.
        num_taps: FIR taps used for low-band extraction.

    Returns:
        IMDProxyMetrics containing per-path values and acceptance booleans.

    Raises:
        ValueError: If arguments are invalid.

    Physical Basis:
        Both signals are passed through the same mild nonlinearity. Lower
        post-nonlinear low-band distortion indicates lower IMD risk.
    """
    _validate_signal_pair(naive_signal, nmse_signal)

    naive_metrics = evaluate_imd_path(
        signal=naive_signal,
        sample_rate=sample_rate,
        clip_drive=clip_drive,
        cutoff_hz=cutoff_hz,
        num_taps=num_taps,
    )
    nmse_metrics = evaluate_imd_path(
        signal=nmse_signal,
        sample_rate=sample_rate,
        clip_drive=clip_drive,
        cutoff_hz=cutoff_hz,
        num_taps=num_taps,
    )

    distortion_reduction_db = float(
        10.0
        * np.log10(
            (naive_metrics.audible_distortion_energy + EPSILON)
            / (nmse_metrics.audible_distortion_energy + EPSILON)
        )
    )
    thdn_improvement_db = float(naive_metrics.thdn_db - nmse_metrics.thdn_db)

    return IMDProxyMetrics(
        naive=naive_metrics,
        nmse=nmse_metrics,
        audible_distortion_reduction_db=distortion_reduction_db,
        thdn_improvement_db=thdn_improvement_db,
        nmse_has_lower_imd=(
            nmse_metrics.audible_distortion_energy
            < naive_metrics.audible_distortion_energy
        ),
        thdn_improvement_over_10db=thdn_improvement_db > 10.0,
    )


def _compute_thdn_like_db(reference: np.ndarray, measured: np.ndarray) -> float:
    residual = _compute_linear_residual(reference, measured)
    gain = _compute_best_fit_gain(reference, measured)
    linear_estimate = gain * reference

    residual_rms = float(np.sqrt(np.mean(np.square(residual))))
    linear_rms = float(np.sqrt(np.mean(np.square(linear_estimate))))
    return float(20.0 * np.log10((residual_rms + EPSILON) / (linear_rms + EPSILON)))


def _compute_linear_residual(reference: np.ndarray, measured: np.ndarray) -> np.ndarray:
    gain = _compute_best_fit_gain(reference, measured)
    linear_estimate = gain * reference
    return np.asarray(measured - linear_estimate, dtype=np.float64)


def _compute_best_fit_gain(reference: np.ndarray, measured: np.ndarray) -> float:
    numerator = float(np.dot(reference, measured))
    denominator = float(np.dot(reference, reference))
    return float(numerator / (denominator + EPSILON))


def _trim_filter_warmup(signal: np.ndarray, num_taps: int) -> np.ndarray:
    return np.asarray(signal[num_taps - 1 :], dtype=np.float64)


def _validate_signal_pair(naive_signal: np.ndarray, nmse_signal: np.ndarray) -> None:
    _validate_signal(naive_signal, "naive_signal")
    _validate_signal(nmse_signal, "nmse_signal")
    if naive_signal.shape != nmse_signal.shape:
        raise ValueError(
            "naive_signal and nmse_signal must have identical shapes. "
            f"Got {naive_signal.shape} and {nmse_signal.shape}."
        )


def _validate_signal(signal: np.ndarray, name: str) -> None:
    if signal.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if signal.size == 0:
        raise ValueError(f"{name} cannot be empty.")


def _validate_sample_rate(sample_rate: int) -> None:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")


def _validate_positive_float(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


__all__ = [
    "IMDPathMetrics",
    "IMDProxyMetrics",
    "apply_soft_clipping",
    "evaluate_imd_path",
    "evaluate_imd_proxy",
]
