"""Stage 1 hard metrics for mirror suppression evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.filters import band_split, design_band_split_filters
from totton_audio_de_mirroring.data.mirror_detection import (
    MirrorDetectionConfig,
    detect_mirror_artifacts,
)

DEFAULT_CUTOFF_HZ = 20_000.0
DEFAULT_MIRROR_BAND_HZ = (20_000.0, 22_050.0)
DEFAULT_ENERGY_CAP = 1.0e-3
DEFAULT_NUM_TAPS = 1025
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512


@dataclass(frozen=True)
class Stage1HardMetrics:
    """Single-pair evaluation metrics for Stage 1 output.

    Args:
        lb_amplitude_error_db: Relative low-band waveform error in dB.
        lb_phase_error_deg: Mean absolute low-band phase error in degrees.
        lb_group_delay_error_samples: Mean absolute low-band group delay error.
        mirror_reduction_ratio: Relative mirror-score reduction from input.
        hb_energy: Mean high-band time-domain energy.
        hb_energy_cap: Configured high-band energy cap.
        hb_energy_cap_violated: Whether high-band energy exceeded the cap.
        touch_metric: Non-mirror high-band deformation ratio.

    Physical Basis:
        Stage 1 must preserve 0-20kHz identity while suppressing mirror-like
        high-band components under a strict energy cap.
    """

    lb_amplitude_error_db: float
    lb_phase_error_deg: float
    lb_group_delay_error_samples: float
    mirror_reduction_ratio: float
    hb_energy: float
    hb_energy_cap: float
    hb_energy_cap_violated: bool
    touch_metric: float


@dataclass(frozen=True)
class SampleEvaluationResult:
    """Per-sample Stage 1 evaluation payload.

    Args:
        sample_id: Sample identifier, typically file stem.
        metrics: Computed Stage 1 hard metrics.

    Physical Basis:
        Per-sample metrics expose outliers that can be hidden by aggregate
        averages in mirror-suppression evaluations.
    """

    sample_id: str
    metrics: Stage1HardMetrics


@dataclass(frozen=True)
class DatasetEvaluationResult:
    """Dataset-level aggregate for Stage 1 hard metrics.

    Args:
        samples: Per-sample evaluation results.
        mean_metrics: Arithmetic mean of continuous metrics.
        hb_energy_cap_violation_rate: Fraction of samples violating the cap.

    Physical Basis:
        Hard requirements are dataset properties; aggregate statistics and
        violation rates are both required to characterize safety/performance.
    """

    samples: tuple[SampleEvaluationResult, ...]
    mean_metrics: Stage1HardMetrics
    hb_energy_cap_violation_rate: float


def evaluate_stage1_hard_metrics(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    energy_cap: float = DEFAULT_ENERGY_CAP,
    num_taps: int = DEFAULT_NUM_TAPS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    mirror_band_hz: tuple[float, float] = DEFAULT_MIRROR_BAND_HZ,
) -> Stage1HardMetrics:
    """Evaluate README hard metrics for a single input/output pair.

    Args:
        input_signal: Stage 1 input signal at 88.2kHz domain.
        output_signal: Stage 1 output signal at the same sample rate.
        sample_rate: Sample rate in Hz.
        cutoff_hz: Band-split crossover frequency in Hz.
        energy_cap: Maximum allowed high-band mean energy.
        num_taps: Number of taps for LPF/HPF split filters.
        n_fft: STFT FFT size.
        hop_length: STFT hop size.
        mirror_band_hz: Mirror detection band in Hz.

    Returns:
        Stage1HardMetrics for the pair.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Metrics map directly to README hard requirements: low-band identity,
        mirror reduction, high-band energy cap, and touch minimization.
    """
    _validate_signal_pair(input_signal, output_signal)
    _validate_sample_rate(sample_rate)
    _validate_cutoff(cutoff_hz, sample_rate)
    _validate_positive_float(energy_cap, "energy_cap")
    _validate_stft_params(n_fft, hop_length)
    _validate_mirror_band(mirror_band_hz, sample_rate)

    lowpass_taps, highpass_taps = design_band_split_filters(
        cutoff_hz=cutoff_hz,
        sample_rate=sample_rate,
        num_taps=num_taps,
    )
    lb_in, hb_in = band_split(input_signal, lowpass_taps, highpass_taps)
    lb_out, hb_out = band_split(output_signal, lowpass_taps, highpass_taps)

    lb_amplitude_error_db = _compute_lowband_amplitude_error_db(lb_in, lb_out)
    lb_phase_error_deg = _compute_lowband_phase_error_deg(
        lb_in, lb_out, sample_rate, cutoff_hz
    )
    lb_group_delay_error_samples = _compute_lowband_group_delay_error_samples(
        lb_in, lb_out, sample_rate, cutoff_hz
    )

    mirror_score_in, mirror_mask = _compute_mirror_score(
        hb_in,
        sample_rate,
        cutoff_hz=cutoff_hz,
        mirror_band_hz=mirror_band_hz,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mirror_score_out, _ = _compute_mirror_score(
        hb_out,
        sample_rate,
        cutoff_hz=cutoff_hz,
        mirror_band_hz=mirror_band_hz,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mirror_reduction_ratio = _compute_relative_reduction(
        mirror_score_in, mirror_score_out
    )

    hb_energy = float(np.mean(np.square(hb_out)))
    hb_energy_cap_violated = hb_energy > energy_cap

    touch_metric = _compute_touch_metric(
        hb_in,
        hb_out,
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        n_fft=n_fft,
        hop_length=hop_length,
        mirror_mask=mirror_mask,
    )

    return Stage1HardMetrics(
        lb_amplitude_error_db=lb_amplitude_error_db,
        lb_phase_error_deg=lb_phase_error_deg,
        lb_group_delay_error_samples=lb_group_delay_error_samples,
        mirror_reduction_ratio=mirror_reduction_ratio,
        hb_energy=hb_energy,
        hb_energy_cap=energy_cap,
        hb_energy_cap_violated=hb_energy_cap_violated,
        touch_metric=touch_metric,
    )


def evaluate_dataset(
    samples: list[tuple[str, np.ndarray, np.ndarray]],
    sample_rate: int,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    energy_cap: float = DEFAULT_ENERGY_CAP,
    num_taps: int = DEFAULT_NUM_TAPS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    mirror_band_hz: tuple[float, float] = DEFAULT_MIRROR_BAND_HZ,
) -> DatasetEvaluationResult:
    """Evaluate Stage 1 metrics for a dataset of paired signals.

    Args:
        samples: List of (sample_id, input_signal, output_signal).
        sample_rate: Sample rate in Hz.
        cutoff_hz: Band-split crossover frequency in Hz.
        energy_cap: Maximum allowed high-band mean energy.
        num_taps: Number of taps for LPF/HPF split filters.
        n_fft: STFT FFT size.
        hop_length: STFT hop size.
        mirror_band_hz: Mirror detection band in Hz.

    Returns:
        DatasetEvaluationResult with per-sample and aggregate metrics.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Dataset evaluation is required to verify hard constraints hold beyond
        one-off examples and detect rare but unsafe cap violations.
    """
    if len(samples) == 0:
        raise ValueError("samples cannot be empty.")

    sample_results: list[SampleEvaluationResult] = []
    for sample_id, input_signal, output_signal in samples:
        metrics = evaluate_stage1_hard_metrics(
            input_signal=input_signal,
            output_signal=output_signal,
            sample_rate=sample_rate,
            cutoff_hz=cutoff_hz,
            energy_cap=energy_cap,
            num_taps=num_taps,
            n_fft=n_fft,
            hop_length=hop_length,
            mirror_band_hz=mirror_band_hz,
        )
        sample_results.append(
            SampleEvaluationResult(sample_id=sample_id, metrics=metrics)
        )

    return _aggregate_results(sample_results)


def sample_result_to_flat_dict(
    result: SampleEvaluationResult,
) -> dict[str, float | bool | str]:
    """Convert per-sample result to flat dictionary for CSV/JSON output.

    Args:
        result: Sample-level evaluation result.

    Returns:
        Flat dictionary containing sample id and all metrics.

    Physical Basis:
        Flat records simplify downstream statistics/plotting workflows.
    """
    flat: dict[str, float | bool | str] = {"sample_id": result.sample_id}
    metric_fields = asdict(result.metrics)
    flat.update(metric_fields)
    return flat


def _aggregate_results(
    samples: list[SampleEvaluationResult],
) -> DatasetEvaluationResult:
    metric_matrix = np.array(
        [
            [
                sample.metrics.lb_amplitude_error_db,
                sample.metrics.lb_phase_error_deg,
                sample.metrics.lb_group_delay_error_samples,
                sample.metrics.mirror_reduction_ratio,
                sample.metrics.hb_energy,
                sample.metrics.touch_metric,
            ]
            for sample in samples
        ],
        dtype=np.float64,
    )
    mean_values = np.mean(metric_matrix, axis=0)
    violation_rate = float(
        np.mean([sample.metrics.hb_energy_cap_violated for sample in samples])
    )
    mean_metrics = Stage1HardMetrics(
        lb_amplitude_error_db=float(mean_values[0]),
        lb_phase_error_deg=float(mean_values[1]),
        lb_group_delay_error_samples=float(mean_values[2]),
        mirror_reduction_ratio=float(mean_values[3]),
        hb_energy=float(mean_values[4]),
        hb_energy_cap=float(samples[0].metrics.hb_energy_cap),
        hb_energy_cap_violated=violation_rate > 0.0,
        touch_metric=float(mean_values[5]),
    )
    return DatasetEvaluationResult(
        samples=tuple(samples),
        mean_metrics=mean_metrics,
        hb_energy_cap_violation_rate=violation_rate,
    )


def _compute_lowband_amplitude_error_db(
    lb_in: np.ndarray,
    lb_out: np.ndarray,
) -> float:
    error_rms = float(np.sqrt(np.mean(np.square(lb_out - lb_in))))
    reference_rms = float(np.sqrt(np.mean(np.square(lb_in))))
    ratio = error_rms / (reference_rms + 1.0e-12)
    return float(20.0 * np.log10(ratio + 1.0e-12))


def _compute_lowband_phase_error_deg(
    lb_in: np.ndarray,
    lb_out: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
) -> float:
    spectrum_in = np.fft.rfft(lb_in)
    spectrum_out = np.fft.rfft(lb_out)
    freqs = np.fft.rfftfreq(lb_in.shape[-1], d=1.0 / sample_rate)

    mask = (freqs > 0.0) & (freqs <= cutoff_hz)
    mask = mask & (np.abs(spectrum_in) > 1.0e-10)
    if not np.any(mask):
        return 0.0

    transfer = spectrum_out[mask] / (spectrum_in[mask] + 1.0e-12)
    phase_diff = np.angle(transfer)
    weights = np.abs(spectrum_in[mask])
    weighted_mean = np.sum(np.abs(phase_diff) * weights) / (np.sum(weights) + 1.0e-12)
    return float(np.rad2deg(weighted_mean))


def _compute_lowband_group_delay_error_samples(
    lb_in: np.ndarray,
    lb_out: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
) -> float:
    spectrum_in = np.fft.rfft(lb_in)
    spectrum_out = np.fft.rfft(lb_out)
    freqs = np.fft.rfftfreq(lb_in.shape[-1], d=1.0 / sample_rate)

    mask = (freqs > 0.0) & (freqs <= cutoff_hz)
    mask = mask & (np.abs(spectrum_in) > 1.0e-10)
    indices = np.where(mask)[0]
    if indices.size < 3:
        return 0.0

    transfer = spectrum_out[indices] / (spectrum_in[indices] + 1.0e-12)
    phase = np.unwrap(np.angle(transfer))
    omega = 2.0 * np.pi * freqs[indices] / sample_rate
    group_delay = -np.diff(phase) / (np.diff(omega) + 1.0e-12)
    weights = np.minimum(
        np.abs(spectrum_in[indices[:-1]]), np.abs(spectrum_in[indices[1:]])
    )
    weighted_mean = np.sum(np.abs(group_delay) * weights) / (np.sum(weights) + 1.0e-12)
    return float(weighted_mean)


def _compute_mirror_score(
    hb_signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    mirror_band_hz: tuple[float, float],
    n_fft: int,
    hop_length: int,
) -> tuple[float, np.ndarray]:
    detection = detect_mirror_artifacts(
        hb_signal,
        sample_rate=sample_rate,
        config=MirrorDetectionConfig(
            cutoff_hz=cutoff_hz,
            mirror_band_hz=mirror_band_hz,
            n_fft=n_fft,
            hop_length=hop_length,
        ),
    )
    freq_mask = (detection.freqs >= mirror_band_hz[0]) & (
        detection.freqs <= mirror_band_hz[1]
    )
    band_magnitude = detection.magnitude[freq_mask, :]
    score = float(np.sum(np.abs(band_magnitude)))
    return score, detection.detection_mask


def _compute_relative_reduction(before: float, after: float) -> float:
    if before <= 1.0e-12:
        return 0.0
    return float((before - after) / before)


def _compute_touch_metric(
    hb_in: np.ndarray,
    hb_out: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    n_fft: int,
    hop_length: int,
    mirror_mask: np.ndarray,
) -> float:
    freqs, _, stft_in = _stft(hb_in, sample_rate, n_fft, hop_length)
    _, _, stft_out = _stft(hb_out, sample_rate, n_fft, hop_length)
    mag_in = np.abs(stft_in)
    mag_out = np.abs(stft_out)

    highband_mask = freqs >= cutoff_hz
    non_mirror = highband_mask[:, None] & (~mirror_mask)

    if not np.any(non_mirror):
        return 0.0

    deformation = np.abs(mag_out[non_mirror] - mag_in[non_mirror])
    baseline = np.abs(mag_in[non_mirror])
    return float(np.sum(deformation) / (np.sum(baseline) + 1.0e-12))


def _stft(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    noverlap = n_fft - hop_length
    freqs, times, stft = sp_signal.stft(
        signal,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=noverlap,
        window="hann",
        boundary="zeros",
        padded=True,
    )
    return (
        np.asarray(freqs, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        np.asarray(stft, dtype=np.complex128),
    )


def _validate_signal_pair(input_signal: np.ndarray, output_signal: np.ndarray) -> None:
    if input_signal.ndim != 1 or output_signal.ndim != 1:
        raise ValueError("input_signal and output_signal must be 1D arrays.")
    if input_signal.size == 0 or output_signal.size == 0:
        raise ValueError("input_signal and output_signal cannot be empty.")
    if input_signal.shape != output_signal.shape:
        raise ValueError(
            "input_signal and output_signal must have identical shapes. "
            f"Got {input_signal.shape} and {output_signal.shape}."
        )


def _validate_sample_rate(sample_rate: int) -> None:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")


def _validate_cutoff(cutoff_hz: float, sample_rate: int) -> None:
    if cutoff_hz <= 0.0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}.")
    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz must be less than Nyquist ({nyquist}).")


def _validate_positive_float(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}.")


def _validate_stft_params(n_fft: int, hop_length: int) -> None:
    if n_fft <= 0:
        raise ValueError(f"n_fft must be positive, got {n_fft}.")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}.")
    if hop_length >= n_fft:
        raise ValueError("hop_length must be smaller than n_fft.")


def _validate_mirror_band(
    mirror_band_hz: tuple[float, float], sample_rate: int
) -> None:
    lower, upper = mirror_band_hz
    if lower <= 0.0:
        raise ValueError(f"mirror_band lower must be positive, got {lower}.")
    if upper <= lower:
        raise ValueError("mirror_band upper must be greater than lower.")
    nyquist = sample_rate / 2.0
    if upper > nyquist:
        raise ValueError(
            f"mirror_band upper must be <= Nyquist ({nyquist}), got {upper}."
        )
