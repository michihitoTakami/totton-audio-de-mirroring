"""Low-band (0-20kHz) preservation metrics for Stage 1 evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from totton_audio_de_mirroring.data.filters import band_split, design_band_split_filters

DEFAULT_CUTOFF_HZ = 20_000.0
DEFAULT_NUM_TAPS = 1025


@dataclass(frozen=True)
class LowBandPreservationMetrics:
    """Low-band preservation metrics for one input/output signal pair.

    Args:
        waveform_error_db: Relative low-band waveform error in dB.
        waveform_mse: Mean squared error in low-band waveform.
        phase_error_deg: Mean absolute low-band phase error in degrees.
        group_delay_error_samples: Mean absolute low-band group-delay error.
        group_delay_error_ms: Mean absolute low-band group-delay error in ms.

    Physical Basis:
        Hard Requirement #1 requires preserving 0-20kHz waveform, phase, and
        group delay. These metrics quantify that requirement directly.
    """

    waveform_error_db: float
    waveform_mse: float
    phase_error_deg: float
    group_delay_error_samples: float
    group_delay_error_ms: float


def evaluate_lowband_preservation(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    num_taps: int = DEFAULT_NUM_TAPS,
) -> LowBandPreservationMetrics:
    """Evaluate low-band identity between input and output signals.

    Args:
        input_signal: Reference input signal at Stage 1 domain (88.2kHz).
        output_signal: Candidate output signal at the same sample rate.
        sample_rate: Sample rate in Hz.
        cutoff_hz: Low/high split cutoff in Hz.
        num_taps: Band-split FIR taps (odd number).

    Returns:
        LowBandPreservationMetrics for the pair.

    Raises:
        ValueError: If any argument is invalid.

    Physical Basis:
        The low-band is isolated through the same band-split process used by
        Stage 1 so the metrics match system behavior and guarantee checks.
    """
    _validate_signal_pair(input_signal, output_signal)
    _validate_sample_rate(sample_rate)
    _validate_cutoff(cutoff_hz, sample_rate)
    _validate_num_taps(num_taps)

    lowpass_taps, highpass_taps = design_band_split_filters(
        cutoff_hz=cutoff_hz,
        sample_rate=sample_rate,
        num_taps=num_taps,
    )
    low_band_input, _ = band_split(input_signal, lowpass_taps, highpass_taps)
    low_band_output, _ = band_split(output_signal, lowpass_taps, highpass_taps)

    waveform_error_db = _compute_lowband_waveform_error_db(
        low_band_input, low_band_output
    )
    waveform_mse = float(np.mean(np.square(low_band_output - low_band_input)))
    phase_error_deg = _compute_lowband_phase_error_deg(
        low_band_input,
        low_band_output,
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
    )
    group_delay_error_samples = _compute_lowband_group_delay_error_samples(
        low_band_input,
        low_band_output,
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
    )
    group_delay_error_ms = 1000.0 * group_delay_error_samples / float(sample_rate)

    return LowBandPreservationMetrics(
        waveform_error_db=waveform_error_db,
        waveform_mse=waveform_mse,
        phase_error_deg=phase_error_deg,
        group_delay_error_samples=group_delay_error_samples,
        group_delay_error_ms=group_delay_error_ms,
    )


def _compute_lowband_waveform_error_db(
    low_band_input: np.ndarray,
    low_band_output: np.ndarray,
) -> float:
    error_rms = float(np.sqrt(np.mean(np.square(low_band_output - low_band_input))))
    reference_rms = float(np.sqrt(np.mean(np.square(low_band_input))))
    ratio = error_rms / (reference_rms + 1.0e-12)
    return float(20.0 * np.log10(ratio + 1.0e-12))


def _compute_lowband_phase_error_deg(
    low_band_input: np.ndarray,
    low_band_output: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
) -> float:
    spectrum_in = np.fft.rfft(low_band_input)
    spectrum_out = np.fft.rfft(low_band_output)
    freqs = np.fft.rfftfreq(low_band_input.shape[-1], d=1.0 / sample_rate)

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
    low_band_input: np.ndarray,
    low_band_output: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
) -> float:
    spectrum_in = np.fft.rfft(low_band_input)
    spectrum_out = np.fft.rfft(low_band_output)
    freqs = np.fft.rfftfreq(low_band_input.shape[-1], d=1.0 / sample_rate)

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
        np.abs(spectrum_in[indices[:-1]]),
        np.abs(spectrum_in[indices[1:]]),
    )
    weighted_mean = np.sum(np.abs(group_delay) * weights) / (np.sum(weights) + 1.0e-12)
    return float(weighted_mean)


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


def _validate_num_taps(num_taps: int) -> None:
    if num_taps <= 0:
        raise ValueError(f"num_taps must be positive, got {num_taps}.")
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd.")
