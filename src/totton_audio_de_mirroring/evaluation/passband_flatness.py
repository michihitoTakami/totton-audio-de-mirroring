"""Passband flatness and gain-accuracy metrics for Stage 1 evaluation.

Detects audible-band notches/boosts (e.g. the historical ~13 kHz notch) and
systematic level errors (e.g. the 3.89x volume loss seen in a zero-stuff
experiment) by comparing smoothed spectral densities between a reference and
a candidate signal, which may live at different sample rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

DEFAULT_BAND_HZ = (100.0, 18_000.0)
DEFAULT_HF_BAND_HZ = (18_000.0, 20_000.0)
DEFAULT_SMOOTHING_OCTAVES = 1.0 / 6.0
SEGMENT_DURATION_SEC = 8_192 / 88_200
DEFAULT_LB_CUTOFF_HZ = 19_000.0
_EPSILON = 1e-300


@dataclass(frozen=True)
class FlatnessMetrics:
    """Smoothed response deviation between candidate and reference.

    Args:
        max_dip_db: Largest negative deviation in the main band (<= 0).
        max_boost_db: Largest positive deviation in the main band (>= 0).
        hf_max_dip_db: Largest negative deviation in the HF band.
        band_hz: Main analysis band in Hz.
        hf_band_hz: High-frequency analysis band in Hz.

    Physical Basis:
        A 1/6-octave smoothed spectral ratio approximates perceived tonal
        balance; localized dips reveal notches introduced by suppression
        targets, and boosts reveal image/mirror energy folding into band.
    """

    max_dip_db: float
    max_boost_db: float
    hf_max_dip_db: float
    band_hz: tuple[float, float]
    hf_band_hz: tuple[float, float]


def compute_flatness(
    reference: np.ndarray,
    reference_sample_rate: int,
    candidate: np.ndarray,
    candidate_sample_rate: int,
    band_hz: tuple[float, float] = DEFAULT_BAND_HZ,
    hf_band_hz: tuple[float, float] = DEFAULT_HF_BAND_HZ,
    smoothing_octaves: float = DEFAULT_SMOOTHING_OCTAVES,
) -> FlatnessMetrics:
    """Compute smoothed response deviation of candidate vs reference.

    Args:
        reference: Reference waveform (e.g. the 44.1 kHz source probe).
        reference_sample_rate: Reference sample rate in Hz.
        candidate: Candidate waveform (e.g. the 88.2 kHz system output).
        candidate_sample_rate: Candidate sample rate in Hz.
        band_hz: Main analysis band in Hz.
        hf_band_hz: High-frequency analysis band in Hz.
        smoothing_octaves: Smoothing window width in octaves.

    Returns:
        FlatnessMetrics with worst dip/boost in each band.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Welch power spectral densities are per-Hz quantities, so densities
        estimated at different sample rates are directly comparable on a
        common frequency grid; an ideal upsampler yields a 0 dB ratio.
    """
    _validate_signal(reference)
    _validate_signal(candidate)
    if reference_sample_rate <= 0 or candidate_sample_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    if not 0.0 < band_hz[0] < band_hz[1]:
        raise ValueError(f"Invalid band_hz: {band_hz}.")

    grid_hz = _log_grid(band_hz[0], hf_band_hz[1])
    ref_db = _smoothed_psd_db(
        reference, reference_sample_rate, grid_hz, smoothing_octaves
    )
    cand_db = _smoothed_psd_db(
        candidate, candidate_sample_rate, grid_hz, smoothing_octaves
    )
    deviation_db = cand_db - ref_db

    main = (grid_hz >= band_hz[0]) & (grid_hz <= band_hz[1])
    high = (grid_hz >= hf_band_hz[0]) & (grid_hz <= hf_band_hz[1])
    return FlatnessMetrics(
        max_dip_db=float(min(np.min(deviation_db[main]), 0.0)),
        max_boost_db=float(max(np.max(deviation_db[main]), 0.0)),
        hf_max_dip_db=float(min(np.min(deviation_db[high]), 0.0))
        if np.any(high)
        else 0.0,
        band_hz=band_hz,
        hf_band_hz=hf_band_hz,
    )


def compute_lowband_gain_error_db(
    reference: np.ndarray,
    reference_sample_rate: int,
    candidate: np.ndarray,
    candidate_sample_rate: int,
    cutoff_hz: float = DEFAULT_LB_CUTOFF_HZ,
) -> float:
    """Compute low-band RMS gain error of candidate vs reference in dB.

    Args:
        reference: Reference waveform.
        reference_sample_rate: Reference sample rate in Hz.
        candidate: Candidate waveform.
        candidate_sample_rate: Candidate sample rate in Hz.
        cutoff_hz: Low-band upper edge in Hz.

    Returns:
        Gain error in dB (0.0 means exact level match).

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Comparing band-limited RMS across rates is level-invariant to the
        upsampling ratio and catches systematic volume loss immediately.
    """
    _validate_signal(reference)
    _validate_signal(candidate)
    ref_rms = _lowband_rms(reference, reference_sample_rate, cutoff_hz)
    cand_rms = _lowband_rms(candidate, candidate_sample_rate, cutoff_hz)
    return float(20.0 * np.log10(max(cand_rms, _EPSILON) / max(ref_rms, _EPSILON)))


def _lowband_rms(signal: np.ndarray, sample_rate: int, cutoff_hz: float) -> float:
    if cutoff_hz >= sample_rate / 2:
        low = np.asarray(signal, dtype=np.float64)
    else:
        taps = sp_signal.firwin(1025, cutoff_hz, fs=sample_rate)
        low = sp_signal.fftconvolve(
            np.asarray(signal, dtype=np.float64), taps, mode="same"
        )
    return float(np.sqrt(np.mean(np.square(low))))


def _smoothed_psd_db(
    signal: np.ndarray,
    sample_rate: int,
    grid_hz: np.ndarray,
    smoothing_octaves: float,
) -> np.ndarray:
    # Fix the segment DURATION (not the sample count) so that reference and
    # candidate at different rates window the same time spans; the shared
    # noise realization's periodogram fluctuations then cancel in the ratio.
    nperseg = min(int(round(SEGMENT_DURATION_SEC * sample_rate)), signal.size)
    freqs, psd = sp_signal.welch(
        np.asarray(signal, dtype=np.float64),
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )
    smoothed = np.empty_like(grid_hz)
    half_width = 2.0 ** (smoothing_octaves / 2.0)
    for index, center in enumerate(grid_hz):
        lo, hi = center / half_width, center * half_width
        band = (freqs >= lo) & (freqs <= hi)
        if not np.any(band):
            band = np.array([int(np.argmin(np.abs(freqs - center)))])
        smoothed[index] = np.mean(psd[band])
    return np.asarray(10.0 * np.log10(np.maximum(smoothed, _EPSILON)), dtype=np.float64)


def _log_grid(low_hz: float, high_hz: float, points_per_octave: int = 12) -> np.ndarray:
    octaves = np.log2(high_hz / low_hz)
    count = max(2, int(round(octaves * points_per_octave)) + 1)
    return np.asarray(np.geomspace(low_hz, high_hz, count), dtype=np.float64)


def _validate_signal(signal: np.ndarray) -> None:
    array = np.asarray(signal)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("signal must be a non-empty 1D array.")
