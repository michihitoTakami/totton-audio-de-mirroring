"""Core audio filter utilities for band split and Bessel FIR design."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal as sp_signal

DEFAULT_CUTOFF_HZ = 20_000.0
DEFAULT_FILTER_ORDER = 8
DEFAULT_NUM_TAPS = 4097
DEFAULT_WINDOW = "hamming"

# Transparent upsampler defaults: pass 0-20kHz, reject >=22.05kHz images.
DEFAULT_PASSBAND_HZ = 20_000.0
DEFAULT_STOPBAND_HZ = 22_050.0
DEFAULT_STOPBAND_DB = 180.0


def kaiser_params_for_stopband(
    stopband_db: float,
    transition_hz: float,
    sample_rate: int,
) -> tuple[int, float]:
    """Derive Kaiser FIR tap count and beta for a target stopband attenuation.

    Args:
        stopband_db: Desired stopband attenuation (and passband ripple) in dB.
        transition_hz: Transition-band width in Hz.
        sample_rate: Sample rate the FIR runs at (the upsampled rate) in Hz.

    Returns:
        Tuple of (num_taps, beta); num_taps is forced odd (Type I linear phase).

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        The Kaiser window design formulas (Kaiser/Oppenheim) map a desired
        stopband attenuation and transition width to the minimum tap count and
        beta. Targeting ~180 dB keeps both passband ripple and image leakage
        far below the 32-bit float floor (~-144 dB), so float32 arithmetic —
        not the filter — limits transparency.
    """
    _validate_sample_rate(sample_rate)
    if stopband_db <= 0.0:
        raise ValueError(f"stopband_db must be positive, got {stopband_db}.")
    if transition_hz <= 0.0:
        raise ValueError(f"transition_hz must be positive, got {transition_hz}.")
    nyquist = sample_rate / 2.0
    width_norm = transition_hz / nyquist
    if not 0.0 < width_norm < 1.0:
        raise ValueError("transition_hz must be within (0, Nyquist).")

    num_taps, beta = sp_signal.kaiserord(stopband_db, width_norm)
    if num_taps % 2 == 0:
        num_taps += 1
    return int(num_taps), float(beta)


def design_transparent_upsampler_fir(
    *,
    source_sr: int,
    ratio: int,
    passband_hz: float = DEFAULT_PASSBAND_HZ,
    stopband_hz: float = DEFAULT_STOPBAND_HZ,
    stopband_db: float = DEFAULT_STOPBAND_DB,
    num_taps: int | None = None,
) -> tuple[np.ndarray, float]:
    """Design a linear-phase Kaiser FIR anti-imaging filter for 2x upsampling.

    Args:
        source_sr: Input sample rate in Hz (e.g., 44100).
        ratio: Integer upsampling ratio (e.g., 2).
        passband_hz: Top of the flat passband to preserve (e.g., 20000).
        stopband_hz: Start of the image stopband to reject (e.g., 22050).
        stopband_db: Target stopband attenuation in dB.
        num_taps: Optional explicit tap count (odd); if None, derived from the
            Kaiser formula for the minimum transparent filter.

    Returns:
        Tuple of (taps, beta). Taps are gain-compensated by ``ratio`` so the
        zero-stuffed passband returns to unity gain.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Zero-stuffing by ``ratio`` creates spectral images above the source
        Nyquist; a sharp linear-phase low-pass removes them. A Kaiser window at
        ~180 dB makes the filter transparent below the 32-bit float floor while
        keeping the tap count (and thus the impulse-response/ringing length)
        minimal, so residual transient ringing stays local enough for a
        neural de-ringer to address.
    """
    _validate_positive_int(source_sr, "source_sr")
    _validate_positive_int(ratio, "ratio")
    if ratio < 2:
        raise ValueError(f"ratio must be >= 2, got {ratio}.")
    target_sr = source_sr * ratio
    if not 0.0 < passband_hz < stopband_hz <= target_sr / 2.0:
        raise ValueError("require 0 < passband_hz < stopband_hz <= target Nyquist.")

    transition_hz = stopband_hz - passband_hz
    beta = kaiser_params_for_stopband(stopband_db, transition_hz, target_sr)[1]
    if num_taps is None:
        num_taps = kaiser_params_for_stopband(stopband_db, transition_hz, target_sr)[0]
    _validate_positive_int(num_taps, "num_taps")
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd for linear-phase design.")

    cutoff_hz = 0.5 * (passband_hz + stopband_hz)
    taps = sp_signal.firwin(
        num_taps,
        cutoff_hz,
        window=("kaiser", beta),
        pass_zero="lowpass",
        fs=target_sr,
    )
    taps = np.asarray(taps, dtype=np.float64) * float(ratio)
    return taps, beta


@lru_cache(maxsize=8)
def _cached_transparent_taps(
    source_sr: int,
    ratio: int,
    passband_hz: float,
    stopband_hz: float,
    stopband_db: float,
    num_taps: int | None,
) -> tuple[np.ndarray, float]:
    return design_transparent_upsampler_fir(
        source_sr=source_sr,
        ratio=ratio,
        passband_hz=passband_hz,
        stopband_hz=stopband_hz,
        stopband_db=stopband_db,
        num_taps=num_taps,
    )


def upsample_transparent_reference(
    *,
    signal: np.ndarray,
    source_sr: int,
    target_sr: int,
    passband_hz: float = DEFAULT_PASSBAND_HZ,
    stopband_hz: float = DEFAULT_STOPBAND_HZ,
    stopband_db: float = DEFAULT_STOPBAND_DB,
    num_taps: int | None = None,
) -> np.ndarray:
    """Upsample via the 32-bit-transparent Kaiser FIR (image-free reconstruction).

    Args:
        signal: 1D input signal at ``source_sr``.
        source_sr: Source sample rate in Hz.
        target_sr: Target sample rate in Hz (integer multiple of source).
        passband_hz: Flat passband edge to preserve.
        stopband_hz: Image stopband edge to reject.
        stopband_db: Target stopband attenuation in dB.
        num_taps: Optional explicit odd tap count (else minimal derived).

    Returns:
        Upsampled signal at ``target_sr`` (float64), images rejected to well
        below the 32-bit float floor.

    Raises:
        ValueError: If the rate ratio is not a valid integer >= 2.

    Physical Basis:
        Drop-in replacement for the legacy Bessel IIR path; eliminates the
        mirror/image leakage at the source instead of suppressing it later.
    """
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive.")
    ratio = target_sr // source_sr
    if ratio < 2 or ratio * source_sr != target_sr:
        raise ValueError("target_sr must be an integer (>=2) multiple of source_sr.")
    taps, _ = _cached_transparent_taps(
        source_sr, ratio, passband_hz, stopband_hz, stopband_db, num_taps
    )
    return upsample_fir(np.asarray(signal, dtype=np.float64), ratio, taps)


def upsample_fir(
    signal: np.ndarray,
    ratio: int,
    taps: np.ndarray,
) -> np.ndarray:
    """Upsample a 1D signal by zero-stuffing and linear-phase FIR filtering.

    Args:
        signal: 1D input signal.
        ratio: Integer upsampling ratio.
        taps: Linear-phase FIR taps (odd length, gain-compensated).

    Returns:
        Upsampled signal of length ``len(signal) * ratio``, group-delay aligned.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        ``upfirdn`` performs the zero-stuff + convolution efficiently. Removing
        the linear-phase group delay ((num_taps-1)/2 samples) aligns the output
        to the ideal band-limited interpolation grid.
    """
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got {signal.ndim}D.")
    if signal.size == 0:
        raise ValueError("signal cannot be empty.")
    _validate_positive_int(ratio, "ratio")
    _validate_taps(taps)
    if taps.size % 2 == 0:
        raise ValueError("taps must be odd length for linear-phase alignment.")

    filtered = sp_signal.upfirdn(taps, np.asarray(signal, dtype=np.float64), up=ratio)
    delay = (taps.size - 1) // 2
    expected = signal.shape[-1] * ratio
    aligned = filtered[delay : delay + expected]
    if aligned.shape[-1] < expected:
        aligned = np.pad(aligned, (0, expected - aligned.shape[-1]))
    return np.asarray(aligned, dtype=np.float64)


def design_bessel_fir(
    cutoff_hz: float,
    sample_rate: int,
    order: int = DEFAULT_FILTER_ORDER,
    num_taps: int = DEFAULT_NUM_TAPS,
    window: str | tuple[str, float] = DEFAULT_WINDOW,
) -> np.ndarray:
    """Design a Bessel-based FIR low-pass filter.

    Args:
        cutoff_hz: Cutoff frequency in Hz.
        sample_rate: Sample rate in Hz.
        order: Bessel filter order (higher = flatter group delay).
        num_taps: Number of FIR taps.
        window: Window name for truncation.

    Returns:
        FIR filter taps (1D array).

    Raises:
        ValueError: If cutoff_hz or sample_rate or num_taps are invalid.

    Physical Basis:
        A Bessel prototype has maximally flat group delay in the passband,
        preserving transients. We sample the digital Bessel IIR impulse
        response and window it to obtain a practical FIR approximation.
    """
    _validate_sample_rate(sample_rate)
    _validate_cutoff(cutoff_hz, sample_rate)
    _validate_positive_int(order, "order")
    _validate_positive_int(num_taps, "num_taps")

    b, a = sp_signal.bessel(
        order,
        cutoff_hz,
        btype="lowpass",
        analog=False,
        output="ba",
        norm="phase",
        fs=sample_rate,
    )
    _, impulse = sp_signal.dimpulse((b, a, 1.0 / sample_rate), n=num_taps)
    impulse_response = np.squeeze(impulse[0]).astype(np.float64)

    window_values = sp_signal.get_window(window, num_taps, fftbins=False)
    taps = impulse_response * window_values

    tap_sum = float(np.sum(taps))
    if tap_sum == 0.0:
        raise ValueError("Bessel FIR taps sum to zero; check parameters.")
    taps = taps / tap_sum

    return np.asarray(taps, dtype=np.float64)


def design_band_split_filters(
    cutoff_hz: float,
    sample_rate: int,
    num_taps: int = DEFAULT_NUM_TAPS,
    window: str | tuple[str, float] = DEFAULT_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Design complementary linear-phase band-split filters.

    Args:
        cutoff_hz: Cutoff frequency in Hz.
        sample_rate: Sample rate in Hz.
        num_taps: Number of FIR taps (must be odd).
        window: Window name.

    Returns:
        Tuple of (lowpass_taps, highpass_taps).

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Linear-phase FIR filters preserve phase relationships within the
        passband. Using matched LPF/HPF taps yields near-complementary
        band splitting suitable for low-band identity preservation.
    """
    _validate_sample_rate(sample_rate)
    _validate_cutoff(cutoff_hz, sample_rate)
    _validate_positive_int(num_taps, "num_taps")
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd for complementary FIR design.")

    lowpass_taps = sp_signal.firwin(
        num_taps,
        cutoff_hz,
        pass_zero="lowpass",
        fs=sample_rate,
        window=window,
    )
    highpass_taps = sp_signal.firwin(
        num_taps,
        cutoff_hz,
        pass_zero="highpass",
        fs=sample_rate,
        window=window,
    )

    return lowpass_taps.astype(np.float64), highpass_taps.astype(np.float64)


def apply_fir_filter(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply an FIR filter to a 1D or 2D signal.

    Args:
        signal: Input signal (1D or 2D). Time axis must be last.
        taps: FIR filter taps.

    Returns:
        Filtered signal with the same shape as input.

    Raises:
        ValueError: If signal or taps are invalid.

    Physical Basis:
        FIR filtering performs convolution with fixed coefficients, allowing
        precise control of magnitude and phase characteristics.
    """
    _validate_signal(signal)
    _validate_taps(taps)

    filtered = sp_signal.lfilter(taps, [1.0], signal, axis=-1)
    return np.asarray(filtered)


def band_split(
    signal: np.ndarray,
    lowpass_taps: np.ndarray,
    highpass_taps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a signal into low-band and high-band components.

    Args:
        signal: Input signal (1D or 2D). Time axis must be last.
        lowpass_taps: FIR taps for low-pass filter.
        highpass_taps: FIR taps for high-pass filter.

    Returns:
        Tuple of (low_band, high_band).

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Low/high-pass decomposition isolates 0–20kHz content from
        20–44kHz content, enabling low-band bypass and high-band processing.
    """
    _validate_signal(signal)
    _validate_taps(lowpass_taps)
    _validate_taps(highpass_taps)
    if lowpass_taps.size != highpass_taps.size:
        raise ValueError("lowpass_taps and highpass_taps must be the same length.")

    low_band = apply_fir_filter(signal, lowpass_taps)
    high_band = apply_fir_filter(signal, highpass_taps)

    return low_band, high_band


def _validate_sample_rate(sample_rate: int) -> None:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")


def _validate_cutoff(cutoff_hz: float, sample_rate: int) -> None:
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}.")
    nyquist = sample_rate / 2
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"cutoff_hz must be less than Nyquist ({nyquist} Hz), got {cutoff_hz}."
        )


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_signal(signal: np.ndarray) -> None:
    if signal.ndim not in (1, 2):
        raise ValueError(f"signal must be 1D or 2D, got {signal.ndim}D.")
    if signal.size == 0:
        raise ValueError("signal cannot be empty.")


def _validate_taps(taps: np.ndarray) -> None:
    if taps.ndim != 1:
        raise ValueError("taps must be a 1D array.")
    if taps.size == 0:
        raise ValueError("taps cannot be empty.")
