"""Core audio filter utilities for band split and Bessel FIR design."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

DEFAULT_CUTOFF_HZ = 20_000.0
DEFAULT_FILTER_ORDER = 8
DEFAULT_NUM_TAPS = 4097
DEFAULT_WINDOW = "hamming"


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
