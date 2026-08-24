"""Deterministic reference sample-rate conversion for CAPB evaluation."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def upsample_bessel_reference(
    signal: np.ndarray,
    source_sr: int,
    target_sr: int,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    """Upsample a waveform through the Bessel comparison path.

    Args:
        signal: Non-empty mono or channel-first waveform.
        source_sr: Source sample rate in Hz.
        target_sr: Integer-multiple target sample rate in Hz.
        cutoff_hz: Bessel low-pass cutoff at the target rate.
        order: Positive Bessel filter order.

    Returns:
        A new float64 waveform at ``target_sr``.

    Raises:
        ValueError: If an argument violates the SRC contract.

    Physical Basis:
        CAPB uses the gradual Bessel response only as its no-added-ringing
        comparison reference. Zero stuffing followed by a phase-normalized
        Bessel IIR reproduces that fixed baseline without entering CAPB's
        learned interpolation path.
    """
    values = np.asarray(signal)
    if values.ndim not in (1, 2) or values.size == 0:
        raise ValueError("signal must be a non-empty 1D or 2D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values.")
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive.")
    ratio_float = target_sr / source_sr
    if target_sr <= source_sr or not ratio_float.is_integer():
        raise ValueError("target_sr must be an integer multiple of source_sr.")
    if not 0.0 < cutoff_hz < target_sr / 2.0:
        raise ValueError("cutoff_hz must lie between zero and target Nyquist.")
    if order <= 0:
        raise ValueError("order must be positive.")

    ratio = int(ratio_float)
    output_shape = list(values.shape)
    output_shape[-1] *= ratio
    zero_stuffed = np.zeros(output_shape, dtype=np.float64)
    zero_stuffed[..., ::ratio] = np.asarray(values, dtype=np.float64)
    numerator, denominator = sp_signal.bessel(
        order,
        cutoff_hz,
        btype="lowpass",
        analog=False,
        output="ba",
        norm="phase",
        fs=target_sr,
    )
    filtered = sp_signal.lfilter(numerator, denominator, zero_stuffed, axis=-1)
    return np.asarray(filtered * ratio, dtype=np.float64)
