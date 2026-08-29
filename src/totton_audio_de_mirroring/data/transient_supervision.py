"""Physical helpers for source-rate transient supervision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransientSupervisionConfig:
    """Configure event-focused transient supervision.

    Args:
        enabled: Enable event-focused chunk extraction and masks.
        focus_signal_types: Families whose event must enter every chunk.
        clean_probability: Fraction using no noise or clipping.
        center_fraction_range: Allowed event-center fractions.
        context_ms: Context retained on both sides of the event.
        pre_echo_guard_ms: Gap before the measured pre-echo window.
        pre_echo_window_ms: Duration of the measured pre-echo window.
        edge_supervision_signal_types: Families with physical edge labels.

    Physical Basis:
        Sparse clicks and bursts cannot teach pre-echo behavior when random
        chunking discards the event. Focused extraction guarantees exposure,
        while clean and noisy views separate absolute silence supervision
        from robustness to low-level contamination.
    """

    enabled: bool = False
    focus_signal_types: tuple[str, ...] = ("isolated_click", "tone_burst")
    clean_probability: float = 0.7
    center_fraction_range: tuple[float, float] = (0.2, 0.8)
    context_ms: float = 5.0
    pre_echo_guard_ms: float = 0.5
    pre_echo_window_ms: float = 3.5
    edge_supervision_signal_types: tuple[str, ...] = (
        "square_wave",
        "step_plateau",
        "isolated_click",
    )

    def __post_init__(self) -> None:
        """Validate transient-supervision settings."""
        low, high = self.center_fraction_range
        if not 0.0 <= self.clean_probability <= 1.0:
            raise ValueError("clean_probability must be in [0, 1].")
        if not np.isfinite((low, high)).all() or not 0.0 <= low <= high <= 1.0:
            raise ValueError("center_fraction_range must stay within [0, 1].")
        if self.context_ms < self.pre_echo_guard_ms + self.pre_echo_window_ms:
            raise ValueError("context_ms must cover the complete pre-echo window.")
        if self.pre_echo_guard_ms < 0.0 or self.pre_echo_window_ms <= 0.0:
            raise ValueError("Pre-echo guard/window values must be valid.")
        if not self.focus_signal_types:
            raise ValueError("focus_signal_types must not be empty.")


def cardinal_upsample(signal: np.ndarray, ratio: int) -> np.ndarray:
    """Band-limit a source waveform while preserving its sample lattice.

    Args:
        signal: Non-empty, one-dimensional source-rate waveform.
        ratio: Integer interpolation ratio greater than one.

    Returns:
        Periodically band-limited waveform with ``ratio`` times more samples.

    Physical Basis:
        Fourier zero-padding is an exact periodic cardinal interpolation:
        the original source appears unchanged at every ``ratio``-th output
        sample and no spectrum is placed above the source Nyquist frequency.
        This exposes true source-rate impulses without inventing unavailable
        high-frequency information.
    """
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("signal must be a non-empty 1D array.")
    if ratio <= 1:
        raise ValueError("ratio must be greater than one.")
    source_spectrum = np.fft.rfft(np.asarray(signal, dtype=np.float64))
    target_size = signal.size * ratio
    target_spectrum = np.zeros(target_size // 2 + 1, dtype=np.complex128)
    scale = float(ratio)
    if signal.size % 2 == 0:
        source_nyquist = signal.size // 2
        target_spectrum[:source_nyquist] = scale * source_spectrum[:source_nyquist]
        target_spectrum[source_nyquist] = 0.5 * scale * source_spectrum[source_nyquist]
    else:
        target_spectrum[: source_spectrum.size] = scale * source_spectrum
    result = np.asarray(np.fft.irfft(target_spectrum, n=target_size), dtype=np.float64)
    if not np.allclose(result[::ratio], signal, atol=1.0e-12, rtol=1.0e-12):
        raise RuntimeError("Cardinal interpolation failed source-lattice parity.")
    return result


def find_event_bounds(clean_signal: np.ndarray) -> tuple[int, int]:
    """Return the active event's half-open sample bounds."""
    if clean_signal.ndim != 1 or clean_signal.size == 0:
        raise ValueError("clean_signal must be a non-empty 1D array.")
    peak = float(np.max(np.abs(clean_signal)))
    active = np.flatnonzero(np.abs(clean_signal) > max(peak * 1.0e-6, 1.0e-12))
    if active.size == 0:
        raise ValueError("Focused transient generator produced no event.")
    return int(active[0]), int(active[-1] + 1)


def compute_pre_echo_mask(
    num_samples: int,
    event_start: int,
    sample_rate: int,
    guard_ms: float = 0.5,
    window_ms: float = 3.5,
) -> np.ndarray:
    """Mark the gate-aligned silence immediately before a transient.

    Physical Basis:
        The G2b gate measures mean-square energy from 4.0 to 0.5 ms before
        an event. Using the identical interval makes the differentiable loss
        respond to the same pre-ringing that determines acceptance.
    """
    if num_samples <= 0 or sample_rate <= 0:
        raise ValueError("num_samples and sample_rate must be positive.")
    if not 0 <= event_start < num_samples:
        raise ValueError("event_start must lie inside the chunk.")
    if guard_ms < 0.0 or window_ms <= 0.0:
        raise ValueError("guard_ms/window_ms must define a positive window.")
    guard = int(round(guard_ms * sample_rate / 1_000.0))
    window = int(round(window_ms * sample_rate / 1_000.0))
    stop = max(0, event_start - guard)
    start = max(0, stop - window)
    mask = np.zeros(num_samples, dtype=np.float64)
    mask[start:stop] = 1.0
    return mask
