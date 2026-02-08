"""Time-domain visualization for transient and phase analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class SquareWaveMetrics:
    """Square wave response metrics for ringing evaluation.

    Args:
        time_ms: Time axis in milliseconds.
        response: Step response waveform.
        overshoot_percent: Overshoot percentage.
        settling_time_ms: Time to settle within 5% of final value.
        has_ringing: Whether visible ringing is present.

    Physical Basis:
        Bessel filters have maximally flat group delay, resulting in
        zero overshoot and no ringing on step response. This property
        should be preserved through NMSE processing.
    """

    time_ms: np.ndarray
    response: np.ndarray
    overshoot_percent: float
    settling_time_ms: float
    has_ringing: bool


@dataclass(frozen=True)
class ImpulseResponseMetrics:
    """Impulse response metrics for group delay and phase analysis.

    Args:
        time_ms: Time axis in milliseconds.
        impulse: Impulse response waveform.
        peak_time_ms: Time of peak response.
        group_delay_samples: Group delay in samples.
        symmetry_score: Symmetry score (0=perfect, higher=worse).

    Physical Basis:
        Group delay characterizes phase distortion. Flat group delay
        indicates linear phase, preserving transient waveforms without
        pre-echo or post-ringing.
    """

    time_ms: np.ndarray
    impulse: np.ndarray
    peak_time_ms: float
    group_delay_samples: float
    symmetry_score: float


@dataclass(frozen=True)
class WaveformComparisonMetrics:
    """Waveform comparison metrics for time-domain preservation.

    Args:
        time_ms: Time axis in milliseconds.
        input_signal: Input waveform.
        target_signal: Target waveform.
        output_signal: Output waveform.
        mse_input_output: MSE between input and output.
        correlation: Correlation coefficient.

    Physical Basis:
        Time-domain comparison reveals whether 0-20kHz content is
        preserved without amplitude or phase distortion.
    """

    time_ms: np.ndarray
    input_signal: np.ndarray
    target_signal: np.ndarray
    output_signal: np.ndarray
    mse_input_output: float
    correlation: float


def compute_square_wave_response(
    system_signal: np.ndarray,
    sample_rate: int,
    transition_time_ms: float = 0.5,
) -> SquareWaveMetrics:
    """Compute square wave (step) response metrics.

    Args:
        system_signal: Signal processed through the system.
        sample_rate: Sample rate in Hz.
        transition_time_ms: Display window around transition (ms).

    Returns:
        SquareWaveMetrics containing response and overshoot metrics.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Square wave response reveals transient behavior. Overshoot
        and ringing indicate phase distortion or insufficient damping.
        Bessel filters have zero overshoot by design.
    """
    if system_signal.ndim != 1:
        raise ValueError(f"Signal must be 1D, got {system_signal.ndim}D")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    # Find transition point (zero crossing or midpoint)
    transition_idx = len(system_signal) // 2

    # Extract window around transition
    window_samples = int(transition_time_ms * sample_rate / 1000.0)
    start_idx = max(0, transition_idx - window_samples)
    end_idx = min(len(system_signal), transition_idx + window_samples)

    response = system_signal[start_idx:end_idx]
    time_samples = np.arange(len(response))
    time_ms = (time_samples - window_samples) * 1000.0 / sample_rate

    # Compute overshoot (assume step response settles to 1.0)
    final_value = np.mean(response[-int(sample_rate * 0.01) :])  # Last 10ms
    peak_value = np.max(response)
    overshoot_percent = float((peak_value - final_value) / abs(final_value) * 100.0)

    # Compute settling time (5% criterion)
    tolerance = 0.05 * abs(final_value)
    settled_mask = np.abs(response - final_value) <= tolerance
    if np.any(settled_mask):
        settling_idx = np.where(settled_mask)[0][0]
        settling_time_ms = float(time_ms[settling_idx])
    else:
        settling_time_ms = float(time_ms[-1])

    # Detect ringing (oscillations after peak)
    peak_idx = np.argmax(response)
    if peak_idx < len(response) - 10:
        post_peak = response[peak_idx:]
        # Simple ringing detection: count zero crossings
        crossings = np.where(np.diff(np.sign(post_peak - final_value)))[0]
        has_ringing = len(crossings) > 2
    else:
        has_ringing = False

    return SquareWaveMetrics(
        time_ms=time_ms,
        response=response,
        overshoot_percent=overshoot_percent,
        settling_time_ms=settling_time_ms,
        has_ringing=has_ringing,
    )


def compute_impulse_response(
    system_signal: np.ndarray,
    sample_rate: int,
    window_ms: float = 2.0,
) -> ImpulseResponseMetrics:
    """Compute impulse response metrics.

    Args:
        system_signal: Impulse response from the system.
        sample_rate: Sample rate in Hz.
        window_ms: Display window around peak (ms).

    Returns:
        ImpulseResponseMetrics containing group delay and symmetry.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Impulse response reveals group delay (peak time) and phase
        linearity (symmetry). Linear phase systems have symmetric
        impulse responses; minimum phase systems are asymmetric.
    """
    if system_signal.ndim != 1:
        raise ValueError(f"Signal must be 1D, got {system_signal.ndim}D")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    # Find peak
    peak_idx = np.argmax(np.abs(system_signal))
    peak_time_ms = float(peak_idx * 1000.0 / sample_rate)

    # Extract window around peak
    window_samples = int(window_ms * sample_rate / 1000.0)
    start_idx = max(0, int(peak_idx) - window_samples)
    end_idx = min(len(system_signal), int(peak_idx) + window_samples)

    impulse = system_signal[start_idx:end_idx]
    time_samples = np.arange(len(impulse))
    time_ms = (time_samples - (peak_idx - start_idx)) * 1000.0 / sample_rate

    # Compute group delay (samples from start to peak)
    group_delay_samples = float(peak_idx)

    # Compute symmetry score (compare left and right of peak)
    peak_in_window = int(peak_idx - start_idx)
    left_len = int(peak_in_window)
    right_len = int(len(impulse) - peak_in_window - 1)
    min_len = int(min(left_len, right_len))

    if min_len > 0:
        left = impulse[peak_in_window - min_len : peak_in_window]
        right = impulse[peak_in_window + 1 : peak_in_window + 1 + min_len]
        right_flipped = right[::-1]
        symmetry_score = float(np.mean(np.abs(left - right_flipped)))
    else:
        symmetry_score = 0.0

    return ImpulseResponseMetrics(
        time_ms=time_ms,
        impulse=impulse,
        peak_time_ms=peak_time_ms,
        group_delay_samples=group_delay_samples,
        symmetry_score=symmetry_score,
    )


def compute_waveform_comparison(
    input_signal: np.ndarray,
    target_signal: np.ndarray,
    output_signal: np.ndarray,
    sample_rate: int,
    window_ms: float = 10.0,
    offset_ms: float = 0.0,
) -> WaveformComparisonMetrics:
    """Compute waveform comparison metrics.

    Args:
        input_signal: Original input signal.
        target_signal: Reference target signal.
        output_signal: System output signal.
        sample_rate: Sample rate in Hz.
        window_ms: Display window duration (ms).
        offset_ms: Start time offset (ms).

    Returns:
        WaveformComparisonMetrics for visual comparison.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Time-domain overlay reveals amplitude and phase preservation.
        High correlation and low MSE indicate faithful reproduction
        of the 0-20kHz content without distortion.
    """
    if input_signal.ndim != 1 or target_signal.ndim != 1 or output_signal.ndim != 1:
        raise ValueError("All signals must be 1D")
    if not (input_signal.shape == target_signal.shape == output_signal.shape):
        raise ValueError("All signals must have same shape")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    # Extract window
    offset_samples = int(offset_ms * sample_rate / 1000.0)
    window_samples = int(window_ms * sample_rate / 1000.0)
    start_idx = offset_samples
    end_idx = min(len(input_signal), offset_samples + window_samples)

    input_window = input_signal[start_idx:end_idx]
    target_window = target_signal[start_idx:end_idx]
    output_window = output_signal[start_idx:end_idx]

    time_samples = np.arange(len(input_window))
    time_ms = time_samples * 1000.0 / sample_rate + offset_ms

    # Compute metrics
    mse_input_output = float(np.mean((input_window - output_window) ** 2))
    correlation = float(
        np.corrcoef(input_window, output_window)[0, 1] if len(input_window) > 1 else 1.0
    )

    return WaveformComparisonMetrics(
        time_ms=time_ms,
        input_signal=input_window,
        target_signal=target_window,
        output_signal=output_window,
        mse_input_output=mse_input_output,
        correlation=correlation,
    )


def plot_square_wave_response(
    before_metrics: SquareWaveMetrics,
    after_metrics: SquareWaveMetrics,
    output_path: Path,
    title: str = "Square Wave Response",
) -> None:
    """Plot square wave response comparison.

    Args:
        before_metrics: Metrics before NMSE processing.
        after_metrics: Metrics after NMSE processing.
        output_path: Path to save PNG.
        title: Plot title.

    Physical Basis:
        Visual inspection reveals overshoot and ringing. Bessel filters
        maintain zero overshoot; NMSE should preserve this property.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Before
    ax1.plot(before_metrics.time_ms, before_metrics.response, linewidth=1.5)
    ax1.axhline(1.0, color="red", linestyle="--", alpha=0.3, label="Target")
    ax1.axhline(0.0, color="gray", linestyle="-", alpha=0.3)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(
        f"{title} - Before | Overshoot: {before_metrics.overshoot_percent:.2f}%"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # After
    ax2.plot(
        after_metrics.time_ms, after_metrics.response, linewidth=1.5, color="orange"
    )
    ax2.axhline(1.0, color="red", linestyle="--", alpha=0.3, label="Target")
    ax2.axhline(0.0, color="gray", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title(
        f"{title} - After | Overshoot: {after_metrics.overshoot_percent:.2f}%"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    metrics_text = (
        f"Before: Overshoot={before_metrics.overshoot_percent:.2f}%, "
        f"Settling={before_metrics.settling_time_ms:.2f}ms, "
        f"Ringing={'Yes' if before_metrics.has_ringing else 'No'}\n"
        f"After: Overshoot={after_metrics.overshoot_percent:.2f}%, "
        f"Settling={after_metrics.settling_time_ms:.2f}ms, "
        f"Ringing={'Yes' if after_metrics.has_ringing else 'No'}"
    )
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=9, family="monospace")

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_impulse_response(
    before_metrics: ImpulseResponseMetrics,
    after_metrics: ImpulseResponseMetrics,
    output_path: Path,
    title: str = "Impulse Response",
) -> None:
    """Plot impulse response comparison.

    Args:
        before_metrics: Metrics before NMSE processing.
        after_metrics: Metrics after NMSE processing.
        output_path: Path to save PNG.
        title: Plot title.

    Physical Basis:
        Impulse response shape reveals group delay and phase linearity.
        Symmetric responses indicate linear phase; asymmetry suggests
        minimum phase or phase distortion.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Before
    ax1.plot(before_metrics.time_ms, before_metrics.impulse, linewidth=1.5)
    ax1.axvline(0, color="red", linestyle="--", alpha=0.3, label="Peak")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(
        f"{title} - Before | Group Delay: {before_metrics.group_delay_samples:.1f} samples"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # After
    ax2.plot(
        after_metrics.time_ms, after_metrics.impulse, linewidth=1.5, color="orange"
    )
    ax2.axvline(0, color="red", linestyle="--", alpha=0.3, label="Peak")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title(
        f"{title} - After | Group Delay: {after_metrics.group_delay_samples:.1f} samples"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    metrics_text = (
        f"Before: Group Delay={before_metrics.group_delay_samples:.1f} samples, "
        f"Symmetry Score={before_metrics.symmetry_score:.4f}\n"
        f"After: Group Delay={after_metrics.group_delay_samples:.1f} samples, "
        f"Symmetry Score={after_metrics.symmetry_score:.4f}"
    )
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=9, family="monospace")

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_waveform_comparison(
    metrics: WaveformComparisonMetrics,
    output_path: Path,
    title: str = "Waveform Comparison",
) -> None:
    """Plot time-domain waveform comparison.

    Args:
        metrics: Waveform comparison metrics.
        output_path: Path to save PNG.
        title: Plot title.

    Physical Basis:
        Overlay of input, target, and output reveals preservation
        of 0-20kHz content. Deviations indicate amplitude or phase
        distortion introduced by processing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Overlay
    ax1.plot(
        metrics.time_ms, metrics.input_signal, label="Input", alpha=0.7, linewidth=1.5
    )
    ax1.plot(
        metrics.time_ms, metrics.target_signal, label="Target", alpha=0.7, linewidth=1.5
    )
    ax1.plot(
        metrics.time_ms,
        metrics.output_signal,
        label="Output (NMSE)",
        alpha=0.7,
        linewidth=1.5,
    )
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"{title} - Overlay")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Difference (Output - Input)
    difference = metrics.output_signal - metrics.input_signal
    ax2.plot(
        metrics.time_ms, difference, label="Output - Input", color="red", linewidth=1.5
    )
    ax2.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Difference")
    ax2.set_title("Difference (Output - Input)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    metrics_text = (
        f"MSE (Input vs Output): {metrics.mse_input_output:.6e}  |  "
        f"Correlation: {metrics.correlation:.6f}"
    )
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=9, family="monospace")

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "SquareWaveMetrics",
    "ImpulseResponseMetrics",
    "WaveformComparisonMetrics",
    "compute_square_wave_response",
    "compute_impulse_response",
    "compute_waveform_comparison",
    "plot_square_wave_response",
    "plot_impulse_response",
    "plot_waveform_comparison",
]
