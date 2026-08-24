"""Frequency response visualization for audio quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FrequencyResponseMetrics:
    """Frequency response evaluation metrics.

    Args:
        frequencies: Frequency bins in Hz.
        magnitude_db: Magnitude spectrum in dB.
        nyquist_hz: Nyquist frequency in Hz.
        attenuation_44khz_db: Attenuation at 44.1kHz in dB.
        imaging_energy_100khz_plus: Energy above 100kHz.

    Physical Basis:
        Frequency response characterizes the system's spectral behavior,
        verifying anti-aliasing and absence of ultrasonic artifacts.
    """

    frequencies: np.ndarray
    magnitude_db: np.ndarray
    nyquist_hz: float
    attenuation_44khz_db: float
    imaging_energy_100khz_plus: float


def compute_frequency_response(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int = 8192,
) -> FrequencyResponseMetrics:
    """Compute frequency response from audio signal.

    Args:
        signal: Input audio signal (1D array).
        sample_rate: Sample rate in Hz.
        n_fft: FFT size for frequency resolution.

    Returns:
        FrequencyResponseMetrics containing spectrum and metrics.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        FFT magnitude spectrum reveals the frequency content, allowing
        verification of anti-aliasing filters and detection of imaging
        artifacts beyond the original Nyquist limit.
    """
    if signal.ndim != 1:
        raise ValueError(f"Signal must be 1D, got {signal.ndim}D")
    if signal.size == 0:
        raise ValueError("Signal cannot be empty")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    if n_fft <= 0:
        raise ValueError(f"n_fft must be positive, got {n_fft}")

    # Compute FFT
    spectrum = np.fft.rfft(signal, n=n_fft)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    magnitude = np.abs(spectrum)

    # Convert to dB (with epsilon to avoid log(0))
    epsilon = 1e-12
    magnitude_db = 20.0 * np.log10(magnitude + epsilon)

    # Normalize to 0dB at peak
    magnitude_db -= np.max(magnitude_db)

    # Compute metrics
    nyquist_hz = sample_rate / 2.0

    # Find attenuation at 44.1kHz
    idx_44khz = np.argmin(np.abs(frequencies - 44_100.0))
    attenuation_44khz_db = float(magnitude_db[idx_44khz])

    # Compute energy above 100kHz
    mask_100khz_plus = frequencies >= 100_000.0
    if np.any(mask_100khz_plus):
        imaging_energy = float(np.sum(magnitude[mask_100khz_plus] ** 2))
    else:
        imaging_energy = 0.0

    return FrequencyResponseMetrics(
        frequencies=frequencies,
        magnitude_db=magnitude_db,
        nyquist_hz=nyquist_hz,
        attenuation_44khz_db=attenuation_44khz_db,
        imaging_energy_100khz_plus=imaging_energy,
    )


def plot_frequency_response(
    before_metrics: FrequencyResponseMetrics,
    after_metrics: FrequencyResponseMetrics,
    output_path: Path,
    title: str = "Frequency Response Comparison",
    max_freq_khz: float | None = None,
) -> None:
    """Plot frequency response comparison before/after processing.

    Args:
        before_metrics: Reference frequency response.
        after_metrics: CAPB frequency response.
        output_path: Path to save PNG image.
        title: Plot title.
        max_freq_khz: Maximum frequency to display in kHz (None = Nyquist).

    Raises:
        ValueError: If metrics are incompatible.

    Physical Basis:
        Visual comparison reveals spectral changes: attenuation beyond
        44.1kHz confirms anti-aliasing, while low imaging energy (>100kHz)
        indicates successful removal of upsampling artifacts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Convert frequencies to kHz for readability
    freq_before_khz = before_metrics.frequencies / 1000.0
    freq_after_khz = after_metrics.frequencies / 1000.0

    # Determine plot range
    if max_freq_khz is None:
        max_freq_khz = min(before_metrics.nyquist_hz, after_metrics.nyquist_hz) / 1000.0

    # Top panel: Before and After overlay
    ax1.plot(
        freq_before_khz,
        before_metrics.magnitude_db,
        label="Before (Naive Bessel)",
        alpha=0.7,
        linewidth=1.5,
    )
    ax1.plot(
        freq_after_khz,
        after_metrics.magnitude_db,
        label="CAPB output",
        alpha=0.7,
        linewidth=1.5,
    )
    ax1.axvline(
        44.1, color="red", linestyle="--", alpha=0.5, label="44.1kHz (Original Nyquist)"
    )
    ax1.axvline(100, color="orange", linestyle="--", alpha=0.5, label="100kHz")
    ax1.set_xlim(0, max_freq_khz)
    ax1.set_ylim(-120, 5)
    ax1.set_xlabel("Frequency (kHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title(f"{title} - Overlay")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Difference (After - Before)
    # Ensure same frequency bins
    if len(freq_before_khz) == len(freq_after_khz):
        diff_db = after_metrics.magnitude_db - before_metrics.magnitude_db
        ax2.plot(
            freq_after_khz,
            diff_db,
            label="Difference (After - Before)",
            color="green",
            linewidth=1.5,
        )
        ax2.axhline(0, color="black", linestyle="-", alpha=0.3)
        ax2.axvline(44.1, color="red", linestyle="--", alpha=0.5)
        ax2.axvline(100, color="orange", linestyle="--", alpha=0.5)
        ax2.set_xlim(0, max_freq_khz)
        ax2.set_xlabel("Frequency (kHz)")
        ax2.set_ylabel("Difference (dB)")
        ax2.set_title("Spectral Change (After - Before)")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "Difference plot unavailable\n(incompatible frequency bins)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    # Add metrics text
    metrics_text = (
        f"Before: Attenuation @ 44.1kHz = {before_metrics.attenuation_44khz_db:.1f} dB, "
        f"Imaging >100kHz = {before_metrics.imaging_energy_100khz_plus:.2e}\n"
        f"After: Attenuation @ 44.1kHz = {after_metrics.attenuation_44khz_db:.1f} dB, "
        f"Imaging >100kHz = {after_metrics.imaging_energy_100khz_plus:.2e}"
    )
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=9, family="monospace")

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_frequency_response_pair(
    before_signal: np.ndarray,
    after_signal: np.ndarray,
    sample_rate: int,
    output_path: Path,
    n_fft: int = 8192,
    title: str = "Frequency Response",
    max_freq_khz: float | None = None,
) -> tuple[FrequencyResponseMetrics, FrequencyResponseMetrics]:
    """Evaluate and plot frequency response for signal pair.

    Args:
        before_signal: Reference signal.
        after_signal: CAPB output signal.
        sample_rate: Sample rate in Hz.
        output_path: Path to save visualization.
        n_fft: FFT size for analysis.
        title: Plot title.
        max_freq_khz: Maximum frequency to display in kHz.

    Returns:
        Tuple of (before_metrics, after_metrics).

    Raises:
        ValueError: If signals are incompatible.

    Physical Basis:
        Comparing the reference and CAPB frequency responses verifies
        that the network suppresses mirror artifacts while preserving
        the 0-20kHz passband and not generating ultrasonic content.
    """
    if before_signal.shape != after_signal.shape:
        raise ValueError(
            f"Signals must have same shape, got {before_signal.shape} and {after_signal.shape}"
        )

    before_metrics = compute_frequency_response(before_signal, sample_rate, n_fft)
    after_metrics = compute_frequency_response(after_signal, sample_rate, n_fft)

    plot_frequency_response(
        before_metrics, after_metrics, output_path, title, max_freq_khz
    )

    return before_metrics, after_metrics


__all__ = [
    "FrequencyResponseMetrics",
    "compute_frequency_response",
    "plot_frequency_response",
    "evaluate_frequency_response_pair",
]
