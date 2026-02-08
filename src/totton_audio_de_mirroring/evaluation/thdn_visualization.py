"""THD+N spectrum visualization for distortion analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from totton_audio_de_mirroring.data.filters import band_split, design_band_split_filters
from totton_audio_de_mirroring.evaluation.imd_proxy import apply_soft_clipping

DEFAULT_CUTOFF_HZ = 20_000.0
DEFAULT_NUM_TAPS = 1025
DEFAULT_CLIP_DRIVE = 2.0
EPSILON = 1.0e-12


@dataclass(frozen=True)
class THDNSpectrumMetrics:
    """THD+N spectrum analysis metrics.

    Args:
        frequencies: Frequency bins in Hz.
        distortion_spectrum_db: Distortion spectrum in dB.
        signal_spectrum_db: Clean signal spectrum in dB.
        thdn_db: Overall THD+N ratio in dB.
        max_harmonic_db: Maximum harmonic distortion level in dB.

    Physical Basis:
        THD+N spectrum reveals nonlinear distortion introduced by
        processing. Audible-band distortion should remain below -60dB.
    """

    frequencies: np.ndarray
    distortion_spectrum_db: np.ndarray
    signal_spectrum_db: np.ndarray
    thdn_db: float
    max_harmonic_db: float


def compute_thdn_spectrum(
    reference_signal: np.ndarray,
    measured_signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    num_taps: int = DEFAULT_NUM_TAPS,
    clip_drive: float = DEFAULT_CLIP_DRIVE,
    n_fft: int = 8192,
) -> THDNSpectrumMetrics:
    """Compute THD+N spectrum from signal pair.

    Args:
        reference_signal: Original input signal.
        measured_signal: Signal after processing (potentially distorted).
        sample_rate: Sample rate in Hz.
        cutoff_hz: Low-band cutoff frequency in Hz.
        num_taps: FIR filter taps for band extraction.
        clip_drive: Soft-clipping drive for nonlinearity simulation.
        n_fft: FFT size for spectrum analysis.

    Returns:
        THDNSpectrumMetrics containing distortion spectrum and metrics.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Applying soft clipping simulates mild analog nonlinearity.
        The distortion spectrum reveals harmonics and intermodulation
        products in the audible band that arise from high-frequency
        artifacts folding down through nonlinear transfer functions.
    """
    if reference_signal.ndim != 1 or measured_signal.ndim != 1:
        raise ValueError("Signals must be 1D arrays")
    if reference_signal.shape != measured_signal.shape:
        raise ValueError("Signals must have the same shape")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    # Apply soft clipping to simulate nonlinearity
    clipped_reference = apply_soft_clipping(reference_signal, drive=clip_drive)
    clipped_measured = apply_soft_clipping(measured_signal, drive=clip_drive)

    # Extract low-band (audible) components
    lowpass_taps, highpass_taps = design_band_split_filters(
        cutoff_hz=cutoff_hz,
        sample_rate=sample_rate,
        num_taps=num_taps,
    )

    lb_ref, _ = band_split(clipped_reference, lowpass_taps, highpass_taps)
    lb_measured, _ = band_split(clipped_measured, lowpass_taps, highpass_taps)

    # Trim filter warmup
    lb_ref = lb_ref[num_taps - 1 :]
    lb_measured = lb_measured[num_taps - 1 :]

    # Compute best-fit gain
    gain = float(np.dot(lb_ref, lb_measured) / (np.dot(lb_ref, lb_ref) + EPSILON))
    lb_ref_scaled = gain * lb_ref

    # Compute distortion (residual)
    distortion = lb_measured - lb_ref_scaled

    # Compute spectra
    ref_spectrum = np.fft.rfft(lb_ref_scaled, n=n_fft)
    distortion_spectrum = np.fft.rfft(distortion, n=n_fft)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # Convert to dB
    ref_magnitude = np.abs(ref_spectrum) + EPSILON
    distortion_magnitude = np.abs(distortion_spectrum) + EPSILON

    signal_spectrum_db = 20.0 * np.log10(ref_magnitude)
    distortion_spectrum_db = 20.0 * np.log10(distortion_magnitude)

    # Normalize signal to 0dB at peak
    signal_spectrum_db -= np.max(signal_spectrum_db)
    # Normalize distortion relative to signal peak
    distortion_spectrum_db -= np.max(signal_spectrum_db)

    # Compute overall THD+N
    ref_rms = float(np.sqrt(np.mean(lb_ref_scaled**2)))
    distortion_rms = float(np.sqrt(np.mean(distortion**2)))
    thdn_db = float(20.0 * np.log10((distortion_rms + EPSILON) / (ref_rms + EPSILON)))

    # Find max harmonic in audible band
    audible_mask = frequencies <= cutoff_hz
    max_harmonic_db = float(np.max(distortion_spectrum_db[audible_mask]))

    return THDNSpectrumMetrics(
        frequencies=frequencies,
        distortion_spectrum_db=distortion_spectrum_db,
        signal_spectrum_db=signal_spectrum_db,
        thdn_db=thdn_db,
        max_harmonic_db=max_harmonic_db,
    )


def plot_thdn_spectrum(
    before_metrics: THDNSpectrumMetrics,
    after_metrics: THDNSpectrumMetrics,
    output_path: Path,
    title: str = "THD+N Spectrum Comparison",
    max_freq_khz: float = 20.0,
) -> None:
    """Plot THD+N spectrum comparison before/after processing.

    Args:
        before_metrics: THD+N metrics before NMSE processing.
        after_metrics: THD+N metrics after NMSE processing.
        output_path: Path to save PNG image.
        title: Plot title.
        max_freq_khz: Maximum frequency to display in kHz.

    Raises:
        ValueError: If metrics are incompatible.

    Physical Basis:
        Visualization reveals whether NMSE processing introduces nonlinear
        distortion in the audible band. Harmonics should remain below -60dB.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Convert to kHz
    freq_before_khz = before_metrics.frequencies / 1000.0
    freq_after_khz = after_metrics.frequencies / 1000.0

    # Top panel: Signal + Distortion for Before
    mask_before = freq_before_khz <= max_freq_khz
    ax1.plot(
        freq_before_khz[mask_before],
        before_metrics.signal_spectrum_db[mask_before],
        label="Signal",
        alpha=0.7,
        linewidth=1.5,
    )
    ax1.plot(
        freq_before_khz[mask_before],
        before_metrics.distortion_spectrum_db[mask_before],
        label="Distortion",
        alpha=0.7,
        linewidth=1.5,
        color="red",
    )
    ax1.axhline(-60, color="orange", linestyle="--", alpha=0.5, label="-60dB Target")
    ax1.set_xlim(0, max_freq_khz)
    ax1.set_ylim(-120, 5)
    ax1.set_xlabel("Frequency (kHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title(
        f"{title} - Before (Naive Bessel) | THD+N: {before_metrics.thdn_db:.1f} dB"
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Middle panel: Signal + Distortion for After
    mask_after = freq_after_khz <= max_freq_khz
    ax2.plot(
        freq_after_khz[mask_after],
        after_metrics.signal_spectrum_db[mask_after],
        label="Signal",
        alpha=0.7,
        linewidth=1.5,
    )
    ax2.plot(
        freq_after_khz[mask_after],
        after_metrics.distortion_spectrum_db[mask_after],
        label="Distortion",
        alpha=0.7,
        linewidth=1.5,
        color="red",
    )
    ax2.axhline(-60, color="orange", linestyle="--", alpha=0.5, label="-60dB Target")
    ax2.set_xlim(0, max_freq_khz)
    ax2.set_ylim(-120, 5)
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_title(f"{title} - After (NMSE) | THD+N: {after_metrics.thdn_db:.1f} dB")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Bottom panel: Distortion comparison
    if len(freq_before_khz) == len(freq_after_khz):
        ax3.plot(
            freq_before_khz[mask_before],
            before_metrics.distortion_spectrum_db[mask_before],
            label="Before Distortion",
            alpha=0.7,
            linewidth=1.5,
        )
        ax3.plot(
            freq_after_khz[mask_after],
            after_metrics.distortion_spectrum_db[mask_after],
            label="After Distortion",
            alpha=0.7,
            linewidth=1.5,
        )
        ax3.axhline(
            -60, color="orange", linestyle="--", alpha=0.5, label="-60dB Target"
        )
        ax3.set_xlim(0, max_freq_khz)
        ax3.set_ylim(-120, 5)
        ax3.set_xlabel("Frequency (kHz)")
        ax3.set_ylabel("Distortion (dB)")
        ax3.set_title("Distortion Spectrum Comparison")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "Comparison unavailable\n(incompatible frequency bins)",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )

    # Add metrics text
    thdn_improvement = before_metrics.thdn_db - after_metrics.thdn_db
    metrics_text = (
        f"Before: THD+N = {before_metrics.thdn_db:.1f} dB, Max Harmonic = {before_metrics.max_harmonic_db:.1f} dB\n"
        f"After: THD+N = {after_metrics.thdn_db:.1f} dB, Max Harmonic = {after_metrics.max_harmonic_db:.1f} dB\n"
        f"Improvement: {thdn_improvement:.1f} dB"
    )
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=9, family="monospace")

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_thdn_spectrum_pair(
    before_signal: np.ndarray,
    after_signal: np.ndarray,
    sample_rate: int,
    output_path: Path,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    num_taps: int = DEFAULT_NUM_TAPS,
    clip_drive: float = DEFAULT_CLIP_DRIVE,
    n_fft: int = 8192,
    title: str = "THD+N Spectrum",
) -> tuple[THDNSpectrumMetrics, THDNSpectrumMetrics]:
    """Evaluate and plot THD+N spectrum for signal pair.

    Args:
        before_signal: Signal before NMSE processing.
        after_signal: Signal after NMSE processing.
        sample_rate: Sample rate in Hz.
        output_path: Path to save visualization.
        cutoff_hz: Audible band cutoff in Hz.
        num_taps: FIR filter taps.
        clip_drive: Nonlinearity drive.
        n_fft: FFT size.
        title: Plot title.

    Returns:
        Tuple of (before_metrics, after_metrics).

    Raises:
        ValueError: If signals are incompatible.

    Physical Basis:
        THD+N analysis verifies that NMSE processing does not introduce
        nonlinear distortion in the audible band (0-20kHz), which would
        indicate the neural network is modifying the passband content.
    """
    if before_signal.shape != after_signal.shape:
        raise ValueError(
            f"Signals must have same shape, got {before_signal.shape} and {after_signal.shape}"
        )

    # Use naive upsampled signal as reference
    before_metrics = compute_thdn_spectrum(
        reference_signal=before_signal,
        measured_signal=before_signal,
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        num_taps=num_taps,
        clip_drive=clip_drive,
        n_fft=n_fft,
    )

    after_metrics = compute_thdn_spectrum(
        reference_signal=before_signal,  # Compare against same reference
        measured_signal=after_signal,
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        num_taps=num_taps,
        clip_drive=clip_drive,
        n_fft=n_fft,
    )

    plot_thdn_spectrum(before_metrics, after_metrics, output_path, title)

    return before_metrics, after_metrics


__all__ = [
    "THDNSpectrumMetrics",
    "compute_thdn_spectrum",
    "plot_thdn_spectrum",
    "evaluate_thdn_spectrum_pair",
]
