"""Tests for THD+N visualization module."""

from pathlib import Path

import numpy as np
import pytest

from totton_audio_de_mirroring.evaluation.thdn_visualization import (
    THDNSpectrumMetrics,
    compute_thdn_spectrum,
    evaluate_thdn_spectrum_pair,
)


def test_compute_thdn_spectrum_basic():
    """Test basic THD+N spectrum computation."""
    sample_rate = 88_200
    duration = 1.0

    # Generate clean 1kHz sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * 1000 * t)

    # Compute THD+N spectrum (self-comparison)
    metrics = compute_thdn_spectrum(
        reference_signal=signal,
        measured_signal=signal,
        sample_rate=sample_rate,
        n_fft=4096,
    )

    # Verify output types
    assert isinstance(metrics, THDNSpectrumMetrics)
    assert isinstance(metrics.frequencies, np.ndarray)
    assert isinstance(metrics.distortion_spectrum_db, np.ndarray)
    assert isinstance(metrics.signal_spectrum_db, np.ndarray)
    assert isinstance(metrics.thdn_db, float)
    assert isinstance(metrics.max_harmonic_db, float)

    # Self-comparison should have very low THD+N
    assert metrics.thdn_db < -60.0  # Better than -60dB


def test_compute_thdn_spectrum_invalid_input():
    """Test that invalid inputs raise appropriate errors."""
    signal = np.random.randn(88_200)

    # 2D signal should raise error
    with pytest.raises(ValueError, match="must be 1D"):
        compute_thdn_spectrum(np.random.randn(10, 10), signal, 88_200)

    # Shape mismatch should raise error
    with pytest.raises(ValueError, match="same shape"):
        compute_thdn_spectrum(signal, np.random.randn(1000), 88_200)

    # Invalid sample rate should raise error
    with pytest.raises(ValueError, match="must be positive"):
        compute_thdn_spectrum(signal, signal, -1)

    # Empty signal should raise error
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_thdn_spectrum(np.array([]), np.array([]), 88_200)

    # Invalid FFT/taps should raise error
    with pytest.raises(ValueError, match="n_fft must be positive"):
        compute_thdn_spectrum(signal, signal, 88_200, n_fft=0)
    with pytest.raises(ValueError, match="num_taps must be positive"):
        compute_thdn_spectrum(signal, signal, 88_200, num_taps=0)
    with pytest.raises(ValueError, match="num_taps must be odd"):
        compute_thdn_spectrum(signal, signal, 88_200, num_taps=100)
    with pytest.raises(ValueError, match="must be >= num_taps"):
        compute_thdn_spectrum(signal[:100], signal[:100], 88_200, num_taps=129)


def test_evaluate_thdn_spectrum_pair_creates_file(tmp_path: Path):
    """Test that THD+N visualization file is created."""
    sample_rate = 88_200
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Clean signal
    signal_clean = np.sin(2 * np.pi * 1000 * t)

    # Slightly distorted signal (add 3rd harmonic)
    signal_distorted = signal_clean + 0.01 * np.sin(2 * np.pi * 3000 * t)

    output_path = tmp_path / "test_thdn.png"

    before_metrics, after_metrics = evaluate_thdn_spectrum_pair(
        before_signal=signal_clean,
        after_signal=signal_distorted,
        sample_rate=sample_rate,
        output_path=output_path,
        n_fft=2048,
    )

    # Verify file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Distorted signal should have worse THD+N
    assert after_metrics.thdn_db > before_metrics.thdn_db


def test_thdn_spectrum_detects_harmonics():
    """Test that THD+N computation detects harmonic distortion."""
    sample_rate = 88_200
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Fundamental frequency
    freq = 1000.0
    signal_clean = np.sin(2 * np.pi * freq * t)

    # Add 2nd and 3rd harmonics (distortion)
    signal_distorted = (
        signal_clean
        + 0.05 * np.sin(2 * np.pi * 2 * freq * t)  # 2nd harmonic
        + 0.03 * np.sin(2 * np.pi * 3 * freq * t)  # 3rd harmonic
    )

    # Compare clean to distorted
    metrics = compute_thdn_spectrum(
        reference_signal=signal_clean,
        measured_signal=signal_distorted,
        sample_rate=sample_rate,
        cutoff_hz=20_000.0,
        n_fft=8192,
    )

    # Should detect distortion
    assert metrics.thdn_db > -40.0  # Significant distortion present
    assert metrics.max_harmonic_db > -60.0  # Harmonics visible


def test_thdn_spectrum_audible_band_only():
    """Test that THD+N focuses on audible band."""
    sample_rate = 88_200
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Audible tone + ultrasonic noise
    audible = np.sin(2 * np.pi * 1000 * t)
    ultrasonic = 0.5 * np.sin(2 * np.pi * 30_000 * t)
    signal = audible + ultrasonic

    metrics = compute_thdn_spectrum(
        reference_signal=signal,
        measured_signal=signal,
        sample_rate=sample_rate,
        cutoff_hz=20_000.0,
        n_fft=8192,
    )

    # Frequency array should be available
    assert len(metrics.frequencies) > 0
    assert np.max(metrics.frequencies) <= sample_rate / 2
