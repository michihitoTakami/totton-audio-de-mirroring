"""Tests for frequency response visualization module."""

from pathlib import Path

import numpy as np
import pytest

from totton_audio_de_mirroring.evaluation.frequency_response import (
    FrequencyResponseMetrics,
    compute_frequency_response,
    evaluate_frequency_response_pair,
)


def test_compute_frequency_response_basic():
    """Test basic frequency response computation."""
    # Generate 1kHz sine wave at 48kHz
    sample_rate = 48_000
    duration = 1.0
    freq = 1000.0

    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * freq * t)

    # Compute frequency response
    metrics = compute_frequency_response(signal, sample_rate, n_fft=4096)

    # Verify output types
    assert isinstance(metrics, FrequencyResponseMetrics)
    assert isinstance(metrics.frequencies, np.ndarray)
    assert isinstance(metrics.magnitude_db, np.ndarray)
    assert isinstance(metrics.nyquist_hz, float)
    assert isinstance(metrics.attenuation_44khz_db, float)
    assert isinstance(metrics.imaging_energy_100khz_plus, float)

    # Verify frequency range
    assert metrics.frequencies[0] == 0.0
    assert metrics.frequencies[-1] <= sample_rate / 2
    assert metrics.nyquist_hz == sample_rate / 2


def test_compute_frequency_response_invalid_input():
    """Test that invalid inputs raise appropriate errors."""
    signal = np.random.randn(1000)

    # 2D signal should raise error
    with pytest.raises(ValueError, match="must be 1D"):
        compute_frequency_response(np.random.randn(10, 10), 48_000)

    # Empty signal should raise error
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_frequency_response(np.array([]), 48_000)

    # Invalid sample rate should raise error
    with pytest.raises(ValueError, match="must be positive"):
        compute_frequency_response(signal, -1)

    # Invalid n_fft should raise error
    with pytest.raises(ValueError, match="must be positive"):
        compute_frequency_response(signal, 48_000, n_fft=-1)


def test_evaluate_frequency_response_pair_shape_mismatch():
    """Test that shape mismatch raises error."""
    signal1 = np.random.randn(1000)
    signal2 = np.random.randn(2000)

    with pytest.raises(ValueError, match="same shape"):
        evaluate_frequency_response_pair(
            signal1, signal2, 48_000, Path("/tmp/test.png")
        )


def test_evaluate_frequency_response_pair_creates_file(tmp_path: Path):
    """Test that visualization file is created."""
    sample_rate = 48_000
    signal = np.random.randn(sample_rate)

    output_path = tmp_path / "test_freq_response.png"

    before_metrics, after_metrics = evaluate_frequency_response_pair(
        before_signal=signal,
        after_signal=signal,
        sample_rate=sample_rate,
        output_path=output_path,
        n_fft=2048,
    )

    # Verify file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Verify metrics are identical (same signal)
    assert np.allclose(
        before_metrics.magnitude_db, after_metrics.magnitude_db, atol=1e-6
    )


def test_frequency_response_attenuation_44khz():
    """Test 44.1kHz attenuation measurement."""
    # Create signal with strong 44.1kHz component
    sample_rate = 88_200  # Stage 1 sample rate
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Low frequency + 44.1kHz tone
    signal = np.sin(2 * np.pi * 1000 * t) + 0.1 * np.sin(2 * np.pi * 44_100 * t)

    metrics = compute_frequency_response(signal, sample_rate, n_fft=8192)

    # 44.1kHz should have some attenuation (negative dB)
    assert metrics.attenuation_44khz_db < 0


def test_frequency_response_imaging_energy():
    """Test imaging energy measurement above 100kHz."""
    # Low sample rate: no content above 100kHz
    sample_rate_low = 88_200
    signal_low = np.random.randn(sample_rate_low)
    metrics_low = compute_frequency_response(signal_low, sample_rate_low, n_fft=4096)
    assert metrics_low.imaging_energy_100khz_plus == 0.0

    # High sample rate: potential content above 100kHz
    sample_rate_high = 705_600
    signal_high = np.random.randn(sample_rate_high)
    metrics_high = compute_frequency_response(signal_high, sample_rate_high, n_fft=8192)
    assert metrics_high.imaging_energy_100khz_plus >= 0.0
