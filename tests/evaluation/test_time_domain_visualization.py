"""Tests for time-domain visualization module."""

from pathlib import Path

import numpy as np
import pytest

from totton_audio_de_mirroring.evaluation.time_domain_visualization import (
    EdgeAlignedRingingMetrics,
    ImpulseResponseMetrics,
    RingingComparisonMetrics,
    SquareWaveMetrics,
    WaveformComparisonMetrics,
    compare_edge_aligned_ringing,
    compute_edge_aligned_ringing_metrics,
    compute_impulse_response,
    compute_square_wave_response,
    compute_waveform_comparison,
    plot_impulse_response,
    plot_square_wave_response,
    plot_waveform_comparison,
)


def test_compute_square_wave_response_basic():
    """Test basic square wave response computation."""
    sample_rate = 88_200
    duration = 0.1

    # Create step response (0 to 1)
    samples = int(sample_rate * duration)
    signal = np.concatenate([np.zeros(samples // 2), np.ones(samples // 2)])

    metrics = compute_square_wave_response(signal, sample_rate, transition_time_ms=5.0)

    # Verify output types
    assert isinstance(metrics, SquareWaveMetrics)
    assert isinstance(metrics.time_ms, np.ndarray)
    assert isinstance(metrics.response, np.ndarray)
    assert isinstance(metrics.overshoot_percent, float)
    assert isinstance(metrics.settling_time_ms, float)
    assert isinstance(metrics.has_ringing, bool)


def test_compute_square_wave_response_invalid_input():
    """Test that invalid inputs raise appropriate errors."""
    signal = np.random.randn(1000)

    # 2D signal should raise error
    with pytest.raises(ValueError, match="must be 1D"):
        compute_square_wave_response(np.random.randn(10, 10), 48_000)

    # Invalid sample rate should raise error
    with pytest.raises(ValueError, match="must be positive"):
        compute_square_wave_response(signal, -1)

    # Empty signal should raise error
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_square_wave_response(np.array([]), 48_000)


def test_compute_impulse_response_basic():
    """Test basic impulse response computation."""
    sample_rate = 88_200
    duration = 0.01

    # Create impulse (peak at center)
    samples = int(sample_rate * duration)
    signal = np.zeros(samples)
    signal[samples // 2] = 1.0

    metrics = compute_impulse_response(signal, sample_rate, window_ms=2.0)

    # Verify output types
    assert isinstance(metrics, ImpulseResponseMetrics)
    assert isinstance(metrics.time_ms, np.ndarray)
    assert isinstance(metrics.impulse, np.ndarray)
    assert isinstance(metrics.peak_time_ms, float)
    assert isinstance(metrics.group_delay_samples, float)
    assert isinstance(metrics.symmetry_score, float)

    # Peak should be near center
    assert metrics.group_delay_samples == samples // 2


def test_compute_impulse_response_invalid_input():
    """Test that invalid inputs raise appropriate errors."""
    signal = np.random.randn(1000)

    # 2D signal should raise error
    with pytest.raises(ValueError, match="must be 1D"):
        compute_impulse_response(np.random.randn(10, 10), 48_000)

    # Invalid sample rate should raise error
    with pytest.raises(ValueError, match="must be positive"):
        compute_impulse_response(signal, -1)


def test_compute_waveform_comparison_basic():
    """Test basic waveform comparison."""
    sample_rate = 88_200
    duration = 0.01
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create signals
    input_signal = np.sin(2 * np.pi * 1000 * t)
    target_signal = input_signal.copy()
    output_signal = input_signal + 0.01 * np.random.randn(len(input_signal))

    metrics = compute_waveform_comparison(
        input_signal, target_signal, output_signal, sample_rate, window_ms=10.0
    )

    # Verify output types
    assert isinstance(metrics, WaveformComparisonMetrics)
    assert isinstance(metrics.time_ms, np.ndarray)
    assert isinstance(metrics.input_signal, np.ndarray)
    assert isinstance(metrics.target_signal, np.ndarray)
    assert isinstance(metrics.output_signal, np.ndarray)
    assert isinstance(metrics.mse_input_output, float)
    assert isinstance(metrics.correlation, float)

    # Correlation should be high for similar signals
    assert metrics.correlation > 0.9


def test_compute_waveform_comparison_invalid_input():
    """Test that invalid inputs raise appropriate errors."""
    signal = np.random.randn(1000)

    # 2D signal should raise error
    with pytest.raises(ValueError, match="must be 1D"):
        compute_waveform_comparison(np.random.randn(10, 10), signal, signal, 48_000)

    # Shape mismatch should raise error
    with pytest.raises(ValueError, match="same shape"):
        compute_waveform_comparison(signal, np.random.randn(500), signal, 48_000)

    # Invalid sample rate should raise error
    with pytest.raises(ValueError, match="must be positive"):
        compute_waveform_comparison(signal, signal, signal, -1)

    # Invalid window/offset should raise error
    with pytest.raises(ValueError, match="window_ms must be positive"):
        compute_waveform_comparison(signal, signal, signal, 48_000, window_ms=0.0)
    with pytest.raises(ValueError, match="offset_ms must be non-negative"):
        compute_waveform_comparison(signal, signal, signal, 48_000, offset_ms=-1.0)


def test_plot_square_wave_response_creates_file(tmp_path: Path):
    """Test that square wave plot is created."""
    sample_rate = 88_200
    signal = np.concatenate([np.zeros(1000), np.ones(1000)])

    metrics = compute_square_wave_response(signal, sample_rate)
    output_path = tmp_path / "square_wave.png"

    plot_square_wave_response(metrics, metrics, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_impulse_response_creates_file(tmp_path: Path):
    """Test that impulse response plot is created."""
    sample_rate = 88_200
    signal = np.zeros(2000)
    signal[1000] = 1.0

    metrics = compute_impulse_response(signal, sample_rate)
    output_path = tmp_path / "impulse.png"

    plot_impulse_response(metrics, metrics, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_waveform_comparison_creates_file(tmp_path: Path):
    """Test that waveform comparison plot is created."""
    sample_rate = 88_200
    signal = np.sin(2 * np.pi * 1000 * np.arange(sample_rate) / sample_rate)

    metrics = compute_waveform_comparison(signal, signal, signal, sample_rate)
    output_path = tmp_path / "waveform.png"

    plot_waveform_comparison(metrics, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_square_wave_detects_overshoot():
    """Test that overshoot is detected."""
    sample_rate = 88_200

    # Create signal with overshoot
    base = np.concatenate([np.zeros(1000), np.ones(1000)])
    with_overshoot = base.copy()
    with_overshoot[1005:1010] = 1.2  # Add overshoot

    metrics_no_overshoot = compute_square_wave_response(base, sample_rate)
    metrics_with_overshoot = compute_square_wave_response(with_overshoot, sample_rate)

    # Signal with overshoot should have higher overshoot percentage
    assert (
        metrics_with_overshoot.overshoot_percent
        > metrics_no_overshoot.overshoot_percent
    )


def test_impulse_response_symmetry():
    """Test impulse response symmetry metric."""
    sample_rate = 88_200

    # Symmetric impulse (Gaussian-like)
    x = np.arange(-100, 101)
    symmetric = np.exp(-(x**2) / 100.0)

    # Asymmetric impulse (exponential decay)
    asymmetric = np.zeros(201)
    asymmetric[100] = 1.0
    asymmetric[101:] = np.exp(-np.arange(100) / 20.0)

    metrics_symmetric = compute_impulse_response(symmetric, sample_rate)
    metrics_asymmetric = compute_impulse_response(asymmetric, sample_rate)

    # Symmetric should have lower symmetry score
    assert metrics_symmetric.symmetry_score < metrics_asymmetric.symmetry_score


def test_waveform_comparison_perfect_match():
    """Test waveform comparison with perfect match."""
    sample_rate = 88_200
    signal = np.sin(2 * np.pi * 1000 * np.arange(sample_rate) / sample_rate)

    metrics = compute_waveform_comparison(signal, signal, signal, sample_rate)

    # Perfect match should have zero MSE and correlation = 1
    assert metrics.mse_input_output < 1e-10
    assert abs(metrics.correlation - 1.0) < 1e-6


def test_square_wave_settling_time_non_negative_after_transition():
    """Settling time should be measured from transition onward."""
    sample_rate = 88_200
    signal = np.concatenate([np.zeros(2_000), np.ones(2_000)])

    metrics = compute_square_wave_response(signal, sample_rate, transition_time_ms=1.0)

    assert metrics.settling_time_ms >= 0.0


def test_compute_edge_aligned_ringing_metrics_basic() -> None:
    """Edge-aligned ringing metrics should be computable for square waves."""
    sample_rate = 88_200
    signal = np.concatenate([np.zeros(2_000), np.ones(2_000)])

    metrics = compute_edge_aligned_ringing_metrics(
        signal=signal,
        sample_rate=sample_rate,
        plateau_start_ms=0.1,
        plateau_end_ms=0.8,
        ringing_window_ms=0.8,
    )

    assert isinstance(metrics, EdgeAlignedRingingMetrics)
    assert metrics.edge_index > 0
    assert metrics.plateau_ripple_rms >= 0.0
    assert metrics.plateau_ripple_p2p >= 0.0


def test_edge_aligned_metrics_are_polarity_invariant() -> None:
    """Falling and rising edges must measure the same physical ringing."""
    sample_rate = 88_200
    rising = np.concatenate([np.zeros(2_000), np.ones(2_000)])
    ripple = np.zeros_like(rising)
    ripple[2_010:2_050] = 0.02 * np.sin(np.linspace(0.0, 4.0 * np.pi, 40))
    rising = rising + ripple

    rising_metrics = compute_edge_aligned_ringing_metrics(rising, sample_rate)
    falling_metrics = compute_edge_aligned_ringing_metrics(-rising, sample_rate)

    assert falling_metrics.plateau_ripple_rms == pytest.approx(
        rising_metrics.plateau_ripple_rms
    )
    assert falling_metrics.plateau_ripple_p2p == pytest.approx(
        rising_metrics.plateau_ripple_p2p
    )
    assert falling_metrics.overshoot_abs == pytest.approx(rising_metrics.overshoot_abs)


def test_compare_edge_aligned_ringing_detects_regression() -> None:
    """Comparison API should report higher ripple ratio on degraded signal."""
    sample_rate = 88_200
    before = np.concatenate([np.zeros(2_000), np.ones(2_000)])
    after = before.copy()
    after[2_015:2_060] += 0.05 * np.sin(np.linspace(0.0, 6.0 * np.pi, 45))

    metrics = compare_edge_aligned_ringing(
        before_signal=before,
        after_signal=after,
        sample_rate=sample_rate,
    )

    assert isinstance(metrics, RingingComparisonMetrics)
    assert metrics.plateau_ripple_rms_ratio > 1.0
    assert metrics.plateau_ripple_p2p_ratio > 1.0


def test_compute_edge_aligned_ringing_metrics_raises_without_edge() -> None:
    """Signals without sign-change edge should be rejected."""
    with pytest.raises(ValueError, match="No sign-change edge"):
        compute_edge_aligned_ringing_metrics(
            signal=np.ones(4_000, dtype=np.float64),
            sample_rate=88_200,
        )


def test_compute_edge_aligned_ringing_metrics_rejects_plateau_beyond_signal() -> None:
    """Plateau window that starts beyond signal length should fail."""
    signal = np.concatenate([np.zeros(20), np.ones(20)]).astype(np.float64)
    with pytest.raises(ValueError, match="plateau window starts beyond signal length"):
        compute_edge_aligned_ringing_metrics(
            signal=signal,
            sample_rate=1_000,
            plateau_start_ms=50.0,
            plateau_end_ms=60.0,
            ringing_window_ms=1.0,
        )


def test_compute_edge_aligned_ringing_metrics_rejects_empty_plateau_after_rounding() -> (
    None
):
    """Rounded plateau offsets that collapse to zero-width should fail."""
    signal = np.concatenate([np.zeros(80), np.ones(80)]).astype(np.float64)
    with pytest.raises(ValueError, match="plateau window is empty"):
        compute_edge_aligned_ringing_metrics(
            signal=signal,
            sample_rate=1_000,
            plateau_start_ms=0.1,
            plateau_end_ms=0.2,
            ringing_window_ms=1.0,
        )


def test_waveform_comparison_constant_signals_is_finite():
    """Constant windows should not produce NaN correlation."""
    sample_rate = 88_200
    input_signal = np.ones(1_000, dtype=np.float64)
    output_signal = np.ones(1_000, dtype=np.float64)

    metrics = compute_waveform_comparison(
        input_signal=input_signal,
        target_signal=input_signal,
        output_signal=output_signal,
        sample_rate=sample_rate,
        window_ms=5.0,
    )

    assert metrics.correlation == 1.0
