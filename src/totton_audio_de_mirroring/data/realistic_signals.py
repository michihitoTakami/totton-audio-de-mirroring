"""Procedural CAPB signals derived from real-audio failure modes."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def generate_damped_string(
    fundamental_hz: float,
    event_duration_ms: float,
    sample_rate: int,
    duration_sec: float,
    center_fraction: float = 0.5,
    amplitude: float = 0.9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a finite plucked-string-like harmonic decay.

    Args:
        fundamental_hz: Fundamental frequency in Hz.
        event_duration_ms: Length of the non-zero decay.
        sample_rate: Sample rate in Hz.
        duration_sec: Total signal duration.
        center_fraction: Event-center position as a signal fraction.
        amplitude: Peak output amplitude.
        rng: Random generator for phases and harmonic amplitudes.

    Returns:
        One-dimensional float32 waveform.

    Physical Basis:
        A broadband attack followed by an exponentially decaying harmonic
        body models the transition from ringing-sensitive onset to a safe,
        spectrally persistent string sustain without copying held-out audio.
    """
    _validate_common(sample_rate, duration_sec, center_fraction, amplitude)
    if fundamental_hz <= 0.0 or fundamental_hz >= sample_rate / 2.0:
        raise ValueError("fundamental_hz must lie below Nyquist.")
    event_samples = _event_samples(event_duration_ms, sample_rate)
    generator = rng or np.random.default_rng()
    time = np.arange(event_samples, dtype=np.float64) / sample_rate
    maximum_harmonic = min(16, int((0.95 * sample_rate / 2) // fundamental_hz))
    if maximum_harmonic < 1:
        raise ValueError("fundamental_hz leaves no valid harmonic.")
    body = np.zeros(event_samples, dtype=np.float64)
    for harmonic in range(1, maximum_harmonic + 1):
        decay = np.exp(-time * (18.0 + 2.5 * harmonic))
        phase = float(generator.uniform(-np.pi, np.pi))
        body += (
            decay
            * np.sin(2.0 * np.pi * fundamental_hz * harmonic * time + phase)
            / harmonic
        )
    attack_samples = max(2, round(0.0005 * sample_rate))
    body[:attack_samples] += generator.normal(0.0, 0.35, attack_samples)
    body *= np.exp(-5.0 * time / max(event_duration_ms / 1_000.0, 1.0e-6))
    return _place_event(body, sample_rate, duration_sec, center_fraction, amplitude)


def generate_clustered_impacts(
    impact_count: int,
    cluster_duration_ms: float,
    sample_rate: int,
    duration_sec: float,
    center_fraction: float = 0.5,
    amplitude: float = 0.9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a compact cluster of resonant impacts.

    Args:
        impact_count: Number of impacts in the cluster.
        cluster_duration_ms: Time spanned by impact onsets.
        sample_rate: Sample rate in Hz.
        duration_sec: Total signal duration.
        center_fraction: Cluster-center position.
        amplitude: Peak output amplitude.
        rng: Random generator for onset and resonance variation.

    Returns:
        One-dimensional float32 waveform.

    Physical Basis:
        Closely spaced hard impacts approximate ice collisions and exercise
        whether the controller can remain low-ringing across an event group
        instead of reacting only to a single ideal impulse.
    """
    _validate_common(sample_rate, duration_sec, center_fraction, amplitude)
    if impact_count <= 0:
        raise ValueError("impact_count must be positive.")
    cluster_samples = _event_samples(cluster_duration_ms, sample_rate)
    generator = rng or np.random.default_rng()
    tail_samples = max(4, round(0.02 * sample_rate))
    event = np.zeros(cluster_samples + tail_samples, dtype=np.float64)
    onsets = np.sort(generator.integers(0, cluster_samples, size=impact_count))
    time = np.arange(tail_samples, dtype=np.float64) / sample_rate
    for onset in onsets:
        resonance_hz = float(
            generator.uniform(1_000.0, min(12_000.0, sample_rate * 0.4))
        )
        tail = np.exp(-time * generator.uniform(180.0, 500.0))
        tail *= np.sin(2.0 * np.pi * resonance_hz * time)
        tail[0] += float(generator.uniform(0.6, 1.0))
        stop = min(event.size, int(onset) + tail_samples)
        event[int(onset) : stop] += tail[: stop - int(onset)]
    return _place_event(event, sample_rate, duration_sec, center_fraction, amplitude)


def generate_flowing_noise(
    low_hz: float,
    high_hz: float,
    modulation_hz: float,
    sample_rate: int,
    duration_sec: float,
    amplitude: float = 0.9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate stationary colored noise with slow turbulent modulation.

    Args:
        low_hz: Band-pass lower edge in Hz.
        high_hz: Band-pass upper edge in Hz.
        modulation_hz: Slow envelope modulation rate in Hz.
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration.
        amplitude: Peak output amplitude.
        rng: Random generator for the noise realization.

    Returns:
        One-dimensional float32 waveform.

    Physical Basis:
        Broadband water flow is locally stochastic but statistically steady;
        it therefore supplies a strong sharp-routing and no-modulation label.
    """
    _validate_common(sample_rate, duration_sec, 0.5, amplitude)
    if not 0.0 < low_hz < high_hz < sample_rate / 2.0:
        raise ValueError("noise band must satisfy 0 < low < high < Nyquist.")
    if modulation_hz <= 0.0:
        raise ValueError("modulation_hz must be positive.")
    generator = rng or np.random.default_rng()
    size = max(1, round(sample_rate * duration_sec))
    noise = generator.standard_normal(size)
    sos = sp_signal.butter(
        4, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos"
    )
    colored = sp_signal.sosfiltfilt(sos, noise)
    time = np.arange(size, dtype=np.float64) / sample_rate
    phase = float(generator.uniform(-np.pi, np.pi))
    envelope = 0.75 + 0.25 * np.sin(2.0 * np.pi * modulation_hz * time + phase)
    return _scale(colored * envelope, amplitude)


def generate_string_riff(
    fundamental_hz: float,
    interval_ms: float,
    sample_rate: int,
    duration_sec: float,
    amplitude: float = 0.9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate repeated overlapping plucked-string attacks.

    Args:
        fundamental_hz: Central fundamental frequency in Hz.
        interval_ms: Mean interval between attacks.
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration.
        amplitude: Peak output amplitude.
        rng: Random generator for timing, notes, and phases.

    Returns:
        One-dimensional float32 waveform.

    Physical Basis:
        Repeated attacks over a still-active decay remove the silence cue of
        isolated synthetic bursts and approximate dry guitar rhythm while
        retaining exact procedural onset labels.
    """
    _validate_common(sample_rate, duration_sec, 0.5, amplitude)
    if fundamental_hz <= 0.0 or interval_ms <= 0.0:
        raise ValueError("fundamental_hz and interval_ms must be positive.")
    generator = rng or np.random.default_rng()
    size = max(1, round(sample_rate * duration_sec))
    output = np.zeros(size, dtype=np.float64)
    interval = max(1, round(interval_ms * sample_rate / 1_000.0))
    tail_size = max(8, round(min(0.25, duration_sec) * sample_rate))
    onset = interval // 2
    while onset < size:
        note = fundamental_hz * float(generator.choice((0.75, 1.0, 1.25, 1.5)))
        time = np.arange(tail_size, dtype=np.float64) / sample_rate
        tail = np.zeros(tail_size, dtype=np.float64)
        for harmonic in range(1, 9):
            frequency = note * harmonic
            if frequency >= 0.45 * sample_rate:
                break
            phase = float(generator.uniform(-np.pi, np.pi))
            tail += (
                np.exp(-time * (15.0 + 3.0 * harmonic))
                * np.sin(2.0 * np.pi * frequency * time + phase)
                / harmonic
            )
        attack_size = min(tail_size, max(2, round(0.0007 * sample_rate)))
        tail[:attack_size] += generator.normal(0.0, 0.4, attack_size)
        stop = min(size, onset + tail_size)
        output[onset:stop] += tail[: stop - onset]
        jitter = int(generator.uniform(0.8, 1.2) * interval)
        onset += max(1, jitter)
    return _scale(output, amplitude)


def generate_impact_stream(
    event_rate_hz: float,
    sample_rate: int,
    duration_sec: float,
    amplitude: float = 0.9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a continuous irregular stream of resonant impacts.

    Args:
        event_rate_hz: Mean number of collisions per second.
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration.
        amplitude: Peak output amplitude.
        rng: Random generator for collision timing and resonances.

    Returns:
        One-dimensional float32 waveform.

    Physical Basis:
        Irregular impacts with overlapping resonant tails approximate ice
        movement without copying recorded content or inventing target HF.
    """
    _validate_common(sample_rate, duration_sec, 0.5, amplitude)
    if event_rate_hz <= 0.0:
        raise ValueError("event_rate_hz must be positive.")
    generator = rng or np.random.default_rng()
    size = max(1, round(sample_rate * duration_sec))
    output = np.zeros(size, dtype=np.float64)
    onset = 0
    tail_size = max(8, round(0.025 * sample_rate))
    time = np.arange(tail_size, dtype=np.float64) / sample_rate
    while onset < size:
        spacing = generator.exponential(sample_rate / event_rate_hz)
        onset += max(1, round(spacing))
        if onset >= size:
            break
        frequency = float(generator.uniform(800.0, min(14_000.0, sample_rate * 0.4)))
        tail = np.exp(-time * generator.uniform(120.0, 500.0))
        tail *= np.sin(2.0 * np.pi * frequency * time)
        tail[0] += float(generator.uniform(0.5, 1.0))
        stop = min(size, onset + tail_size)
        output[onset:stop] += tail[: stop - onset]
    return _scale(output, amplitude)


def _validate_common(
    sample_rate: int, duration_sec: float, center_fraction: float, amplitude: float
) -> None:
    if sample_rate <= 0 or duration_sec <= 0.0:
        raise ValueError("sample_rate and duration_sec must be positive.")
    if not 0.0 <= center_fraction <= 1.0:
        raise ValueError("center_fraction must lie in [0, 1].")
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive.")


def _event_samples(duration_ms: float, sample_rate: int) -> int:
    if duration_ms <= 0.0:
        raise ValueError("event duration must be positive.")
    return max(2, round(duration_ms * sample_rate / 1_000.0))


def _place_event(
    event: np.ndarray,
    sample_rate: int,
    duration_sec: float,
    center_fraction: float,
    amplitude: float,
) -> np.ndarray:
    size = max(1, round(sample_rate * duration_sec))
    if event.size >= size:
        raise ValueError("event must be shorter than the complete signal.")
    center = round(center_fraction * (size - 1))
    start = min(max(0, center - event.size // 2), size - event.size)
    output = np.zeros(size, dtype=np.float64)
    output[start : start + event.size] = event
    return _scale(output, amplitude)


def _scale(signal: np.ndarray, amplitude: float) -> np.ndarray:
    peak = float(np.max(np.abs(signal)))
    if peak <= 0.0:
        raise ValueError("generated signal must not be silent.")
    return np.asarray(signal * (amplitude / peak), dtype=np.float32)
