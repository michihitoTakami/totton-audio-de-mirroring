"""Edge-rich and music-like synthetic signal families for CAPB training.

These generators close the coverage gap found in Phase 0/1: the original
signal set contained no squares, plateaus, or isolated clicks, so the exact
probe family used by the ringing gates was out-of-distribution for training.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import signal as sp_signal

DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_DURATION_SEC = 1.0
DEFAULT_AMPLITUDE = 0.9


def generate_square_wave(
    frequency_hz: float = 500.0,
    duty: float = 0.5,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate an ideal square wave with adjustable duty cycle.

    Args:
        frequency_hz: Fundamental frequency in Hz.
        duty: Duty cycle in (0, 1).
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude.
        rng: Optional RNG (unused; squares are deterministic).

    Returns:
        Square-wave signal as float32.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Square-wave discontinuities carry a 1/n harmonic series through the
        Nyquist limit, making plateau ripple after band manipulation directly
        observable - the primary ringing failure mode this system gates on.
    """
    _validate_common(sample_rate, duration_sec, amplitude)
    _validate_optional_rng(rng)
    if not 0.0 < frequency_hz < sample_rate / 2:
        raise ValueError(f"frequency_hz must be in (0, Nyquist), got {frequency_hz}.")
    if not 0.0 < duty < 1.0:
        raise ValueError(f"duty must be in (0, 1), got {duty}.")

    time_axis = _time_axis(sample_rate, duration_sec)
    wave = sp_signal.square(2.0 * np.pi * frequency_hz * time_axis, duty=duty)
    return (amplitude * np.asarray(wave, dtype=np.float64)).astype(np.float32)


def generate_dense_square_wave(
    frequency_hz: float = 5_000.0,
    duty: float = 0.5,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a dense-edge square wave for CAPB transient training.

    Args:
        frequency_hz: Fundamental frequency in Hz.
        duty: Duty cycle in (0, 1).
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude.
        rng: Optional RNG (unused; squares are deterministic).

    Returns:
        Dense square-wave signal as float32.

    Physical Basis:
        Above roughly 4 kHz the intervals between edges are too short for
        the plateau mask to expose settling ripple. Keeping this family
        distinct lets the dataset apply generator-labelled edge supervision
        without enabling a slope detector on stationary broadband noise.
    """
    return generate_square_wave(
        frequency_hz=frequency_hz,
        duty=duty,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        amplitude=amplitude,
        rng=rng,
    )


def generate_sawtooth_wave(
    frequency_hz: float = 500.0,
    width: float = 1.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a sawtooth (width=1) or triangle (width=0.5) wave.

    Args:
        frequency_hz: Fundamental frequency in Hz.
        width: Rising-ramp fraction; 1.0 is sawtooth, 0.5 is triangle.
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude.
        rng: Optional RNG (unused; deterministic).

    Returns:
        Sawtooth/triangle signal as float32.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Sawtooth waves put a dense 1/n harmonic series across the 20 kHz
        boundary - the hardest steady-state case for mirror suppression
        without touching legitimate near-Nyquist content.
    """
    _validate_common(sample_rate, duration_sec, amplitude)
    _validate_optional_rng(rng)
    if not 0.0 < frequency_hz < sample_rate / 2:
        raise ValueError(f"frequency_hz must be in (0, Nyquist), got {frequency_hz}.")
    if not 0.0 <= width <= 1.0:
        raise ValueError(f"width must be in [0, 1], got {width}.")

    time_axis = _time_axis(sample_rate, duration_sec)
    wave = sp_signal.sawtooth(2.0 * np.pi * frequency_hz * time_axis, width=width)
    return (amplitude * np.asarray(wave, dtype=np.float64)).astype(np.float32)


def generate_step_plateau(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
    plateau_ms_range: tuple[float, float] = (5.0, 100.0),
    slew_ms_range: tuple[float, float] = (0.05, 2.0),
    num_levels: int = 4,
) -> np.ndarray:
    """Generate a random-telegraph signal with plateaus and finite slews.

    Args:
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude.
        rng: RNG for plateau lengths, levels, and slew rates.
        plateau_ms_range: Plateau duration range in milliseconds.
        slew_ms_range: Edge ramp duration range in milliseconds.
        num_levels: Number of quantized levels.

    Returns:
        Step/plateau signal as float32.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Plateaus separated by edges of varying slew rate expose exactly the
        edge-adjacent ripple the ringing gates measure, over a continuum of
        edge bandwidths rather than only ideal discontinuities.
    """
    _validate_common(sample_rate, duration_sec, amplitude)
    if rng is None:
        raise ValueError("step_plateau requires an RNG.")
    if num_levels < 2:
        raise ValueError(f"num_levels must be >= 2, got {num_levels}.")

    num_samples = _num_samples(sample_rate, duration_sec)
    signal = np.empty(num_samples, dtype=np.float64)
    levels = np.linspace(-amplitude, amplitude, num_levels)

    position = 0
    current = float(rng.choice(levels))
    while position < num_samples:
        plateau_len = int(rng.uniform(*plateau_ms_range) * sample_rate / 1_000.0)
        plateau_len = max(1, plateau_len)
        end = min(num_samples, position + plateau_len)
        signal[position:end] = current
        position = end
        if position >= num_samples:
            break
        target = float(rng.choice(levels))
        slew_len = max(1, int(rng.uniform(*slew_ms_range) * sample_rate / 1_000.0))
        ramp_end = min(num_samples, position + slew_len)
        ramp = np.linspace(current, target, ramp_end - position, endpoint=False)
        signal[position:ramp_end] = ramp
        position = ramp_end
        current = target
    return signal.astype(np.float32)


def generate_tone_burst(
    frequency_hz: float = 5_000.0,
    burst_ms: float = 20.0,
    center_fraction: float | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a Hann-gated tone burst at a random position.

    Args:
        frequency_hz: Carrier frequency in Hz.
        burst_ms: Burst length in milliseconds.
        center_fraction: Optional normalized burst-center position in [0, 1].
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude.
        rng: RNG for burst position (centered if None).

    Returns:
        Tone-burst signal as float32.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        A gated tone is simultaneously tonal and transient - the decision-
        critical case for a controller that must trade mirror suppression
        against pre-echo at onsets.
    """
    _validate_common(sample_rate, duration_sec, amplitude)
    if not 0.0 < frequency_hz < sample_rate / 2:
        raise ValueError(f"frequency_hz must be in (0, Nyquist), got {frequency_hz}.")
    if burst_ms <= 0.0:
        raise ValueError(f"burst_ms must be positive, got {burst_ms}.")
    if center_fraction is not None and not 0.0 <= center_fraction <= 1.0:
        raise ValueError("center_fraction must be in [0, 1].")

    num_samples = _num_samples(sample_rate, duration_sec)
    burst_len = min(num_samples, max(3, int(burst_ms * sample_rate / 1_000.0)))
    if center_fraction is not None:
        center = int(round(center_fraction * (num_samples - 1)))
        start = int(np.clip(center - burst_len // 2, 0, num_samples - burst_len))
    elif rng is None:
        start = (num_samples - burst_len) // 2
    else:
        start = int(rng.integers(0, max(1, num_samples - burst_len)))

    local_time = np.arange(burst_len, dtype=np.float64) / sample_rate
    burst = np.sin(2.0 * np.pi * frequency_hz * local_time) * np.hanning(burst_len)
    signal = np.zeros(num_samples, dtype=np.float64)
    signal[start : start + burst_len] = amplitude * burst
    return signal.astype(np.float32)


def generate_isolated_click(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
    click_width_samples: int = 3,
) -> np.ndarray:
    """Generate an isolated click surrounded by silence.

    Args:
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude.
        rng: RNG for click position and polarity (centered if None).
        click_width_samples: Click width in samples.

    Returns:
        Click signal as float32.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Silence before an isolated wideband excitation makes pre-echo
        directly measurable: any energy before the click is an artifact of
        the interpolation kernel, not of the signal.
    """
    _validate_common(sample_rate, duration_sec, amplitude)
    if click_width_samples <= 0:
        raise ValueError("click_width_samples must be positive.")

    num_samples = _num_samples(sample_rate, duration_sec)
    if rng is None:
        start = num_samples // 2
        polarity = 1.0
    else:
        margin = max(1, num_samples // 8)
        start = int(rng.integers(margin, num_samples - margin))
        polarity = 1.0 if rng.uniform() < 0.5 else -1.0

    signal = np.zeros(num_samples, dtype=np.float64)
    width = min(click_width_samples, num_samples - start)
    signal[start : start + width] = polarity * amplitude * np.hanning(width + 2)[1:-1]
    return signal.astype(np.float32)


def generate_music_like_mixture(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
    f0_range_hz: tuple[float, float] = (60.0, 800.0),
    partial_count_range: tuple[int, int] = (6, 20),
) -> np.ndarray:
    """Generate a harmonic stack with envelopes, pink bed, and percussion.

    Args:
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude.
        rng: RNG for all stochastic choices.
        f0_range_hz: Fundamental frequency range.
        partial_count_range: Harmonic partial count range.

    Returns:
        Music-like mixture as float32.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Harmonic stacks with 1/n rolloff, amplitude envelopes, a pink-noise
        bed, and percussive hits approximate the spectro-temporal statistics
        of real music, closing the domain gap without real audio.
    """
    _validate_common(sample_rate, duration_sec, amplitude)
    if rng is None:
        raise ValueError("music_like_mixture requires an RNG.")

    num_samples = _num_samples(sample_rate, duration_sec)
    time_axis = _time_axis(sample_rate, duration_sec)

    f0 = float(rng.uniform(*f0_range_hz))
    partials = int(rng.integers(partial_count_range[0], partial_count_range[1] + 1))
    harmonic = np.zeros(num_samples, dtype=np.float64)
    for order in range(1, partials + 1):
        freq = f0 * order
        if freq >= sample_rate / 2:
            break
        phase = rng.uniform(0.0, 2.0 * np.pi)
        harmonic += np.sin(2.0 * np.pi * freq * time_axis + phase) / order
    harmonic *= _adsr_envelope(num_samples, rng)

    white = rng.standard_normal(num_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
    weights = np.ones_like(freqs)
    weights[1:] = 1.0 / np.sqrt(freqs[1:])
    weights[0] = 0.0
    pink = np.fft.irfft(spectrum * weights, n=num_samples)
    pink /= max(np.max(np.abs(pink)), 1e-12)

    percussion = np.zeros(num_samples, dtype=np.float64)
    for _ in range(int(rng.integers(1, 5))):
        start = int(rng.integers(0, max(1, num_samples - 100)))
        decay = rng.uniform(200.0, 2_000.0)
        length = min(num_samples - start, int(sample_rate * 0.05))
        local = np.arange(length, dtype=np.float64) / sample_rate
        percussion[start : start + length] += rng.standard_normal(length) * np.exp(
            -decay * local
        )

    mix = np.asarray(harmonic + 0.15 * pink + 0.5 * percussion, dtype=np.float64)
    peak = max(float(np.max(np.abs(mix))), 1e-12)
    return (amplitude * mix / peak).astype(np.float32)


def _adsr_envelope(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    attack = int(num_samples * rng.uniform(0.005, 0.1))
    decay = int(num_samples * rng.uniform(0.05, 0.2))
    release = int(num_samples * rng.uniform(0.1, 0.3))
    sustain_level = rng.uniform(0.4, 0.9)
    envelope = np.full(num_samples, sustain_level, dtype=np.float64)
    envelope[:attack] = np.linspace(0.0, 1.0, max(attack, 1))
    envelope[attack : attack + decay] = np.linspace(1.0, sustain_level, max(decay, 1))[
        : max(0, num_samples - attack)
    ]
    if release > 0:
        envelope[-release:] *= np.linspace(1.0, 0.0, release)
    return envelope


def _time_axis(sample_rate: int, duration_sec: float) -> np.ndarray:
    return np.arange(_num_samples(sample_rate, duration_sec), dtype=np.float64) / (
        sample_rate
    )


def _num_samples(sample_rate: int, duration_sec: float) -> int:
    return int(round(sample_rate * duration_sec))


def _validate_optional_rng(rng: np.random.Generator | None) -> None:
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator or None.")


def _validate_common(sample_rate: int, duration_sec: float, amplitude: float) -> None:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if duration_sec <= 0.0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}.")
    if amplitude <= 0.0:
        raise ValueError(f"amplitude must be positive, got {amplitude}.")


PROBE_FAMILY_GENERATORS: dict[str, Callable[..., np.ndarray]] = {
    "square_wave": generate_square_wave,
    "dense_square_wave": generate_dense_square_wave,
    "sawtooth_wave": generate_sawtooth_wave,
    "step_plateau": generate_step_plateau,
    "tone_burst": generate_tone_burst,
    "isolated_click": generate_isolated_click,
    "music_like_mixture": generate_music_like_mixture,
}
