"""Synthetic audio generator utilities for training data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.probe_generators import (
    PROBE_FAMILY_GENERATORS,
)
from totton_audio_de_mirroring.data.realistic_signals import (
    generate_clustered_impacts,
    generate_damped_string,
    generate_flowing_noise,
    generate_impact_stream,
    generate_string_riff,
)

DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_DURATION_SEC = 1.0
DEFAULT_AMPLITUDE = 0.9
DEFAULT_SWEEP_START_HZ = 20.0
DEFAULT_SWEEP_END_HZ = 20_000.0
DEFAULT_IMPULSE_INTERVAL_SEC = 0.01
DEFAULT_BANDPASS_TAPS = 513


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for synthetic signal generation.

    Args:
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        seed: Optional RNG seed for reproducibility.

    Physical Basis:
        Consistent sampling parameters control the discrete-time resolution
        used for synthetic stimuli while keeping reproducible randomness.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    duration_sec: float = DEFAULT_DURATION_SEC
    seed: int | None = None


@dataclass(frozen=True)
class SignalRequest:
    """Definition of a signal request for batch generation.

    Args:
        signal_type: Registered signal type name.
        params: Keyword parameters passed to the generator.

    Physical Basis:
        Parameterized requests enable controllable stimulus diversity without
        embedding any target "ground truth" beyond spectral intent.
    """

    signal_type: str
    params: Mapping[str, float | int | Sequence[float] | None]


class SyntheticSignalGenerator:
    """Generate synthetic audio signals with a shared RNG.

    Physical Basis:
        A shared RNG provides reproducible procedural signals that emulate
        varied training stimuli without relying on ideal masters.
    """

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self._config = config or GeneratorConfig()
        _validate_sample_rate(self._config.sample_rate)
        _validate_duration(self._config.duration_sec)
        self._rng = np.random.default_rng(self._config.seed)

    def list_signal_types(self) -> tuple[str, ...]:
        """Return supported signal type names.

        Physical Basis:
            Enumerating available generators ensures controlled coverage of
            stimulus classes aligned with the dataset spec.
        """

        return list_signal_types()

    def generate(self, signal_type: str, **kwargs: object) -> np.ndarray:
        """Generate a synthetic signal by name.

        Args:
            signal_type: Registered signal type name.
            **kwargs: Parameters passed to the underlying generator.

        Returns:
            Generated signal as a 1D float array.

        Raises:
            ValueError: If the signal type is unknown.

        Physical Basis:
            Dispatching by signal type preserves a clear mapping between
            procedural intent and resulting spectral structure.
        """

        generator = _SIGNAL_GENERATORS.get(signal_type)
        if generator is None:
            raise ValueError(f"Unknown signal_type: {signal_type}")
        return generator(
            sample_rate=self._config.sample_rate,
            duration_sec=self._config.duration_sec,
            rng=self._rng,
            **kwargs,
        )

    def generate_batch(self, requests: Sequence[SignalRequest]) -> list[np.ndarray]:
        """Generate a batch of signals from requests.

        Args:
            requests: Sequence of signal generation requests.

        Returns:
            List of generated signals.

        Physical Basis:
            Batch generation supports dataset construction with diverse
            procedural content while keeping parameter control explicit.
        """

        return [self.generate(req.signal_type, **dict(req.params)) for req in requests]


def list_signal_types() -> tuple[str, ...]:
    """List available signal generator names.

    Returns:
        Tuple of supported signal type names.

    Physical Basis:
        Enumerating signal types helps ensure coverage of the synthetic
        stimuli described in the dataset specification.
    """

    return tuple(_SIGNAL_GENERATORS.keys())


def generate_signal(
    signal_type: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    seed: int | None = None,
    **kwargs: object,
) -> np.ndarray:
    """Generate a signal with explicit sampling parameters.

    Args:
        signal_type: Registered signal type name.
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        seed: Optional RNG seed for reproducibility.
        **kwargs: Parameters passed to the underlying generator.

    Returns:
        Generated signal as a 1D float array.

    Raises:
        ValueError: If the signal type is unknown.

    Physical Basis:
        Explicit sampling parameters ensure the discrete-time stimulus
        matches the intended training configuration.
    """

    _validate_sample_rate(sample_rate)
    _validate_duration(duration_sec)

    generator = _SIGNAL_GENERATORS.get(signal_type)
    if generator is None:
        raise ValueError(f"Unknown signal_type: {signal_type}")
    rng = np.random.default_rng(seed)
    return generator(
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        rng=rng,
        **kwargs,
    )


def generate_multitone(
    frequencies_hz: Sequence[float],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitudes: Sequence[float] | None = None,
    phases_rad: Sequence[float] | None = None,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a multitone signal with specified frequencies.

    Args:
        frequencies_hz: Sequence of tone frequencies in Hz.
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitudes: Optional per-tone amplitude weights.
        phases_rad: Optional per-tone initial phases in radians.
        amplitude: Peak amplitude after normalization.
        rng: Optional RNG (unused for deterministic tones).

    Returns:
        Multitone signal.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Summed sinusoids create a controlled harmonic or inharmonic spectrum
        used to stress mirror suppression without inventing new content.
    """

    _validate_sample_rate(sample_rate)
    _validate_duration(duration_sec)
    _validate_frequency_sequence(frequencies_hz, sample_rate)
    _validate_amplitude(amplitude)
    _validate_rng(rng)

    num_tones = len(frequencies_hz)
    if num_tones == 0:
        raise ValueError("frequencies_hz must not be empty.")

    weights = _normalize_weights(amplitudes, num_tones)
    phases = _normalize_phases(phases_rad, num_tones)

    time = _time_axis(sample_rate, duration_sec)
    tones = np.zeros_like(time, dtype=np.float64)
    for freq, weight, phase in zip(frequencies_hz, weights, phases, strict=True):
        tones = tones + weight * np.sin(2.0 * np.pi * freq * time + phase)

    return _scale_to_amplitude(tones, amplitude)


def generate_imd_two_tone(
    low_tone_hz: float,
    high_tone_hz: float,
    amplitude_ratio: float = 4.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a low/high two-tone probe for IMD-resistant training.

    Args:
        low_tone_hz: Low-frequency primary in Hz.
        high_tone_hz: High-frequency primary in Hz.
        amplitude_ratio: Low-to-high peak-amplitude ratio.
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude after normalization.
        rng: Optional RNG (unused for deterministic tones).

    Returns:
        Peak-normalized two-tone waveform.

    Raises:
        ValueError: If frequencies or the amplitude ratio are invalid.

    Physical Basis:
        A fixed linear interpolator preserves only the two primary lines.
        Training on randomized low/high pairs exposes controller-induced
        amplitude modulation without teaching exact gate frequencies.
    """
    if low_tone_hz >= high_tone_hz:
        raise ValueError("low_tone_hz must be lower than high_tone_hz.")
    if not np.isfinite(amplitude_ratio) or amplitude_ratio <= 0.0:
        raise ValueError("amplitude_ratio must be finite and positive.")
    return generate_multitone(
        frequencies_hz=(low_tone_hz, high_tone_hz),
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        amplitudes=(amplitude_ratio, 1.0),
        amplitude=amplitude,
        rng=rng,
    )


def generate_linear_sweep(
    start_hz: float = DEFAULT_SWEEP_START_HZ,
    end_hz: float = DEFAULT_SWEEP_END_HZ,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a linear frequency sweep.

    Args:
        start_hz: Start frequency in Hz.
        end_hz: End frequency in Hz.
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude after normalization.
        rng: Optional RNG (unused for deterministic sweeps).

    Returns:
        Linear chirp signal.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Linear sweeps provide uniform coverage of frequency response and
        expose mirror artifacts across the band.
    """

    return _generate_sweep(
        start_hz=start_hz,
        end_hz=end_hz,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        amplitude=amplitude,
        method="linear",
        rng=rng,
    )


def generate_log_sweep(
    start_hz: float = DEFAULT_SWEEP_START_HZ,
    end_hz: float = DEFAULT_SWEEP_END_HZ,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a logarithmic frequency sweep.

    Args:
        start_hz: Start frequency in Hz.
        end_hz: End frequency in Hz.
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude after normalization.
        rng: Optional RNG (unused for deterministic sweeps).

    Returns:
        Logarithmic chirp signal.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Log sweeps emphasize low-frequency resolution and mimic perceptual
        spacing, capturing mirror patterns across octaves.
    """

    return _generate_sweep(
        start_hz=start_hz,
        end_hz=end_hz,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        amplitude=amplitude,
        method="logarithmic",
        rng=rng,
    )


def generate_impulse_train(
    interval_sec: float = DEFAULT_IMPULSE_INTERVAL_SEC,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate an impulse train signal.

    Args:
        interval_sec: Interval between impulses in seconds.
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude of impulses.
        rng: Optional RNG (unused for deterministic impulses).

    Returns:
        Impulse train signal.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Impulse trains yield comb spectra that reveal aliasing and mirror
        behavior across evenly spaced harmonics.
    """

    _validate_sample_rate(sample_rate)
    _validate_duration(duration_sec)
    _validate_positive_float(interval_sec, "interval_sec")
    _validate_amplitude(amplitude)
    _validate_rng(rng)

    num_samples = _num_samples(sample_rate, duration_sec)
    interval_samples = int(round(interval_sec * sample_rate))
    if interval_samples <= 0:
        raise ValueError("interval_sec too small for sample_rate.")

    signal = np.zeros(num_samples, dtype=np.float64)
    indices = np.arange(0, num_samples, interval_samples)
    signal[indices] = amplitude
    return signal.astype(np.float32)


def generate_percussive_transient(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    decay_rate: float = 8.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a percussive transient signal.

    Args:
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude after normalization.
        decay_rate: Exponential decay rate for the envelope.
        rng: Optional RNG for noise generation.

    Returns:
        Percussive transient signal.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Noise bursts with fast decay approximate percussive transients and
        stress time-response preservation in the audible band.
    """

    _validate_sample_rate(sample_rate)
    _validate_duration(duration_sec)
    _validate_amplitude(amplitude)
    _validate_positive_float(decay_rate, "decay_rate")
    _validate_rng(rng)

    generator = rng or np.random.default_rng()
    num_samples = _num_samples(sample_rate, duration_sec)
    time = _time_axis(sample_rate, duration_sec)
    envelope = np.exp(-decay_rate * time / max(duration_sec, 1e-6))
    noise = generator.normal(0.0, 1.0, num_samples)
    transient = noise * envelope
    transient[0] += 1.0

    return _scale_to_amplitude(transient, amplitude)


def generate_am_tone(
    carrier_hz: float,
    mod_hz: float,
    modulation_index: float = 0.5,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate an amplitude-modulated tone.

    Args:
        carrier_hz: Carrier frequency in Hz.
        mod_hz: Modulation frequency in Hz.
        modulation_index: Modulation depth (0 to 1).
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude after normalization.
        rng: Optional RNG (unused for deterministic AM).

    Returns:
        AM tone signal.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        AM produces symmetric sidebands around the carrier, providing
        structured spectral content to evaluate mirror suppression.
    """

    _validate_sample_rate(sample_rate)
    _validate_duration(duration_sec)
    _validate_frequency(carrier_hz, sample_rate, "carrier_hz")
    _validate_frequency(mod_hz, sample_rate, "mod_hz")
    _validate_amplitude(amplitude)
    _validate_rng(rng)
    if not (0.0 <= modulation_index <= 1.0):
        raise ValueError("modulation_index must be between 0 and 1.")

    time = _time_axis(sample_rate, duration_sec)
    modulation = 1.0 + modulation_index * np.sin(2.0 * np.pi * mod_hz * time)
    tone = modulation * np.sin(2.0 * np.pi * carrier_hz * time)
    return _scale_to_amplitude(tone, amplitude)


def generate_fm_tone(
    carrier_hz: float,
    mod_hz: float,
    modulation_index: float = 2.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a frequency-modulated tone.

    Args:
        carrier_hz: Carrier frequency in Hz.
        mod_hz: Modulation frequency in Hz.
        modulation_index: Modulation index (beta).
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude after normalization.
        rng: Optional RNG (unused for deterministic FM).

    Returns:
        FM tone signal.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        FM produces multiple sidebands governed by Bessel functions,
        increasing spectral richness without creating ultrasonic content.
    """

    _validate_sample_rate(sample_rate)
    _validate_duration(duration_sec)
    _validate_frequency(carrier_hz, sample_rate, "carrier_hz")
    _validate_frequency(mod_hz, sample_rate, "mod_hz")
    _validate_amplitude(amplitude)
    _validate_positive_float(modulation_index, "modulation_index")
    _validate_rng(rng)

    time = _time_axis(sample_rate, duration_sec)
    phase = 2.0 * np.pi * carrier_hz * time
    modulator = modulation_index * np.sin(2.0 * np.pi * mod_hz * time)
    tone = np.sin(phase + modulator)
    return _scale_to_amplitude(tone, amplitude)


def generate_white_noise(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate white noise.

    Args:
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude after normalization.
        rng: Optional RNG for noise generation.

    Returns:
        White noise signal.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        White noise has approximately flat spectral density, useful for
        probing uniform mirror suppression across frequencies.
    """

    _validate_sample_rate(sample_rate)
    _validate_duration(duration_sec)
    _validate_amplitude(amplitude)
    _validate_rng(rng)

    generator = rng or np.random.default_rng()
    num_samples = _num_samples(sample_rate, duration_sec)
    noise = generator.normal(0.0, 1.0, num_samples)
    return _scale_to_amplitude(noise, amplitude)


def generate_pink_noise(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate pink (1/f) noise.

    Args:
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude after normalization.
        rng: Optional RNG for noise generation.

    Returns:
        Pink noise signal.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Pink noise emphasizes low frequencies with a 1/f spectrum, offering
        a complementary stress case to flat-spectrum white noise.
    """

    _validate_sample_rate(sample_rate)
    _validate_duration(duration_sec)
    _validate_amplitude(amplitude)
    _validate_rng(rng)

    generator = rng or np.random.default_rng()
    num_samples = _num_samples(sample_rate, duration_sec)
    white = generator.normal(0.0, 1.0, num_samples)

    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples, 1.0 / sample_rate)
    scaling = np.ones_like(freqs)
    scaling[1:] = 1.0 / np.sqrt(freqs[1:])
    shaped = spectrum * scaling
    pink = np.fft.irfft(shaped, n=num_samples)

    return _scale_to_amplitude(pink, amplitude)


def generate_band_limited_noise(
    low_hz: float,
    high_hz: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    num_taps: int = DEFAULT_BANDPASS_TAPS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate band-limited noise via FIR bandpass filtering.

    Args:
        low_hz: Low cutoff frequency in Hz.
        high_hz: High cutoff frequency in Hz.
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude after normalization.
        num_taps: FIR tap count for bandpass filter.
        rng: Optional RNG for noise generation.

    Returns:
        Band-limited noise signal.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Band-limited noise isolates a frequency region, enabling targeted
        evaluation of mirror suppression in specific bands.
    """

    _validate_sample_rate(sample_rate)
    _validate_duration(duration_sec)
    _validate_amplitude(amplitude)
    _validate_rng(rng)
    _validate_positive_int(num_taps, "num_taps")
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd for symmetric FIR design.")
    _validate_frequency(low_hz, sample_rate, "low_hz")
    _validate_frequency(high_hz, sample_rate, "high_hz")
    if low_hz >= high_hz:
        raise ValueError("low_hz must be less than high_hz.")

    generator = rng or np.random.default_rng()
    num_samples = _num_samples(sample_rate, duration_sec)
    noise = generator.normal(0.0, 1.0, num_samples)

    taps = sp_signal.firwin(
        num_taps,
        [low_hz, high_hz],
        pass_zero=False,
        fs=sample_rate,
    )
    filtered = sp_signal.lfilter(taps, [1.0], noise)

    return _scale_to_amplitude(filtered, amplitude)


def generate_soft_clipped_tone(
    frequency_hz: float,
    drive: float = 2.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
    amplitude: float = DEFAULT_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a soft-clipped sine tone.

    Args:
        frequency_hz: Base sine frequency in Hz.
        drive: Drive amount (>0 increases saturation).
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude after normalization.
        rng: Optional RNG (unused for deterministic tone).

    Returns:
        Soft-clipped tone signal.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Soft clipping introduces controlled harmonics to emulate mild
        nonlinearity without aggressive high-frequency creation.
    """

    _validate_sample_rate(sample_rate)
    _validate_duration(duration_sec)
    _validate_frequency(frequency_hz, sample_rate, "frequency_hz")
    _validate_amplitude(amplitude)
    _validate_positive_float(drive, "drive")
    _validate_rng(rng)

    time = _time_axis(sample_rate, duration_sec)
    sine = np.sin(2.0 * np.pi * frequency_hz * time)
    clipped = apply_soft_clip(sine, drive)
    return _scale_to_amplitude(clipped, amplitude)


def apply_soft_clip(signal: np.ndarray, drive: float = 2.0) -> np.ndarray:
    """Apply soft clipping to a signal.

    Args:
        signal: Input signal array.
        drive: Drive amount (>0 increases saturation).

    Returns:
        Soft-clipped signal.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Soft clipping models gentle saturation that adds harmonics while
        avoiding harsh discontinuities or hard clipping artifacts.
    """

    _validate_signal(signal)
    _validate_positive_float(drive, "drive")

    return np.tanh(signal * drive)


_SIGNAL_GENERATORS: dict[str, Callable[..., np.ndarray]] = {
    "multitone": generate_multitone,
    "imd_two_tone": generate_imd_two_tone,
    "sweep_linear": generate_linear_sweep,
    "sweep_log": generate_log_sweep,
    "impulse_train": generate_impulse_train,
    "percussive": generate_percussive_transient,
    "am_tone": generate_am_tone,
    "fm_tone": generate_fm_tone,
    "white_noise": generate_white_noise,
    "pink_noise": generate_pink_noise,
    "band_limited_noise": generate_band_limited_noise,
    "soft_clipped_tone": generate_soft_clipped_tone,
    "damped_string": generate_damped_string,
    "clustered_impacts": generate_clustered_impacts,
    "flowing_noise": generate_flowing_noise,
    "string_riff": generate_string_riff,
    "impact_stream": generate_impact_stream,
    **PROBE_FAMILY_GENERATORS,
}


def _generate_sweep(
    start_hz: float,
    end_hz: float,
    sample_rate: int,
    duration_sec: float,
    amplitude: float,
    method: str,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Internal helper to generate frequency sweeps.

    Args:
        start_hz: Start frequency in Hz.
        end_hz: End frequency in Hz.
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.
        amplitude: Peak amplitude after normalization.
        method: Sweep method for scipy chirp.
        rng: Optional RNG (unused for deterministic sweeps).

    Returns:
        Sweep signal.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Chirp generation sweeps frequency content to probe mirror artifacts
        across the band while maintaining controlled amplitude.
    """

    _validate_sample_rate(sample_rate)
    _validate_duration(duration_sec)
    _validate_frequency(start_hz, sample_rate, "start_hz")
    _validate_frequency(end_hz, sample_rate, "end_hz")
    _validate_amplitude(amplitude)
    _validate_rng(rng)
    if start_hz >= end_hz:
        raise ValueError("start_hz must be less than end_hz.")

    time = _time_axis(sample_rate, duration_sec)
    sweep = sp_signal.chirp(
        time,
        f0=start_hz,
        f1=end_hz,
        t1=duration_sec,
        method=method,
    )

    return _scale_to_amplitude(sweep, amplitude)


def _normalize_weights(
    amplitudes: Sequence[float] | None, num_tones: int
) -> np.ndarray:
    """Normalize per-tone amplitudes.

    Args:
        amplitudes: Optional per-tone amplitudes.
        num_tones: Expected number of tones.

    Returns:
        Normalized amplitude weights.

    Raises:
        ValueError: If amplitudes are invalid.

    Physical Basis:
        Normalized weights prevent single tones from dominating composite
        spectra while keeping relative energy ratios controlled.
    """

    if amplitudes is None:
        return np.ones(num_tones, dtype=np.float64) / num_tones
    if len(amplitudes) != num_tones:
        raise ValueError("amplitudes length must match frequencies_hz length.")
    weights = np.asarray(amplitudes, dtype=np.float64)
    if np.any(weights < 0.0):
        raise ValueError("amplitudes must be non-negative.")
    total = float(np.sum(weights))
    if total == 0.0:
        raise ValueError("amplitudes sum must be positive.")
    return weights / total


def _normalize_phases(phases: Sequence[float] | None, num_tones: int) -> np.ndarray:
    """Normalize per-tone phases.

    Args:
        phases: Optional per-tone phase angles.
        num_tones: Expected number of tones.

    Returns:
        Phase array.

    Raises:
        ValueError: If phases are invalid.

    Physical Basis:
        Explicit phases allow deterministic multitone waveforms when needed,
        without altering spectral magnitudes.
    """

    if phases is None:
        return np.zeros(num_tones, dtype=np.float64)
    if len(phases) != num_tones:
        raise ValueError("phases_rad length must match frequencies_hz length.")
    return np.asarray(phases, dtype=np.float64)


def _scale_to_amplitude(signal: np.ndarray, amplitude: float) -> np.ndarray:
    """Scale a signal to a target peak amplitude.

    Args:
        signal: Input signal.
        amplitude: Target peak amplitude.

    Returns:
        Scaled signal as float32.

    Physical Basis:
        Peak normalization standardizes dynamic range across stimuli without
        altering their spectral characteristics.
    """

    max_abs = float(np.max(np.abs(signal)))
    if max_abs == 0.0:
        return signal.astype(np.float32)
    scaled = signal * (amplitude / max_abs)
    return scaled.astype(np.float32)


def _time_axis(sample_rate: int, duration_sec: float) -> np.ndarray:
    """Create a time axis array.

    Args:
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.

    Returns:
        Time axis array.

    Physical Basis:
        Uniform sampling ensures consistent discrete-time modeling of
        continuous waveforms.
    """

    num_samples = _num_samples(sample_rate, duration_sec)
    return np.arange(num_samples, dtype=np.float64) / float(sample_rate)


def _num_samples(sample_rate: int, duration_sec: float) -> int:
    """Compute number of samples for duration.

    Args:
        sample_rate: Sample rate in Hz.
        duration_sec: Signal duration in seconds.

    Returns:
        Number of samples (>=1).

    Physical Basis:
        Mapping duration to discrete samples preserves timing fidelity in
        generated signals.
    """

    return max(1, int(round(duration_sec * sample_rate)))


def _validate_sample_rate(sample_rate: int) -> None:
    """Validate sample rate input.

    Args:
        sample_rate: Sample rate in Hz.

    Raises:
        ValueError: If sample_rate is invalid.

    Physical Basis:
        Positive sample rates ensure meaningful discrete-time sampling.
    """

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")


def _validate_duration(duration_sec: float) -> None:
    """Validate duration input.

    Args:
        duration_sec: Signal duration in seconds.

    Raises:
        ValueError: If duration_sec is invalid.

    Physical Basis:
        Positive durations ensure signals contain temporal information.
    """

    if duration_sec <= 0.0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}.")


def _validate_frequency(value: float, sample_rate: int, name: str) -> None:
    """Validate a frequency value.

    Args:
        value: Frequency in Hz.
        sample_rate: Sample rate in Hz.
        name: Field name for error messages.

    Raises:
        ValueError: If frequency is invalid.

    Physical Basis:
        Frequencies must be within Nyquist to avoid aliasing in synthesis.
    """

    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    nyquist = sample_rate / 2.0
    if value >= nyquist:
        raise ValueError(
            f"{name} must be less than Nyquist ({nyquist} Hz), got {value}."
        )


def _validate_frequency_sequence(
    frequencies_hz: Sequence[float], sample_rate: int
) -> None:
    """Validate a sequence of frequencies.

    Args:
        frequencies_hz: Sequence of frequencies in Hz.
        sample_rate: Sample rate in Hz.

    Raises:
        ValueError: If any frequency is invalid.

    Physical Basis:
        Ensuring all tones lie below Nyquist avoids unintended aliasing.
    """

    for freq in frequencies_hz:
        _validate_frequency(float(freq), sample_rate, "frequency")


def _validate_amplitude(amplitude: float) -> None:
    """Validate amplitude input.

    Args:
        amplitude: Target amplitude.

    Raises:
        ValueError: If amplitude is invalid.

    Physical Basis:
        Amplitudes must be positive to represent valid audio magnitudes.
    """

    if amplitude <= 0.0:
        raise ValueError(f"amplitude must be positive, got {amplitude}.")


def _validate_rng(rng: np.random.Generator | None) -> None:
    """Validate an optional RNG instance.

    Args:
        rng: Optional numpy random Generator.

    Raises:
        ValueError: If rng is not a numpy Generator instance.

    Physical Basis:
        Explicit RNG validation keeps procedural randomness controlled and
        reproducible without altering deterministic waveforms.
    """

    if rng is not None and not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator or None.")


def _validate_positive_float(value: float, name: str) -> None:
    """Validate a positive float.

    Args:
        value: Numeric value.
        name: Field name for error messages.

    Raises:
        ValueError: If value is invalid.

    Physical Basis:
        Positive parameters prevent degenerate time-domain signals.
    """

    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_positive_int(value: int, name: str) -> None:
    """Validate a positive integer.

    Args:
        value: Integer value.
        name: Field name for error messages.

    Raises:
        ValueError: If value is invalid.

    Physical Basis:
        Positive integers are required for counts like FIR taps.
    """

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_signal(signal: np.ndarray) -> None:
    """Validate a signal array.

    Args:
        signal: Input signal array.

    Raises:
        ValueError: If signal is invalid.

    Physical Basis:
        Non-empty arrays ensure defined waveform content.
    """

    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got {signal.ndim}D.")
    if signal.size == 0:
        raise ValueError("signal must not be empty.")


__all__ = [
    "GeneratorConfig",
    "SignalRequest",
    "SyntheticSignalGenerator",
    "apply_soft_clip",
    "generate_am_tone",
    "generate_band_limited_noise",
    "generate_fm_tone",
    "generate_impulse_train",
    "generate_linear_sweep",
    "generate_log_sweep",
    "generate_multitone",
    "generate_percussive_transient",
    "generate_pink_noise",
    "generate_signal",
    "generate_soft_clipped_tone",
    "generate_white_noise",
    "list_signal_types",
]
