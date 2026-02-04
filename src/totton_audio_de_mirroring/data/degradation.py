"""Degradation utilities for diverse upsampling SRC profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

DEFAULT_CUTOFF_RANGE = (18_000.0, 22_000.0)
DEFAULT_METHODS = (
    "zoh",
    "linear",
    "sinc_short",
    "sinc_long",
    "iir_bessel",
    "iir_butter",
)
DEFAULT_PHASE_MODES = ("linear", "minimum", "analog")
DEFAULT_QUANT_BITS = (16, 24)
DEFAULT_DITHER_MODES = ("none", "rectangular", "triangular")
DEFAULT_IIR_ORDER = 6
SHORT_SINC_TAPS = 64
LONG_SINC_TAPS = 256
DEFAULT_WINDOW = "hann"


@dataclass(frozen=True)
class DegradationConfig:
    """Configuration for degradation sampling.

    Args:
        cutoff_hz_range: Range of cutoff frequencies in Hz.
        phase_modes: Phase modes to sample.
        quantization_bits: Quantization bit depths.
        dither_modes: Dither modes for quantization.
        methods: Degradation method names.
        iir_order: Order for IIR-based SRC.
        seed: Optional RNG seed for reproducibility.

    Physical Basis:
        Diverse SRC degradations simulate realistic aliasing/mirror patterns
        and phase responses, improving generalization without synthesizing
        ultrasonic content.
    """

    cutoff_hz_range: tuple[float, float] = DEFAULT_CUTOFF_RANGE
    phase_modes: tuple[str, ...] = DEFAULT_PHASE_MODES
    quantization_bits: tuple[int, ...] = DEFAULT_QUANT_BITS
    dither_modes: tuple[str, ...] = DEFAULT_DITHER_MODES
    methods: tuple[str, ...] = DEFAULT_METHODS
    iir_order: int = DEFAULT_IIR_ORDER
    seed: int | None = None


@dataclass(frozen=True)
class DegradationProfile:
    """Concrete degradation profile sampled from configuration.

    Args:
        method: Degradation method name.
        cutoff_hz: Cutoff frequency in Hz.
        phase: Phase mode name.
        quantization_bits: Quantization bits.
        dither: Dither mode name.
        num_taps: FIR tap count (for sinc modes).
        iir_order: IIR order (for IIR modes).

    Physical Basis:
        Each profile encodes a plausible SRC degradation that preserves
        0–20kHz content while diversifying mirror/alias artifacts for training.
    """

    method: str
    cutoff_hz: float
    phase: str
    quantization_bits: int
    dither: str
    num_taps: int | None
    iir_order: int | None


class DegradationProfileManager:
    """Sample and apply degradation profiles with reproducible RNG.

    Physical Basis:
        Randomized degradations prevent overfitting to a single SRC artifact
        pattern while respecting the "no ultrasonic creation" constraint.
    """

    def __init__(self, config: DegradationConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        _validate_config(config)

    def sample_profile(
        self, rng: np.random.Generator | None = None
    ) -> DegradationProfile:
        """Sample a degradation profile.

        Args:
            rng: Optional RNG to override internal RNG.

        Returns:
            Sampled degradation profile.
        """
        generator = rng or self._rng
        cutoff_hz = float(
            generator.uniform(
                self._config.cutoff_hz_range[0],
                self._config.cutoff_hz_range[1],
            )
        )
        method = str(generator.choice(self._config.methods))
        quant_bits = int(generator.choice(self._config.quantization_bits))
        dither = str(generator.choice(self._config.dither_modes))

        if method in {"sinc_short", "sinc_long"}:
            phase_candidates = tuple(
                mode for mode in self._config.phase_modes if mode != "analog"
            )
            if not phase_candidates:
                raise ValueError("phase_modes must include linear or minimum for sinc.")
            phase = str(generator.choice(phase_candidates))
            num_taps = SHORT_SINC_TAPS if method == "sinc_short" else LONG_SINC_TAPS
            iir_order = None
        elif method in {"iir_bessel", "iir_butter"}:
            phase = "analog"
            num_taps = None
            iir_order = self._config.iir_order
        else:
            phase = "linear"
            num_taps = None
            iir_order = None

        return DegradationProfile(
            method=method,
            cutoff_hz=cutoff_hz,
            phase=phase,
            quantization_bits=quant_bits,
            dither=dither,
            num_taps=num_taps,
            iir_order=iir_order,
        )

    def apply(
        self,
        signal: np.ndarray,
        source_sr: int,
        target_sr: int,
        seed: int | None = None,
    ) -> tuple[np.ndarray, DegradationProfile]:
        """Apply a randomly sampled degradation to a signal.

        Args:
            signal: Input audio signal.
            source_sr: Source sample rate in Hz.
            target_sr: Target sample rate in Hz.
            seed: Optional seed for reproducible sampling.

        Returns:
            Tuple of (degraded_signal, profile).
        """
        generator = np.random.default_rng(seed) if seed is not None else self._rng
        profile = self.sample_profile(rng=generator)
        degraded = apply_degradation_profile(
            signal,
            source_sr,
            target_sr,
            profile,
            generator,
        )
        return degraded, profile


def apply_random_degradation(
    signal: np.ndarray,
    source_sr: int,
    target_sr: int,
    config: DegradationConfig | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, DegradationProfile]:
    """Convenience wrapper to apply a randomized degradation profile.

    Args:
        signal: Input signal.
        source_sr: Source sample rate in Hz.
        target_sr: Target sample rate in Hz.
        config: Optional degradation configuration.
        seed: Optional seed for reproducibility.

    Returns:
        Tuple of (degraded_signal, profile).

    Physical Basis:
        Randomized SRC degradations inject realistic mirror artifacts for
        training without altering the intent of low-band preservation.
    """
    manager = DegradationProfileManager(config or DegradationConfig())
    return manager.apply(signal, source_sr, target_sr, seed=seed)


def apply_degradation_profile(
    signal: np.ndarray,
    source_sr: int,
    target_sr: int,
    profile: DegradationProfile,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a specified degradation profile.

    Args:
        signal: Input signal (1D or 2D).
        source_sr: Source sample rate in Hz.
        target_sr: Target sample rate in Hz.
        profile: Degradation profile.
        rng: Random number generator for dithering.

    Returns:
        Degraded signal.

    Physical Basis:
        Each degradation emulates a plausible SRC artifact, which helps
        the model learn mirror suppression without encouraging hallucination.
    """
    _validate_signal(signal)
    _validate_sample_rate(source_sr)
    _validate_sample_rate(target_sr)
    ratio = _validate_upsample_ratio(source_sr, target_sr)

    upsampled = _apply_upsampling_method(signal, ratio, target_sr, profile)
    return apply_quantization(upsampled, profile.quantization_bits, profile.dither, rng)


def apply_quantization(
    signal: np.ndarray,
    bits: int,
    dither_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply bit-depth quantization with optional dithering.

    Args:
        signal: Input signal.
        bits: Quantization bits.
        dither_mode: "none", "rectangular", or "triangular".
        rng: Random number generator.

    Returns:
        Quantized signal.

    Physical Basis:
        Quantization and dither emulate common SRC post-processing artifacts
        that influence perceived high-frequency harshness.
    """
    _validate_positive_int(bits, "bits")
    _validate_dither_mode(dither_mode)

    step = 2.0 / (2**bits)
    noisy = np.asarray(signal, dtype=np.float64)

    if dither_mode == "rectangular":
        noise = rng.uniform(-0.5, 0.5, size=noisy.shape) * step
        noisy = noisy + noise
    elif dither_mode == "triangular":
        noise = (
            rng.uniform(-0.5, 0.5, size=noisy.shape)
            + rng.uniform(-0.5, 0.5, size=noisy.shape)
        ) * step
        noisy = noisy + noise

    quantized = np.round(noisy / step) * step
    return np.asarray(np.clip(quantized, -1.0, 1.0 - step), dtype=np.float64)


def _upsample_zoh(signal: np.ndarray, ratio: int) -> np.ndarray:
    return np.repeat(signal, ratio, axis=-1).astype(np.float64)


def _upsample_linear(signal: np.ndarray, ratio: int) -> np.ndarray:
    def interpolate(channel: np.ndarray) -> np.ndarray:
        num_samples = channel.shape[-1]
        original_x = np.arange(num_samples)
        target_x = np.linspace(0, num_samples - 1, num_samples * ratio)
        return np.asarray(np.interp(target_x, original_x, channel), dtype=np.float64)

    if signal.ndim == 1:
        return interpolate(signal)

    channels = [interpolate(channel) for channel in signal]
    return np.stack(channels, axis=0).astype(np.float64)


def _apply_upsampling_method(
    signal: np.ndarray,
    ratio: int,
    target_sr: int,
    profile: DegradationProfile,
) -> np.ndarray:
    if profile.method == "zoh":
        return _upsample_zoh(signal, ratio)
    if profile.method == "linear":
        return _upsample_linear(signal, ratio)
    if profile.method in {"sinc_short", "sinc_long"}:
        if profile.num_taps is None:
            raise ValueError("num_taps must be provided for sinc degradation.")
        return _upsample_sinc(
            signal,
            ratio,
            target_sr,
            profile.cutoff_hz,
            profile.num_taps,
            profile.phase,
        )
    if profile.method == "iir_bessel":
        if profile.iir_order is None:
            raise ValueError("iir_order must be provided for IIR degradation.")
        return _upsample_iir(
            signal,
            ratio,
            target_sr,
            profile.cutoff_hz,
            profile.iir_order,
            "bessel",
        )
    if profile.method == "iir_butter":
        if profile.iir_order is None:
            raise ValueError("iir_order must be provided for IIR degradation.")
        return _upsample_iir(
            signal,
            ratio,
            target_sr,
            profile.cutoff_hz,
            profile.iir_order,
            "butter",
        )
    raise ValueError(f"Unknown degradation method: {profile.method}.")


def _upsample_sinc(
    signal: np.ndarray,
    ratio: int,
    target_sr: int,
    cutoff_hz: float,
    num_taps: int,
    phase: str,
) -> np.ndarray:
    _validate_cutoff(cutoff_hz, target_sr)
    _validate_positive_int(num_taps, "num_taps")

    taps = sp_signal.firwin(
        num_taps,
        cutoff_hz,
        fs=target_sr,
        window=DEFAULT_WINDOW,
    )
    if phase == "minimum":
        taps = sp_signal.minimum_phase(taps, method="homomorphic")
    elif phase != "linear":
        raise ValueError(f"Unsupported phase mode for sinc: {phase}.")

    filtered = sp_signal.upfirdn(taps, signal, up=ratio, down=1, axis=-1)
    expected_len = signal.shape[-1] * ratio
    return np.asarray(filtered[..., :expected_len], dtype=np.float64)


def _upsample_iir(
    signal: np.ndarray,
    ratio: int,
    target_sr: int,
    cutoff_hz: float,
    order: int,
    kind: str,
) -> np.ndarray:
    _validate_cutoff(cutoff_hz, target_sr)
    _validate_positive_int(order, "order")

    upsampled = _zero_stuff(signal, ratio)

    if kind == "bessel":
        b, a = sp_signal.bessel(
            order,
            cutoff_hz,
            btype="lowpass",
            analog=False,
            output="ba",
            norm="phase",
            fs=target_sr,
        )
    elif kind == "butter":
        b, a = sp_signal.butter(
            order,
            cutoff_hz,
            btype="lowpass",
            analog=False,
            output="ba",
            fs=target_sr,
        )
    else:
        raise ValueError(f"Unsupported IIR kind: {kind}.")

    filtered = sp_signal.lfilter(b, a, upsampled, axis=-1)
    return np.asarray(filtered, dtype=np.float64)


def _zero_stuff(signal: np.ndarray, ratio: int) -> np.ndarray:
    output_shape = list(signal.shape)
    output_shape[-1] = output_shape[-1] * ratio
    upsampled = np.zeros(output_shape, dtype=np.float64)
    upsampled[..., ::ratio] = np.asarray(signal, dtype=np.float64)
    return upsampled


def _validate_signal(signal: np.ndarray) -> None:
    if signal.ndim not in (1, 2):
        raise ValueError(f"signal must be 1D or 2D, got {signal.ndim}D.")
    if signal.size == 0:
        raise ValueError("signal cannot be empty.")


def _validate_sample_rate(sample_rate: int) -> None:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")


def _validate_cutoff(cutoff_hz: float, sample_rate: int) -> None:
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}.")
    nyquist = sample_rate / 2
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"cutoff_hz must be less than Nyquist ({nyquist} Hz), got {cutoff_hz}."
        )


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_dither_mode(mode: str) -> None:
    if mode not in DEFAULT_DITHER_MODES:
        raise ValueError(f"Unsupported dither mode: {mode}.")


def _validate_upsample_ratio(source_sr: int, target_sr: int) -> int:
    if target_sr <= source_sr:
        raise ValueError("target_sr must be greater than source_sr.")
    ratio = target_sr / source_sr
    if not float(ratio).is_integer():
        raise ValueError("target_sr must be an integer multiple of source_sr.")
    return int(ratio)


def _validate_config(config: DegradationConfig) -> None:
    _validate_positive_int(config.iir_order, "iir_order")
    low, high = config.cutoff_hz_range
    if low <= 0 or high <= 0 or low >= high:
        raise ValueError("cutoff_hz_range must be positive and low < high.")
    if len(config.methods) < 5:
        raise ValueError("methods must include at least 5 entries.")
    if len(config.quantization_bits) == 0:
        raise ValueError("quantization_bits cannot be empty.")
    if len(config.dither_modes) == 0:
        raise ValueError("dither_modes cannot be empty.")
    if len(config.phase_modes) == 0:
        raise ValueError("phase_modes cannot be empty.")
