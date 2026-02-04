"""Band-split architecture utilities for Stage 1 processing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from totton_audio_de_mirroring.data.filters import band_split, design_band_split_filters

DEFAULT_CUTOFF_HZ = 20_000.0
DEFAULT_SAMPLE_RATE = 88_200
DEFAULT_NUM_TAPS = 4097
DEFAULT_WINDOW = "hamming"


@dataclass(frozen=True)
class BandSplitConfig:
    """Configuration for band splitting.

    Args:
        cutoff_hz: Cutoff frequency in Hz.
        sample_rate: Sample rate in Hz.
        num_taps: Number of FIR taps (must be odd).
        window: Window name for FIR design.

    Physical Basis:
        Linear-phase FIR filters achieve complementary low/high splitting
        with a constant group delay, preserving low-band phase integrity.
    """

    cutoff_hz: float = DEFAULT_CUTOFF_HZ
    sample_rate: int = DEFAULT_SAMPLE_RATE
    num_taps: int = DEFAULT_NUM_TAPS
    window: str | tuple[str, float] = DEFAULT_WINDOW

    def __post_init__(self) -> None:
        """Validate configuration values at construction."""
        _validate_config(self)
    def delay_samples(self) -> int:
        """Return the constant group delay in samples.

        Returns:
            Constant group delay introduced by the linear-phase FIR.

        Physical Basis:
            A linear-phase FIR filter introduces a constant group delay of
            (num_taps - 1) / 2 samples across the passband.
        """
        return (self.num_taps - 1) // 2


@dataclass(frozen=True)
class BandSplitResult:
    """Result container for band-split processing.

    Args:
        low_band: Low-band output (0-20kHz).
        high_band: High-band output (20-44kHz).
        recombined: Recombined signal (low + high).
        delay_samples: Group delay in samples.

    Physical Basis:
        Low-band bypass plus high-band processing preserves the audible
        band while allowing targeted suppression above 20kHz.
    """

    low_band: np.ndarray
    high_band: np.ndarray
    recombined: np.ndarray
    delay_samples: int


class BandSplitProcessor:
    """Band-split processor with low-band bypass.

    Args:
        config: Band-split configuration.

    Raises:
        ValueError: If configuration values are invalid.

    Physical Basis:
        Splitting the spectrum at 20kHz allows the low band to bypass any
        learned processing, guaranteeing audible-band preservation.
    """

    def __init__(self, config: BandSplitConfig) -> None:
        _validate_config(config)
        lowpass_taps, highpass_taps = design_band_split_filters(
            cutoff_hz=config.cutoff_hz,
            sample_rate=config.sample_rate,
            num_taps=config.num_taps,
            window=config.window,
        )

        self._config = config
        self._lowpass_taps = lowpass_taps
        self._highpass_taps = highpass_taps

    @property
    def config(self) -> BandSplitConfig:
        """Return the configuration used by this processor.

        Physical Basis:
            Configuration parameters determine the filter cutoff and
            delay characteristics that preserve the low-band response.
        """
        return self._config

    @property
    def delay_samples(self) -> int:
        """Return the constant group delay in samples.

        Physical Basis:
            Complementary linear-phase FIR filters share the same constant
            group delay, enabling aligned recombination of low/high bands.
        """
        return (self._lowpass_taps.size - 1) // 2

    def split(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split a signal into low- and high-band components.

        Args:
            signal: Input signal (1D or 2D). Time axis must be last.

        Returns:
            Tuple of (low_band, high_band).

        Raises:
            ValueError: If the signal or filter taps are invalid.

        Physical Basis:
            Low/high-pass decomposition isolates the 0-20kHz band for
            bypass while keeping 20-44kHz available for suppression.
        """
        return band_split(signal, self._lowpass_taps, self._highpass_taps)

    def process(
        self,
        signal: np.ndarray,
        high_band_processor: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> BandSplitResult:
        """Split a signal, process the high band, and recombine.

        Args:
            signal: Input signal (1D or 2D). Time axis must be last.
            high_band_processor: Optional callable to process the high band.

        Returns:
            BandSplitResult containing low band, processed high band,
            recombined signal, and delay in samples.

        Raises:
            ValueError: If high-band processing returns an invalid shape.

        Physical Basis:
            The low band is passed through unchanged while the high band
            can be suppressed, ensuring audible-band fidelity by design.
        """
        low_band, high_band = self.split(signal)

        if high_band_processor is None:
            processed_high = high_band
        else:
            processed_high = np.asarray(high_band_processor(high_band))
            _validate_processed_high(processed_high, high_band)

        recombined = low_band + processed_high

        return BandSplitResult(
            low_band=low_band,
            high_band=processed_high,
            recombined=recombined,
            delay_samples=self.delay_samples,
        )

    def compensate_delay(self, signal: np.ndarray) -> np.ndarray:
        """Compensate for linear-phase filter delay by trimming samples.

        Args:
            signal: Signal to align (1D or 2D). Time axis must be last.

        Returns:
            Aligned signal with initial delay removed.

        Raises:
            ValueError: If delay is invalid for the given signal.

        Physical Basis:
            Linear-phase FIR filtering introduces a constant delay that
            can be compensated by trimming the leading samples.
        """
        return compensate_delay(signal, self.delay_samples)


def compensate_delay(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """Trim initial samples to compensate for linear-phase group delay.

    Args:
        signal: Input signal (1D or 2D). Time axis must be last.
        delay_samples: Delay in samples to remove.

    Returns:
        Delay-compensated signal.

    Raises:
        ValueError: If delay_samples is invalid for the signal length.

    Physical Basis:
        A constant group delay can be removed by trimming the leading
        samples, aligning filtered outputs with the original signal.
    """
    _validate_signal(signal)
    if delay_samples < 0:
        raise ValueError(f"delay_samples must be non-negative, got {delay_samples}.")
    if delay_samples == 0:
        return np.asarray(signal)
    if delay_samples >= signal.shape[-1]:
        raise ValueError(
            "delay_samples must be smaller than signal length "
            f"({signal.shape[-1]}), got {delay_samples}."
        )

    return np.asarray(signal)[..., delay_samples:]


def _validate_config(config: BandSplitConfig) -> None:
    if config.sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {config.sample_rate}.")
    if config.cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {config.cutoff_hz}.")
    nyquist = config.sample_rate / 2
    if config.cutoff_hz >= nyquist:
        raise ValueError(
            f"cutoff_hz must be less than Nyquist ({nyquist} Hz), "
            f"got {config.cutoff_hz}."
        )
    if config.num_taps <= 0:
        raise ValueError(f"num_taps must be positive, got {config.num_taps}.")
    if config.num_taps % 2 == 0:
        raise ValueError("num_taps must be odd for linear-phase FIR design.")


def _validate_processed_high(processed_high: np.ndarray, reference: np.ndarray) -> None:
    if processed_high.shape != reference.shape:
        raise ValueError(
            "high_band_processor must return the same shape as input. "
            f"Expected {reference.shape}, got {processed_high.shape}."
        )


def _validate_signal(signal: np.ndarray) -> None:
    if signal.ndim not in (1, 2):
        raise ValueError(f"signal must be 1D or 2D, got {signal.ndim}D.")
    if signal.size == 0:
        raise ValueError("signal cannot be empty.")
