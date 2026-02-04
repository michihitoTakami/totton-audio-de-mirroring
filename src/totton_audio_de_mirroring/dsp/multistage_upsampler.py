"""Multi-stage 2x upsampling utilities for high-rate interpolation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

DEFAULT_STAGE_CUTOFF_HZ = 43_200.0
DEFAULT_STAGE_TAPS = 255
DEFAULT_STAGE_WINDOW = "hamming"
DEFAULT_STAGE_PHASE = "minimum"
DEFAULT_STAGE_COUNT = 3


@dataclass(frozen=True)
class UpsampleStageConfig:
    """Configuration for a single 2x upsampling stage.

    Args:
        cutoff_hz: Low-pass cutoff frequency in Hz.
        num_taps: Number of FIR taps.
        window: Window function for FIR design.
        phase: "minimum" or "linear" phase response.

    Physical Basis:
        Each stage applies a gentle low-pass filter to suppress imaging
        artifacts from zero-stuffing while maintaining time-response
        characteristics aligned with the minimum-phase preference.
    """

    cutoff_hz: float = DEFAULT_STAGE_CUTOFF_HZ
    num_taps: int = DEFAULT_STAGE_TAPS
    window: str | tuple[str, float] = DEFAULT_STAGE_WINDOW
    phase: str = DEFAULT_STAGE_PHASE


def default_stage_configs(
    num_stages: int = DEFAULT_STAGE_COUNT,
    cutoff_hz: float = DEFAULT_STAGE_CUTOFF_HZ,
    num_taps: int = DEFAULT_STAGE_TAPS,
    window: str | tuple[str, float] = DEFAULT_STAGE_WINDOW,
    phase: str = DEFAULT_STAGE_PHASE,
) -> tuple[UpsampleStageConfig, ...]:
    """Create a tuple of identical stage configurations.

    Args:
        num_stages: Number of 2x stages to chain.
        cutoff_hz: Low-pass cutoff frequency in Hz for each stage.
        num_taps: Number of FIR taps per stage.
        window: Window function for FIR design.
        phase: "minimum" or "linear" phase response.

    Returns:
        Tuple of stage configurations.

    Raises:
        ValueError: If num_stages is invalid.

    Physical Basis:
        Using identical per-stage filters maintains a consistent spectral
        envelope across the 2x cascade while controlling imaging artifacts.
    """
    _validate_positive_int(num_stages, "num_stages")
    return tuple(
        UpsampleStageConfig(
            cutoff_hz=cutoff_hz,
            num_taps=num_taps,
            window=window,
            phase=phase,
        )
        for _ in range(num_stages)
    )


def design_stage_taps(
    config: UpsampleStageConfig,
    target_sample_rate: int,
) -> np.ndarray:
    """Design FIR taps for a single upsampling stage.

    Args:
        config: Stage configuration.
        target_sample_rate: Target sample rate in Hz after upsampling.

    Returns:
        FIR taps for the stage.

    Raises:
        ValueError: If configuration or sample rate is invalid.

    Physical Basis:
        A windowed FIR low-pass filter limits imaging artifacts after
        zero-stuffing. Minimum-phase conversion keeps energy earlier in
        time to align with the time-response preservation goal.
    """
    _validate_sample_rate(target_sample_rate)
    _validate_stage_config(config, target_sample_rate)

    taps = sp_signal.firwin(
        config.num_taps,
        config.cutoff_hz,
        fs=target_sample_rate,
        window=config.window,
    )

    if config.phase == "minimum":
        taps = sp_signal.minimum_phase(taps, method="homomorphic")
    elif config.phase != "linear":
        raise ValueError(f"Unsupported phase: {config.phase}.")

    return np.asarray(taps, dtype=np.float64)


def upsample_by_2(
    signal: np.ndarray,
    sample_rate: int,
    config: UpsampleStageConfig | None = None,
) -> tuple[np.ndarray, int]:
    """Upsample a signal by 2x using a single FIR stage.

    Args:
        signal: Input signal (1D or 2D). Time axis must be last.
        sample_rate: Input sample rate in Hz.
        config: Optional stage configuration.

    Returns:
        Tuple of (upsampled_signal, new_sample_rate).

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        2x upsampling with low-pass filtering suppresses images created
        by zero insertion while maintaining a gentle, minimum-phase
        response to preserve time-domain transients.
    """
    _validate_signal(signal)
    _validate_sample_rate(sample_rate)

    stage_config = config or UpsampleStageConfig()
    target_sample_rate = sample_rate * 2
    taps = design_stage_taps(stage_config, target_sample_rate)

    filtered = sp_signal.upfirdn(taps, signal, up=2, down=1, axis=-1)
    expected_len = signal.shape[-1] * 2
    trimmed = np.asarray(filtered[..., :expected_len], dtype=np.float64)

    return trimmed, target_sample_rate


def multistage_upsample(
    signal: np.ndarray,
    input_sample_rate: int,
    stages: Sequence[UpsampleStageConfig] | None = None,
) -> tuple[np.ndarray, int]:
    """Apply a multi-stage 2x cascade upsampling pipeline.

    Args:
        signal: Input signal (1D or 2D). Time axis must be last.
        input_sample_rate: Input sample rate in Hz.
        stages: Sequence of stage configurations. Defaults to 3 stages.

    Returns:
        Tuple of (upsampled_signal, output_sample_rate).

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Cascading multiple 2x stages reduces per-stage filter complexity
        while achieving high-rate interpolation with controlled group
        delay and limited imaging artifacts.
    """
    _validate_signal(signal)
    _validate_sample_rate(input_sample_rate)

    stage_configs = default_stage_configs() if stages is None else tuple(stages)
    _validate_stage_sequence(stage_configs)

    current_signal = np.asarray(signal, dtype=np.float64)
    current_rate = input_sample_rate

    for stage_config in stage_configs:
        current_signal, current_rate = upsample_by_2(
            current_signal,
            current_rate,
            config=stage_config,
        )

    return current_signal, current_rate


def _validate_stage_sequence(stages: Sequence[UpsampleStageConfig]) -> None:
    if len(stages) == 0:
        raise ValueError("stages must contain at least one stage.")


def _validate_stage_config(
    config: UpsampleStageConfig,
    target_sample_rate: int,
) -> None:
    _validate_cutoff(config.cutoff_hz, target_sample_rate)
    _validate_positive_int(config.num_taps, "num_taps")
    if config.phase not in {"minimum", "linear"}:
        raise ValueError(f"Unsupported phase: {config.phase}.")


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
