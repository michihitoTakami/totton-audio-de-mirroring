"""Overshoot analysis helpers for Stage 2 interpolation filters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.signal as sp_signal


@dataclass(frozen=True)
class OvershootMeasurement:
    """Overshoot measurement for one test signal.

    Attributes:
        ratio: Overshoot ratio defined as ``(peak - reference) / |reference|``.
        peak: Peak output level.
        reference: Reference plateau level estimated from settled samples.

    Physical Basis:
        Overshoot is evaluated against a settled plateau, not against a
        transient point, to quantify ringing behavior of the interpolation
        filter while separating it from expected steady-state scaling.
    """

    ratio: float
    peak: float
    reference: float


@dataclass(frozen=True)
class OvershootEvaluation:
    """Combined overshoot evaluation for step and square-wave probes.

    Attributes:
        step: Overshoot measurement for step response.
        square: Overshoot measurement for square-wave response.
        output_sample_rate: Final sample rate after 2x cascade.

    Physical Basis:
        Step response highlights worst-case transient ringing and square-wave
        response checks behavior under repeated edges. Using both helps avoid
        overfitting design decisions to a single probe.
    """

    step: OvershootMeasurement
    square: OvershootMeasurement
    output_sample_rate: int


def load_stage_taps(config_dir: Path, num_stages: int = 3) -> tuple[np.ndarray, ...]:
    """Load per-stage FIR taps from ``stage{i}_taps.txt`` files.

    Args:
        config_dir: Directory containing stage tap text files.
        num_stages: Number of stages to load.

    Returns:
        Tuple of FIR tap arrays.

    Raises:
        FileNotFoundError: If a tap file does not exist.
        ValueError: If any loaded taps are empty or invalid.
        RuntimeError: If a file cannot be parsed.

    Physical Basis:
        Stage 2 is modeled as a cascade of 2x FIR interpolation stages.
        Evaluating the same tap set used in deployment is necessary for
        physically meaningful overshoot verification.
    """
    if num_stages <= 0:
        raise ValueError(f"num_stages must be positive, got {num_stages}")
    if not config_dir.exists():
        raise FileNotFoundError(f"config directory not found: {config_dir}")
    if not config_dir.is_dir():
        raise ValueError(f"config_dir must be a directory, got: {config_dir}")

    taps: list[np.ndarray] = []
    for stage_index in range(1, num_stages + 1):
        path = config_dir / f"stage{stage_index}_taps.txt"
        if not path.exists():
            raise FileNotFoundError(f"stage tap file not found: {path}")
        try:
            loaded = np.loadtxt(path, dtype=np.float64)
        except Exception as exc:
            raise RuntimeError(f"failed to load stage taps from {path}: {exc}") from exc

        loaded = np.atleast_1d(loaded).astype(np.float64, copy=False)
        _validate_1d_signal(loaded, name=f"stage{stage_index}_taps")
        taps.append(loaded)
    return tuple(taps)


def upsample_2x_fir(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply one 2x zero-stuffing + FIR interpolation stage.

    Args:
        signal: Input time-domain signal.
        taps: FIR taps for this stage.

    Returns:
        2x upsampled signal.

    Physical Basis:
        Zero stuffing halves the baseband amplitude. The unity-DC FIR output
        is therefore multiplied by the interpolation ratio so a settled
        waveform retains its input level at every stage.
    """
    _validate_1d_signal(signal, name="signal")
    _validate_1d_signal(taps, name="taps")

    signal_64 = np.asarray(signal, dtype=np.float64)
    taps_64 = np.asarray(taps, dtype=np.float64)

    zero_stuffed = np.zeros(signal_64.shape[0] * 2, dtype=np.float64)
    zero_stuffed[::2] = signal_64
    filtered = sp_signal.lfilter(taps_64, [1.0], zero_stuffed)
    return np.asarray(filtered * 2.0, dtype=np.float64)


def cascade_upsample(
    signal: np.ndarray, stage_taps: Sequence[np.ndarray]
) -> np.ndarray:
    """Apply all 2x interpolation stages in sequence.

    Args:
        signal: Input signal at source sample rate.
        stage_taps: Sequence of per-stage FIR taps.

    Returns:
        Upsampled signal after all stages.

    Physical Basis:
        Stage 2 uses cascaded 2x stages (2x×2x×2x) to balance computational
        cost and transition-band control versus a single high-ratio filter.
    """
    _validate_1d_signal(signal, name="signal")
    if len(stage_taps) == 0:
        raise ValueError("stage_taps must not be empty")

    current = np.asarray(signal, dtype=np.float64)
    for index, taps in enumerate(stage_taps, start=1):
        _validate_1d_signal(taps, name=f"stage_taps[{index}]")
        current = upsample_2x_fir(current, taps)
    return current


def evaluate_stage2_overshoot(
    stage_taps: Sequence[np.ndarray],
    source_sample_rate: int = 88_200,
    step_length: int = 4_096,
    square_frequency_hz: float = 1_000.0,
    square_duration_sec: float = 0.2,
    settle_fraction: float = 0.75,
    reference_quantile: float = 0.95,
) -> OvershootEvaluation:
    """Measure step and square-wave overshoot for Stage 2 taps.

    Args:
        stage_taps: Sequence of FIR taps, one array per 2x stage.
        source_sample_rate: Input sample rate before Stage 2.
        step_length: Number of input samples for step probe.
        square_frequency_hz: Frequency of square-wave probe.
        square_duration_sec: Duration of square-wave probe.
        settle_fraction: Fraction of response tail used as settled region.
        reference_quantile: Quantile used for plateau reference.

    Returns:
        Combined overshoot evaluation.

    Physical Basis:
        Step response captures worst-case edge ringing; square-wave response
        validates repeated transient behavior. Quantile-based plateau
        estimation reduces sensitivity to polarity alternation in polyphase
        interpolation outputs.
    """
    if source_sample_rate <= 0:
        raise ValueError(
            f"source_sample_rate must be positive, got {source_sample_rate}"
        )
    if step_length <= 1:
        raise ValueError(f"step_length must be > 1, got {step_length}")
    if square_frequency_hz <= 0.0:
        raise ValueError(
            f"square_frequency_hz must be positive, got {square_frequency_hz}"
        )
    if square_duration_sec <= 0.0:
        raise ValueError(
            f"square_duration_sec must be positive, got {square_duration_sec}"
        )
    if not 0.0 < settle_fraction < 1.0:
        raise ValueError(f"settle_fraction must be in (0, 1), got {settle_fraction}")
    if not 0.5 <= reference_quantile < 1.0:
        raise ValueError(
            f"reference_quantile must be in [0.5, 1), got {reference_quantile}"
        )

    step_input = np.ones(step_length, dtype=np.float64)
    step_output = cascade_upsample(step_input, stage_taps)
    step = _measure_overshoot(step_output, settle_fraction, reference_quantile)

    square_input = _generate_square_wave(
        sample_rate=source_sample_rate,
        frequency_hz=square_frequency_hz,
        duration_sec=square_duration_sec,
    )
    square_output = cascade_upsample(square_input, stage_taps)
    square = _measure_overshoot(
        square_output,
        settle_fraction,
        reference_quantile,
        positive_only=True,
    )

    output_sample_rate = source_sample_rate * (2 ** len(stage_taps))
    return OvershootEvaluation(
        step=step, square=square, output_sample_rate=output_sample_rate
    )


def _generate_square_wave(
    sample_rate: int, frequency_hz: float, duration_sec: float
) -> np.ndarray:
    """Generate a unit-amplitude square wave for overshoot probing.

    Physical Basis:
        Repeated discontinuities in square waves expose ringing accumulation
        and damping behavior across consecutive edges.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if frequency_hz <= 0.0:
        raise ValueError(f"frequency_hz must be positive, got {frequency_hz}")
    if duration_sec <= 0.0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}")

    num_samples = int(round(sample_rate * duration_sec))
    if num_samples < 4:
        raise ValueError(
            "duration_sec is too short for square-wave measurement; "
            f"requires >=4 samples, got {num_samples}"
        )

    time_axis = np.arange(num_samples, dtype=np.float64) / float(sample_rate)
    square = sp_signal.square(2.0 * np.pi * frequency_hz * time_axis)
    return np.asarray(square, dtype=np.float64)


def _measure_overshoot(
    response: np.ndarray,
    settle_fraction: float,
    reference_quantile: float,
    positive_only: bool = False,
) -> OvershootMeasurement:
    """Estimate overshoot ratio from a response signal tail region.

    Physical Basis:
        Settled-tail quantiles provide a robust plateau estimate when output
        samples contain interleaved polyphase values.
    """
    _validate_1d_signal(response, name="response")
    if not 0.0 < settle_fraction < 1.0:
        raise ValueError(f"settle_fraction must be in (0, 1), got {settle_fraction}")
    if not 0.5 <= reference_quantile < 1.0:
        raise ValueError(
            f"reference_quantile must be in [0.5, 1), got {reference_quantile}"
        )

    settle_start = int(response.shape[0] * settle_fraction)
    if settle_start >= response.shape[0]:
        raise ValueError(
            "settle_fraction produced an empty settled region: "
            f"settle_start={settle_start}, length={response.shape[0]}"
        )

    settled = response[settle_start:]
    if positive_only:
        settled = settled[settled > 0.0]
        if settled.size == 0:
            raise ValueError(
                "no positive settled samples available for overshoot measurement"
            )
    reference = float(np.quantile(settled, reference_quantile))
    if np.isclose(reference, 0.0):
        raise ValueError("reference level is near zero; overshoot ratio is undefined")

    peak = float(np.max(response))
    ratio = max(0.0, (peak - reference) / abs(reference))
    return OvershootMeasurement(ratio=ratio, peak=peak, reference=reference)


def _validate_1d_signal(signal: np.ndarray, name: str) -> None:
    """Validate a 1D finite signal array.

    Physical Basis:
        1D finite arrays are required to define stable time-domain response
        metrics for deterministic FIR interpolation analysis.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"{name} must be numpy.ndarray, got {type(signal)}")
    if signal.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {signal.shape}")
    if signal.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(signal).all():
        raise ValueError(f"{name} must contain only finite values")
