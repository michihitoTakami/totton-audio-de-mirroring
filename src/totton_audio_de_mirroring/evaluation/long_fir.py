"""Structural and long-tail evaluation for experimental CAPB FIR banks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal as sp_signal

from totton_audio_de_mirroring.models.proto_bank import PrototypeBank

DEFAULT_PHASE_LOW_HZ = 100.0
DEFAULT_PHASE_HIGH_HZ = 20_000.0
DEFAULT_PHASE_POINTS = 32_768
DEFAULT_PHASE_TOLERANCE_DEG = 1.0e-6
DEFAULT_DELAY_TOLERANCE_SAMPLES = 1.0e-9
DEFAULT_SYMMETRY_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class PhaseAlignmentMetrics:
    """Measured common-center and passband-phase properties."""

    kernel_length: int
    expected_group_delay_samples: int
    group_delay_samples: tuple[float, ...]
    peak_indices: tuple[int, ...]
    symmetry_relative_error: float
    max_phase_spread_deg: float
    max_group_delay_spread_samples: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class LongEchoMetrics:
    """Near- and far-tail mean-square energy around one event."""

    pre_0p5_4ms: float
    pre_4_12ms: float
    post_0p5_4ms: float
    post_4_12ms: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def evaluate_phase_alignment(
    bank: PrototypeBank,
    *,
    low_hz: float = DEFAULT_PHASE_LOW_HZ,
    high_hz: float = DEFAULT_PHASE_HIGH_HZ,
    num_points: int = DEFAULT_PHASE_POINTS,
) -> PhaseAlignmentMetrics:
    """Measure shared phase, delay, symmetry, and impulse centers.

    Args:
        bank: Centered prototype bank to inspect.
        low_hz: Lowest passband frequency included in the phase fit.
        high_hz: Highest passband frequency included in the phase fit.
        num_points: Dense frequency-grid size.

    Returns:
        PhaseAlignmentMetrics for every prototype.

    Raises:
        ValueError: If the bank or analysis band is invalid.

    Physical Basis:
        Equal odd length and symmetric centering impose the same linear phase
        on sharp, mid, and gentle. Measuring the fitted delay and residual
        pairwise phase catches one-sample or asymmetric-padding mistakes that
        would otherwise cause cancellation during convex blending.
    """
    _validate_bank(bank)
    nyquist = bank.sample_rate / 2.0
    if not 0.0 < low_hz < high_hz < nyquist:
        raise ValueError("Require 0 < low_hz < high_hz < Nyquist.")
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    frequencies = np.linspace(low_hz, high_hz, num_points)
    relative_phases: list[np.ndarray] = []
    delays: list[float] = []
    for kernel in bank.kernels:
        _, response = sp_signal.freqz(
            np.asarray(kernel, dtype=np.float64),
            worN=frequencies,
            fs=bank.sample_rate,
        )
        phase = np.unwrap(np.angle(response))
        slope, _ = np.polyfit(frequencies, phase, 1)
        delays.append(float(-slope * bank.sample_rate / (2.0 * np.pi)))
        relative_phases.append(phase - phase[0])

    phase_matrix = np.stack(relative_phases)
    symmetry = float(
        np.max(np.abs(bank.kernels - bank.kernels[:, ::-1]))
        / np.max(np.abs(bank.kernels))
    )
    return PhaseAlignmentMetrics(
        kernel_length=int(bank.kernels.shape[1]),
        expected_group_delay_samples=bank.group_delay_samples,
        group_delay_samples=tuple(delays),
        peak_indices=tuple(int(np.argmax(np.abs(row))) for row in bank.kernels),
        symmetry_relative_error=symmetry,
        max_phase_spread_deg=float(
            np.max(np.ptp(phase_matrix, axis=0)) * 180.0 / np.pi
        ),
        max_group_delay_spread_samples=float(np.ptp(delays)),
    )


def validate_phase_alignment(
    metrics: PhaseAlignmentMetrics,
    *,
    phase_tolerance_deg: float = DEFAULT_PHASE_TOLERANCE_DEG,
    delay_tolerance_samples: float = DEFAULT_DELAY_TOLERANCE_SAMPLES,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
) -> None:
    """Raise when a prototype bank does not share one phase center.

    Args:
        metrics: Measurements returned by evaluate_phase_alignment.
        phase_tolerance_deg: Maximum pairwise passband phase spread.
        delay_tolerance_samples: Maximum fitted group-delay spread.
        symmetry_tolerance: Maximum relative coefficient asymmetry.

    Raises:
        ValueError: If symmetry, phase, delay, or center alignment fails.

    Physical Basis:
        A convex magnitude blend is phase-safe only when all endpoints share
        the same linear-phase center. This is a hard structural gate.
    """
    if metrics.symmetry_relative_error > symmetry_tolerance:
        raise ValueError("Prototype symmetry exceeds tolerance.")
    if metrics.max_phase_spread_deg > phase_tolerance_deg:
        raise ValueError("Prototype passband phase spread exceeds tolerance.")
    if metrics.max_group_delay_spread_samples > delay_tolerance_samples:
        raise ValueError("Prototype group-delay spread exceeds tolerance.")
    if any(
        index != metrics.expected_group_delay_samples for index in metrics.peak_indices
    ):
        raise ValueError("Prototype impulse peaks do not share the common center.")


def evaluate_long_echo(
    signal: np.ndarray,
    *,
    center_index: int,
    sample_rate: int,
) -> LongEchoMetrics:
    """Measure pre/post-event mean-square energy through 12 ms.

    Args:
        signal: Candidate output waveform, not modified.
        center_index: Expected event center on the output timeline.
        sample_rate: Output sample rate in Hz.

    Returns:
        Near (0.5--4 ms) and far (4--12 ms) echo measurements.

    Raises:
        ValueError: If the requested windows do not fit the waveform.

    Physical Basis:
        The frozen G2b window observes 0.5--4 ms. A 2047-tap filter can extend
        to roughly 12 ms on either side, so the additional far window prevents
        long low-level ringing from escaping the existing gate aperture.
    """
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("signal must be a non-empty 1D array.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    offsets = {
        "near_start": _milliseconds_to_samples(0.5, sample_rate),
        "near_end": _milliseconds_to_samples(4.0, sample_rate),
        "far_end": _milliseconds_to_samples(12.0, sample_rate),
    }
    if center_index - offsets["far_end"] < 0:
        raise ValueError("Insufficient samples before center_index.")
    if center_index + offsets["far_end"] >= samples.size:
        raise ValueError("Insufficient samples after center_index.")
    return LongEchoMetrics(
        pre_0p5_4ms=_mean_square(
            samples[
                center_index - offsets["near_end"] : center_index
                - offsets["near_start"]
            ]
        ),
        pre_4_12ms=_mean_square(
            samples[
                center_index - offsets["far_end"] : center_index - offsets["near_end"]
            ]
        ),
        post_0p5_4ms=_mean_square(
            samples[
                center_index + offsets["near_start"] : center_index
                + offsets["near_end"]
            ]
        ),
        post_4_12ms=_mean_square(
            samples[
                center_index + offsets["near_end"] : center_index + offsets["far_end"]
            ]
        ),
    )


def _validate_bank(bank: PrototypeBank) -> None:
    """Validate the minimum structural contract needed by phase analysis."""
    kernels = np.asarray(bank.kernels)
    if kernels.ndim != 2 or kernels.shape[0] == 0:
        raise ValueError("bank.kernels must be a non-empty 2D array.")
    if kernels.shape[1] % 2 == 0:
        raise ValueError("Prototype kernel length must be odd.")
    expected_delay = (kernels.shape[1] - 1) // 2
    if bank.group_delay_samples != expected_delay:
        raise ValueError("bank.group_delay_samples does not match kernel length.")


def _milliseconds_to_samples(milliseconds: float, sample_rate: int) -> int:
    """Convert a physical duration to its nearest sample count."""
    return int(round(milliseconds * sample_rate / 1_000.0))


def _mean_square(samples: np.ndarray) -> float:
    """Return mean-square energy for a non-empty analysis window."""
    if samples.size == 0:
        raise ValueError("Echo window must not be empty.")
    return float(np.mean(np.square(samples)))
