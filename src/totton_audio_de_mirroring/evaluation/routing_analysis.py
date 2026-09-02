"""Content-aligned diagnostics for CAPB prototype routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RoutingSummary:
    """Summarize sharp/gentle routing on active safe and risky frames.

    Physical Basis:
        Sharp is preferred on active low-transient content for image
        suppression, while gentle is preferred around the strongest slope
        changes where a sharper FIR can expose pre/post ringing.
    """

    sharp_safe_mean: float
    sharp_risk_mean: float
    middle_risk_mean: float
    gentle_risk_mean: float
    protective_risk_mean: float
    routing_contrast: float
    weight_motion_rms: float
    active_frame_fraction: float
    risk_frame_fraction: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-compatible metric mapping."""
        return {key: float(value) for key, value in asdict(self).items()}


def transient_strength(signal: np.ndarray, frame_count: int) -> np.ndarray:
    """Measure normalized slope energy at controller-frame resolution.

    Args:
        signal: Non-empty mono source-rate waveform.
        frame_count: Number of controller weight frames.

    Returns:
        Non-negative short-time envelope-change strength clipped at one.

    Physical Basis:
        FIR pre/post ringing is perceptually exposed by an onset or offset,
        not by every cycle of a stationary high-frequency tone. Measuring
        changes in frame RMS separates those envelope events from sustain.
    """
    clean = _validate_signal(signal)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    level = np.sqrt(_frame_mean(np.square(clean), frame_count))
    strength = np.abs(np.diff(level, prepend=0.0))
    scale = float(np.quantile(strength, 0.99))
    if scale <= np.finfo(np.float64).tiny:
        return np.zeros(frame_count, dtype=np.float64)
    return np.asarray(np.clip(strength / scale, 0.0, 1.0), dtype=np.float64)


def summarize_routing(
    signal: np.ndarray,
    weights: np.ndarray,
    sharp_index: int,
    gentle_index: int,
    middle_index: int | None = None,
) -> RoutingSummary:
    """Compute policy diagnostics for one waveform and weight trajectory.

    Args:
        signal: Non-empty mono source-rate waveform.
        weights: Convex weights shaped (prototype, frame).
        sharp_index: Index of the sharp endpoint.
        gentle_index: Index of the gentle endpoint.
        middle_index: Optional transient-compensation prototype index.

    Returns:
        RoutingSummary over active safe/risk frame subsets.

    Physical Basis:
        Silence is excluded because its blend has no acoustic consequence.
        Risk is defined only from the top five percent of active transient
        strength and remains a diagnostic, not a label for held-out audio.
    """
    clean = _validate_signal(signal)
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("weights must have shape (prototype, frame).")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("weights must be finite and non-negative.")
    if not np.allclose(np.sum(values, axis=0), 1.0, atol=1.0e-5):
        raise ValueError("weights must sum to one per frame.")
    if not 0 <= sharp_index < values.shape[0]:
        raise ValueError("sharp_index is out of range.")
    if not 0 <= gentle_index < values.shape[0]:
        raise ValueError("gentle_index is out of range.")
    if middle_index is not None and not 0 <= middle_index < values.shape[0]:
        raise ValueError("middle_index is out of range.")
    level = _frame_mean(np.abs(clean), values.shape[1])
    active = level > max(float(np.max(level)) * 1.0e-3, 1.0e-8)
    strength = transient_strength(clean, values.shape[1])
    active_strength = strength[active]
    risk_threshold = (
        float(np.quantile(active_strength, 0.95)) if active_strength.size else np.inf
    )
    safe_threshold = (
        float(np.quantile(active_strength, 0.50)) if active_strength.size else -np.inf
    )
    risk = active & (strength >= risk_threshold)
    safe = active & (strength <= safe_threshold)
    sharp_safe = _masked_mean(values[sharp_index], safe)
    sharp_risk = _masked_mean(values[sharp_index], risk)
    middle_risk = (
        _masked_mean(values[middle_index], risk) if middle_index is not None else 0.0
    )
    gentle_risk = _masked_mean(values[gentle_index], risk)
    motion = np.diff(values, axis=1)
    return RoutingSummary(
        sharp_safe_mean=sharp_safe,
        sharp_risk_mean=sharp_risk,
        middle_risk_mean=middle_risk,
        gentle_risk_mean=gentle_risk,
        protective_risk_mean=middle_risk + gentle_risk,
        routing_contrast=sharp_safe - sharp_risk,
        weight_motion_rms=float(np.sqrt(np.mean(np.square(motion))))
        if motion.size
        else 0.0,
        active_frame_fraction=float(np.mean(active)),
        risk_frame_fraction=float(np.mean(risk)),
    )


def _frame_mean(values: np.ndarray, frame_count: int) -> np.ndarray:
    boundaries = np.linspace(0, values.size, frame_count + 1, dtype=np.int64)
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    counts = np.maximum(1, np.diff(boundaries))
    means = (cumulative[boundaries[1:]] - cumulative[boundaries[:-1]]) / counts
    return np.asarray(means, dtype=np.float64)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask])) if np.any(mask) else float("nan")


def _validate_signal(signal: np.ndarray) -> np.ndarray:
    clean = np.asarray(signal, dtype=np.float64)
    if clean.ndim != 1 or clean.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(clean)):
        raise ValueError("signal must contain only finite samples.")
    return clean
