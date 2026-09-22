"""Transient dwell projection for CAPB blend weights.

The controller's convex weights are post-processed by a deterministic,
parameter-free projection that confines the gentle prototype to a short
window around detected waveform events, routes the surrounding ring to the
middle prototype, and releases everything else back to sharp. The rule
follows directly from the prototype impulse-response supports rather than
from learned behaviour, so it cannot be un-learned by the controller.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

DEFAULT_CREST_THRESHOLD = 4.0
DEFAULT_LEVEL_CHANGE_DB = 6.0
DEFAULT_GENTLE_ZONE_FRAMES = 1
DEFAULT_SHARP_ZONE_FRAMES = 4
DEFAULT_RAMP_FRAMES = 0
DEFAULT_FLOOR_DB = 80.0
_PEAK_ACTIVITY_FLOOR = 1.0e-4
_ENERGY_EPS = 1.0e-20
_RMS_EPS = 1.0e-12


@dataclass(frozen=True)
class TransientDwellConfig:
    """Constants of the transient dwell projection saved with a checkpoint.

    Args:
        enabled: Apply the projection after the controller softmax.
        crest_threshold: Frame derivative peak-to-RMS ratio above which a
            frame is treated as containing an edge (square edges reach
            5.7 or more, Gaussian noise stays near 2.5 to 3.5).
        level_change_db: Absolute short-time energy change across the two
            neighbouring frames that marks an envelope onset or offset.
        gentle_zone_frames: Frames on either side of an event where the
            controller mix is left untouched.
        sharp_zone_frames: Frames on either side of an event that exclude
            the sharp prototype; the ring between the gentle zone and this
            radius is routed to the middle prototype.
        ramp_frames: Half-width of an extra frame-rate linear ramp at zone
            edges. Zero relies on the sample-rate interpolation of the frame
            weights, which already spreads every zone edge over one frame
            (about 1.5 ms); one frame halves the recovered high band on
            percussive material for a 10 dB lower modulation floor on
            band-limited recordings (-105 versus -96 dBr).
        release_to_sharp: Outside the sharp zone move all mass to sharp.
        floor_db: Level changes whose louder neighbour lies more than this
            far below the loudest frame are ignored as noise-floor wander.

    Physical Basis:
        The sharp prototype rings for about 5.8 ms on either side of a
        discontinuity, the middle prototype for 0.54 ms and gentle for
        0.46 ms. Gentle therefore only has to carry the edge itself, the
        middle prototype can carry the sharp-exclusion ring without losing
        20 kHz passband, and beyond the sharp support nothing needs
        protection at all. Default frame radii of 1 and 4 map to 1.5 and
        5.8 ms at 44.1 kHz input (1.3 and 5.3 ms at 48 kHz).
    """

    enabled: bool = False
    crest_threshold: float = DEFAULT_CREST_THRESHOLD
    level_change_db: float = DEFAULT_LEVEL_CHANGE_DB
    gentle_zone_frames: int = DEFAULT_GENTLE_ZONE_FRAMES
    sharp_zone_frames: int = DEFAULT_SHARP_ZONE_FRAMES
    ramp_frames: int = DEFAULT_RAMP_FRAMES
    release_to_sharp: bool = True
    floor_db: float = DEFAULT_FLOOR_DB

    def __post_init__(self) -> None:
        if self.crest_threshold <= 1.0:
            raise ValueError("crest_threshold must exceed 1 (the sinusoid floor).")
        if self.level_change_db <= 0.0:
            raise ValueError("level_change_db must be positive.")
        if self.gentle_zone_frames < 0:
            raise ValueError("gentle_zone_frames must be non-negative.")
        if self.sharp_zone_frames <= self.gentle_zone_frames:
            raise ValueError("sharp_zone_frames must exceed gentle_zone_frames.")
        if self.ramp_frames < 0:
            raise ValueError("ramp_frames must be non-negative.")
        if self.floor_db <= 0.0:
            raise ValueError("floor_db must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/torch.save friendly mapping."""
        return {
            "enabled": bool(self.enabled),
            "crest_threshold": float(self.crest_threshold),
            "level_change_db": float(self.level_change_db),
            "gentle_zone_frames": int(self.gentle_zone_frames),
            "sharp_zone_frames": int(self.sharp_zone_frames),
            "ramp_frames": int(self.ramp_frames),
            "release_to_sharp": bool(self.release_to_sharp),
            "floor_db": float(self.floor_db),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> TransientDwellConfig:
        """Build a config from checkpoint or YAML data (disabled when absent)."""
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("transient_dwell must be a mapping.")
        defaults = cls()
        return cls(
            enabled=bool(raw.get("enabled", True)),
            crest_threshold=float(raw.get("crest_threshold", defaults.crest_threshold)),
            level_change_db=float(raw.get("level_change_db", defaults.level_change_db)),
            gentle_zone_frames=int(
                raw.get("gentle_zone_frames", defaults.gentle_zone_frames)
            ),
            sharp_zone_frames=int(
                raw.get("sharp_zone_frames", defaults.sharp_zone_frames)
            ),
            ramp_frames=int(raw.get("ramp_frames", defaults.ramp_frames)),
            release_to_sharp=bool(raw.get("release_to_sharp", True)),
            floor_db=float(raw.get("floor_db", defaults.floor_db)),
        )


def detect_transient_frames(
    normalized_source: torch.Tensor,
    frames: int,
    control_stride: int,
    config: TransientDwellConfig,
) -> torch.Tensor:
    """Return a (batch, 1, frames) event indicator in {0, 1}.

    Args:
        normalized_source: Peak-normalized waveform (batch, time).
        frames: Controller frame count the indicator must match.
        control_stride: Samples per controller frame.
        config: Detector thresholds.

    Returns:
        Float tensor with one on frames that contain a discontinuity or a
        short-time level change and zero elsewhere.

    Raises:
        ValueError: If the input is not a non-empty batched waveform.

    Physical Basis:
        A single-frame derivative crest isolates sample-scale
        discontinuities (steps, clicks, square edges, including every frame
        of a fast periodic square) from noise-like content. A symmetric
        energy change across one frame on either side catches envelope
        onsets and offsets that lack a sharp edge; only the loud side of the
        change is marked so an isolated spike stays one frame wide. Both are
        measured with centred pooling so the indicator has no causal bias.
    """
    if normalized_source.dim() != 2 or normalized_source.shape[-1] == 0:
        raise ValueError("normalized_source must be a non-empty (batch, time) tensor.")
    if frames <= 0:
        raise ValueError("frames must be positive.")
    if control_stride <= 0:
        raise ValueError("control_stride must be positive.")
    waveform = normalized_source.unsqueeze(1)
    padded = torch.cat([waveform[..., :1], waveform], dim=-1)
    derivative = padded[..., 1:] - padded[..., :-1]
    kernel = control_stride + 1
    padding = control_stride // 2
    peak = F.max_pool1d(
        torch.abs(derivative),
        kernel_size=kernel,
        stride=control_stride,
        padding=padding,
    )
    energy = F.avg_pool1d(
        derivative.square(),
        kernel_size=kernel,
        stride=control_stride,
        padding=padding,
        count_include_pad=False,
    )
    if peak.shape[-1] != frames:
        peak = F.adaptive_max_pool1d(peak, frames)
        energy = F.adaptive_avg_pool1d(energy, frames)
    crest = peak / (torch.sqrt(energy.clamp_min(_RMS_EPS)))
    active = (peak > _PEAK_ACTIVITY_FLOOR).to(peak.dtype)
    crest_event = (crest > config.crest_threshold).to(peak.dtype) * active

    level_db = 10.0 * torch.log10(energy + _ENERGY_EPS)
    previous = F.pad(level_db[..., :-1], (1, 0), mode="replicate")
    following = F.pad(level_db[..., 1:], (0, 1), mode="replicate")
    change = torch.abs(following - previous)
    louder = torch.maximum(following, previous)
    # Only the loud side of a level change is the event: this keeps the
    # quiet neighbours of an isolated spike (already a crest event) from
    # widening the indicator by one frame on each side.
    loud_side = level_db >= louder - config.level_change_db
    floor = level_db.amax(dim=-1, keepdim=True) - config.floor_db
    level_event = (
        (change > config.level_change_db) & loud_side & (level_db > floor)
    ).to(peak.dtype)
    return torch.maximum(crest_event, level_event)


def _zone_keep(event: torch.Tensor, inner: int, ramp: int) -> torch.Tensor:
    """Return a mask that is one within ``inner`` frames of an event and
    decays linearly to zero over ``2 * ramp`` further frames."""
    dilated = event
    radius = inner + ramp
    if radius > 0:
        dilated = F.max_pool1d(
            event, kernel_size=2 * radius + 1, stride=1, padding=radius
        )
    if ramp == 0:
        return dilated
    return F.avg_pool1d(
        dilated,
        kernel_size=2 * ramp + 1,
        stride=1,
        padding=ramp,
        count_include_pad=False,
    )


def apply_transient_dwell(
    weights: torch.Tensor,
    normalized_source: torch.Tensor,
    prototype_names: tuple[str, ...],
    control_stride: int,
    config: TransientDwellConfig,
) -> torch.Tensor:
    """Project convex blend weights onto the transient dwell policy.

    Args:
        weights: Convex prototype weights (batch, prototypes, frames).
        normalized_source: Peak-normalized waveform (batch, time).
        prototype_names: Prototype order of the weight axis.
        control_stride: Samples per controller frame.
        config: Projection constants; returned unchanged when disabled.

    Returns:
        New convex weights of the same shape. The input is not modified.

    Raises:
        ValueError: If shapes or prototype names are invalid.

    Physical Basis:
        Blend response is linear in the weights, so passband droop and
        image leakage scale with the gentle weight while long ringing scales
        with the sharp weight. Moving gentle mass to the flat middle
        prototype outside the edge core, and all non-sharp mass back to sharp
        beyond sharp's own support, removes both costs where they buy no
        protection. Mass only moves between prototypes, so the result stays
        on the simplex and the controller retains its choice inside the core.
    """
    if not config.enabled:
        return weights
    if weights.dim() != 3 or weights.shape[1] != len(prototype_names):
        raise ValueError("weights must be (batch, prototypes, frames).")
    if "sharp" not in prototype_names or "gentle" not in prototype_names:
        raise ValueError("Transient dwell requires sharp and gentle prototypes.")
    frames = weights.shape[-1]
    event = detect_transient_frames(
        normalized_source, frames, control_stride, config
    ).to(weights.dtype)
    keep_gentle = _zone_keep(event, config.gentle_zone_frames, config.ramp_frames)
    protect = _zone_keep(event, config.sharp_zone_frames, config.ramp_frames)

    sharp_index = prototype_names.index("sharp")
    gentle_index = prototype_names.index("gentle")
    channels = [
        weights[:, index : index + 1, :] for index in range(len(prototype_names))
    ]
    if "mid" in prototype_names:
        mid_index = prototype_names.index("mid")
        moved = channels[gentle_index] * (1.0 - keep_gentle)
        channels[gentle_index] = channels[gentle_index] - moved
        channels[mid_index] = channels[mid_index] + moved
    if config.release_to_sharp:
        released = torch.zeros_like(channels[sharp_index])
        for index in range(len(prototype_names)):
            if index == sharp_index:
                continue
            released = released + channels[index] * (1.0 - protect)
            channels[index] = channels[index] * protect
        channels[sharp_index] = channels[sharp_index] + released
    return torch.cat(channels, dim=1)
