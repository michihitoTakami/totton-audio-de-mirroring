"""DSP utilities for high-rate interpolation."""

from .multistage_upsampler import (
    UpsampleStageConfig,
    default_stage_configs,
    design_stage_taps,
    multistage_upsample,
    upsample_by_2,
)

__all__ = [
    "UpsampleStageConfig",
    "default_stage_configs",
    "design_stage_taps",
    "multistage_upsample",
    "upsample_by_2",
]
