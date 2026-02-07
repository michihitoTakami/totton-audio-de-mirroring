"""Stage 2 DSP utilities for high-rate interpolation analysis."""

from totton_audio_de_mirroring.stage2.cpp_backend import (
    CppStage2RuntimeConfig,
    CppStage2Upsampler,
)
from totton_audio_de_mirroring.stage2.overshoot import (
    OvershootEvaluation,
    OvershootMeasurement,
    cascade_upsample,
    evaluate_stage2_overshoot,
    load_stage_taps,
    upsample_2x_fir,
)

__all__ = [
    "CppStage2RuntimeConfig",
    "CppStage2Upsampler",
    "OvershootEvaluation",
    "OvershootMeasurement",
    "cascade_upsample",
    "evaluate_stage2_overshoot",
    "load_stage_taps",
    "upsample_2x_fir",
]
