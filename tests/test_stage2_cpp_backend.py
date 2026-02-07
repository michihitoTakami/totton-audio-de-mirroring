"""Tests for C++ Stage 2 backend bridge."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from totton_audio_de_mirroring.stage2.cpp_backend import (
    CppStage2RuntimeConfig,
    CppStage2Upsampler,
)
from totton_audio_de_mirroring.stage2.overshoot import cascade_upsample, load_stage_taps


def test_cpp_stage2_backend_matches_python_reference(tmp_path: Path) -> None:
    """C++ core API output should match Python cascade for deterministic taps."""
    if shutil.which("cmake") is None:
        pytest.skip("cmake not available")

    config_dir = tmp_path / "taps"
    config_dir.mkdir()
    for i in range(1, 4):
        (config_dir / f"stage{i}_taps.txt").write_text("1.0\n", encoding="utf-8")

    signal = np.array([0.5, -0.25, 1.0, -0.5], dtype=np.float64)
    stage_taps = load_stage_taps(config_dir, num_stages=3)
    expected = cascade_upsample(signal, stage_taps)

    runtime = CppStage2RuntimeConfig(
        config_dir=config_dir,
        num_stages=3,
        cpp_project_dir=Path("cpp"),
        cpp_build_dir=tmp_path / "build",
    )
    with CppStage2Upsampler(runtime) as upsampler:
        output = upsampler.process(signal)

    np.testing.assert_allclose(output, expected, rtol=0.0, atol=1e-12)


def test_cpp_stage2_backend_rejects_missing_config_dir(tmp_path: Path) -> None:
    """Bridge should fail clearly when Stage 2 config directory is missing."""
    if shutil.which("cmake") is None:
        pytest.skip("cmake not available")

    runtime = CppStage2RuntimeConfig(
        config_dir=tmp_path / "missing",
        num_stages=3,
        cpp_project_dir=Path("cpp"),
        cpp_build_dir=tmp_path / "build_missing",
    )

    with pytest.raises(RuntimeError, match="config directory not found"):
        _ = CppStage2Upsampler(runtime)
