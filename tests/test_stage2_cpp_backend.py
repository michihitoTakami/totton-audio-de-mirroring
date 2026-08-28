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
    """C++ core API should match Python for injected and production taps."""
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

    default_signal = np.linspace(-1.0, 1.0, 128, dtype=np.float64)
    default_taps = load_stage_taps(Path("cpp/configs"), num_stages=3)
    default_expected = cascade_upsample(default_signal, default_taps)
    default_runtime = CppStage2RuntimeConfig(
        config_dir=Path("cpp/configs"),
        num_stages=3,
        cpp_project_dir=Path("cpp"),
        cpp_build_dir=tmp_path / "build",
    )
    with CppStage2Upsampler(default_runtime) as upsampler:
        default_output = upsampler.process(default_signal)

    np.testing.assert_allclose(default_output, default_expected, rtol=0.0, atol=1e-12)


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


def test_cpp_stage2_backend_preserves_settled_dc_level(tmp_path: Path) -> None:
    """Three unity-gain stages should not attenuate a settled waveform.

    Physical Basis:
        Each 2x zero-stuff stage requires interpolation-ratio gain
        compensation; otherwise three stages reduce level by 18.06 dB.
    """
    if shutil.which("cmake") is None:
        pytest.skip("cmake not available")

    config_dir = tmp_path / "gain_taps"
    config_dir.mkdir()
    for index in range(1, 4):
        (config_dir / f"stage{index}_taps.txt").write_text(
            "0.5\n0.5\n", encoding="utf-8"
        )
    runtime = CppStage2RuntimeConfig(
        config_dir=config_dir,
        num_stages=3,
        cpp_project_dir=Path("cpp"),
        cpp_build_dir=tmp_path / "gain_build",
    )

    with CppStage2Upsampler(runtime) as upsampler:
        output = upsampler.process(np.ones(128, dtype=np.float64))

    np.testing.assert_allclose(output[256:], 1.0, rtol=0.0, atol=1e-12)
