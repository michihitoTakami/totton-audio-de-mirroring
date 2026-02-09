"""Tests for full-song comparison CLI helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scripts.compare_full_song import _build_pipeline_config, _read_wav_mono


def test_build_pipeline_config_rejects_invalid_type() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        _ = _build_pipeline_config(raw=[])  # type: ignore[arg-type]


def test_read_wav_mono_converts_stereo(tmp_path: Path) -> None:
    wav_path = tmp_path / "stereo.wav"
    sr = 44_100
    left = np.linspace(-0.5, 0.5, 128, dtype=np.float64)
    right = np.linspace(0.5, -0.5, 128, dtype=np.float64)
    stereo = np.stack([left, right], axis=1)
    sf.write(wav_path, stereo, sr)

    mono = _read_wav_mono(wav_path, sample_rate=sr)
    assert mono.ndim == 1
    assert mono.shape[0] == 128
