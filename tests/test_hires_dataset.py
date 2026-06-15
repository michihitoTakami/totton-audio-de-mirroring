"""Tests for the hi-res teacher dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from totton_audio_de_mirroring.data.hires_corpus import HiResCorpusConfig
from totton_audio_de_mirroring.data.hires_dataset import HiResTeacherDataset
from totton_audio_de_mirroring.data.pipeline_config import DataPipelineConfig


def _write_hires_wav(path: Path, *, sample_rate: int = 88_200) -> None:
    t = np.arange(int(3.0 * sample_rate)) / sample_rate
    signal = 0.3 * np.sin(2 * np.pi * 5_000 * t) + 0.2 * np.sin(2 * np.pi * 30_000 * t)
    block = np.stack([signal, signal * 0.9], axis=1).astype(np.float32)
    sf.write(str(path), block, sample_rate, subtype="PCM_24")


def _config(seed: int | None = 1234) -> DataPipelineConfig:
    return DataPipelineConfig(
        num_samples=4,
        source_sample_rate=44_100,
        target_sample_rate=88_200,
        source_duration_sec=1.0,
        chunk_duration_sec=0.25,
        seed=seed,
        teacher_type="raw_88k2",
    )


def test_hires_dataset_matches_stage1_contract(tmp_path: Path) -> None:
    _write_hires_wav(tmp_path / "tone.wav")
    dataset = HiResTeacherDataset(_config(), HiResCorpusConfig(root=tmp_path))
    assert len(dataset) == 4
    sample = dataset[0]
    expected_keys = {
        "source",
        "x_full",
        "low_band",
        "high_band",
        "hb_target",
        "mirror_mask",
        "teacher_type",
        "input_route",
        "target_route",
        "profile",
        "signal_type",
        "chunk_start",
    }
    assert expected_keys.issubset(sample.keys())
    assert sample["high_band"].shape[-1] == int(0.25 * 88_200)
    assert sample["hb_target"].shape == sample["high_band"].shape
    assert sample["mirror_mask"].ndim == 2
    assert sample["signal_type"] == "hires"
    assert sample["teacher_type"] == "raw_88k2"


def test_hires_dataset_is_deterministic_by_seed(tmp_path: Path) -> None:
    _write_hires_wav(tmp_path / "tone.wav")
    dataset = HiResTeacherDataset(_config(), HiResCorpusConfig(root=tmp_path))
    first = dataset[1]
    second = dataset[1]
    assert torch.allclose(first["high_band"], second["high_band"])
    assert torch.allclose(first["hb_target"], second["hb_target"])


def test_hires_dataset_rejects_non_raw_teacher(tmp_path: Path) -> None:
    _write_hires_wav(tmp_path / "tone.wav")
    config = DataPipelineConfig(
        num_samples=4,
        teacher_type="bessel_88k2",
    )
    with pytest.raises(ValueError, match="requires a raw teacher type"):
        HiResTeacherDataset(config, HiResCorpusConfig(root=tmp_path))


def test_hires_dataset_index_out_of_range(tmp_path: Path) -> None:
    _write_hires_wav(tmp_path / "tone.wav")
    dataset = HiResTeacherDataset(_config(), HiResCorpusConfig(root=tmp_path))
    with pytest.raises(IndexError):
        _ = dataset[99]
