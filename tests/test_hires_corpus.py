"""Tests for the hi-res teacher corpus loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from totton_audio_de_mirroring.data.hires_corpus import (
    HiResCorpus,
    HiResCorpusConfig,
    discover_hires_files,
    high_frequency_energy_ratio,
    resample_signal,
)


def _write_hires_wav(
    path: Path,
    *,
    sample_rate: int = 88_200,
    duration_sec: float = 2.0,
    include_ultrasonic: bool = True,
    channels: int = 2,
) -> None:
    t = np.arange(int(duration_sec * sample_rate)) / sample_rate
    signal = 0.3 * np.sin(2 * np.pi * 5_000 * t)
    if include_ultrasonic:
        signal = signal + 0.2 * np.sin(2 * np.pi * 30_000 * t)
    block = np.stack([signal] * channels, axis=1).astype(np.float32)
    sf.write(str(path), block, sample_rate, subtype="PCM_24")


def test_discover_hires_files_filters_extensions(tmp_path: Path) -> None:
    _write_hires_wav(tmp_path / "a.wav")
    (tmp_path / "note.txt").write_text("ignore me", encoding="utf-8")
    files = discover_hires_files(tmp_path, (".wav", ".flac"))
    assert files == [tmp_path / "a.wav"]


def test_discover_hires_files_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_hires_files(tmp_path / "nope", (".wav",))


def test_high_frequency_energy_ratio_detects_ultrasonic() -> None:
    sr = 88_200
    t = np.arange(sr) / sr
    ultrasonic = np.sin(2 * np.pi * 30_000 * t)
    low = np.sin(2 * np.pi * 1_000 * t)
    assert high_frequency_energy_ratio(ultrasonic, sr, split_hz=22_050.0) > 0.9
    assert high_frequency_energy_ratio(low, sr, split_hz=22_050.0) < 1.0e-3


def test_resample_signal_changes_length_by_ratio() -> None:
    sr_in, sr_out = 96_000, 88_200
    signal = np.sin(2 * np.pi * 1_000 * np.arange(sr_in) / sr_in)
    out = resample_signal(signal, sr_in, sr_out)
    assert abs(out.shape[0] - sr_out) <= 2


def test_corpus_loads_target_length_segment(tmp_path: Path) -> None:
    _write_hires_wav(tmp_path / "tone.wav", sample_rate=88_200, duration_sec=3.0)
    corpus = HiResCorpus(
        HiResCorpusConfig(root=tmp_path),
        target_sample_rate=88_200,
        source_duration_sec=1.0,
    )
    assert corpus.num_files == 1
    rng = np.random.default_rng(0)
    segment = corpus.load_teacher_source(0, rng)
    assert segment.shape[0] == 88_200


def test_corpus_rejects_low_sample_rate_files(tmp_path: Path) -> None:
    _write_hires_wav(tmp_path / "low.wav", sample_rate=48_000)
    with pytest.raises(FileNotFoundError):
        HiResCorpus(
            HiResCorpusConfig(root=tmp_path, min_sample_rate=88_200),
            target_sample_rate=88_200,
            source_duration_sec=1.0,
        )


def test_corpus_rejects_segment_without_ultrasonic_energy(tmp_path: Path) -> None:
    _write_hires_wav(
        tmp_path / "flat.wav", sample_rate=88_200, include_ultrasonic=False
    )
    corpus = HiResCorpus(
        HiResCorpusConfig(root=tmp_path, min_hf_energy_ratio=1.0e-2),
        target_sample_rate=88_200,
        source_duration_sec=1.0,
    )
    with pytest.raises(ValueError, match="high-frequency energy"):
        corpus.load_teacher_source(0, np.random.default_rng(0))


def test_corpus_retries_past_low_energy_segments(tmp_path: Path) -> None:
    """Loader should skip low-HF files and return a genuinely hi-res one."""
    _write_hires_wav(tmp_path / "a_flat.wav", include_ultrasonic=False)
    _write_hires_wav(tmp_path / "b_rich.wav", include_ultrasonic=True)
    corpus = HiResCorpus(
        HiResCorpusConfig(root=tmp_path, min_hf_energy_ratio=1.0e-2),
        target_sample_rate=88_200,
        source_duration_sec=1.0,
    )
    # index 0 maps to the flat file first; retry must advance to the rich file.
    segment = corpus.load_teacher_source(0, np.random.default_rng(0))
    assert segment.shape[0] == 88_200


def test_corpus_config_rejects_subrate_min(tmp_path: Path) -> None:
    _write_hires_wav(tmp_path / "tone.wav")
    with pytest.raises(ValueError, match="min_sample_rate must be >="):
        HiResCorpus(
            HiResCorpusConfig(root=tmp_path, min_sample_rate=44_100),
            target_sample_rate=88_200,
            source_duration_sec=1.0,
        )
