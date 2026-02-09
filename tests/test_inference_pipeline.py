"""Tests for Stage 1 -> Stage 2 integrated inference pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from totton_audio_de_mirroring.inference.chunk_processor import (
    ChunkProcessingConfig,
    HannOverlapAddStreamer,
    iterate_chunk_frames,
)
from totton_audio_de_mirroring.inference.pipeline import (
    PipelineConfig,
    ReferenceStage1Processor,
    run_stage1_stage2_pipeline,
)


def test_chunk_frames_with_50_percent_overlap() -> None:
    """Chunk frames should preserve deterministic 50% overlap coverage."""
    signal = np.arange(20, dtype=np.float64)
    frames = tuple(iterate_chunk_frames(signal, chunk_samples=8, overlap_samples=4))

    assert len(frames) == 4
    assert frames[0].samples.shape[0] == 8
    assert frames[1].samples.shape[0] == 8
    assert frames[2].samples.shape[0] == 8
    assert frames[3].samples.shape[0] == 8
    assert np.allclose(frames[0].samples[-4:], frames[1].samples[:4])


def test_hann_ola_streamer_stitches_without_dropouts() -> None:
    """Hann OLA streamer should stitch two chunks with stable continuity."""
    streamer = HannOverlapAddStreamer(chunk_samples=8, overlap_samples=4, window="hann")
    left = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    right = np.linspace(1.0, 0.0, 8, dtype=np.float64)

    first = streamer.process_chunk(left)
    second = streamer.process_chunk(right)
    tail = streamer.finalize()
    merged = np.concatenate([first, second, tail])

    assert merged.shape[0] == 12
    assert np.all(np.isfinite(merged))
    assert np.max(np.abs(merged)) <= 1.05


def test_run_pipeline_reference_mode_outputs_16x_rate(tmp_path: Path) -> None:
    """Reference Stage 1 + identity Stage 2 taps should run end-to-end."""
    config_dir = tmp_path / "taps"
    config_dir.mkdir()
    for i in range(1, 4):
        (config_dir / f"stage{i}_taps.txt").write_text("1.0\n", encoding="utf-8")

    config = PipelineConfig(
        stage2_config_dir=config_dir,
        stage2_backend="python",
        chunk_duration_sec=0.02,
        overlap_ratio=0.5,
        chunk_window="hann",
    )
    processor = ReferenceStage1Processor()
    signal = np.sin(2.0 * np.pi * 440.0 * np.arange(4410, dtype=np.float64) / 44_100.0)

    result = run_stage1_stage2_pipeline(
        signal, stage1_processor=processor, config=config
    )

    assert result.output_signal.ndim == 1
    assert result.output_signal.shape[0] == signal.shape[0] * 16
    assert result.stage1_signal is not None
    assert result.stage1_reference is not None
    assert result.stage1_metrics is not None
    assert result.performance.latency_sec >= 0.0
    assert result.performance.throughput_x_realtime > 0.0
    assert result.performance.num_chunks >= 1
    assert result.performance.chunk_latency_ms >= 0.0


def test_pipeline_stage1_energy_cap_violation_detectable(tmp_path: Path) -> None:
    """Hard-metric payload should flag cap violations under tiny cap."""
    config_dir = tmp_path / "taps_cap"
    config_dir.mkdir()
    for i in range(1, 4):
        (config_dir / f"stage{i}_taps.txt").write_text("1.0\n", encoding="utf-8")

    config = PipelineConfig(
        stage2_config_dir=config_dir,
        stage2_backend="python",
        chunk_duration_sec=0.02,
        overlap_ratio=0.5,
        chunk_window="hann",
        stage1_energy_cap=1.0e-10,
    )
    processor = ReferenceStage1Processor()
    noise = np.random.default_rng(0).standard_normal(4410).astype(np.float64) * 0.1

    result = run_stage1_stage2_pipeline(
        noise, stage1_processor=processor, config=config
    )

    assert result.stage1_metrics is not None
    assert result.stage1_metrics.hb_energy_cap_violated


def test_run_pipeline_cpp_backend_outputs_16x_rate(tmp_path: Path) -> None:
    """Default C++ Stage 2 backend should run end-to-end when cmake is available."""
    if shutil.which("cmake") is None:
        pytest.skip("cmake not available")

    config = PipelineConfig(
        stage2_config_dir=Path("cpp/configs"),
        stage2_num_stages=3,
        stage2_backend="cpp",
        stage2_cpp_project_dir=Path("cpp"),
        stage2_cpp_build_dir=tmp_path / "cpp_build",
        chunk_duration_sec=0.02,
        overlap_ratio=0.5,
        chunk_window="hann",
        evaluate_stage1_metrics=False,
    )
    processor = ReferenceStage1Processor()
    signal = np.sin(2.0 * np.pi * 440.0 * np.arange(2205, dtype=np.float64) / 44_100.0)

    result = run_stage1_stage2_pipeline(
        signal, stage1_processor=processor, config=config
    )

    assert result.output_signal.ndim == 1
    assert result.output_signal.shape[0] == signal.shape[0] * 16
    assert result.stage1_signal is None
    assert result.stage1_reference is None
    assert result.stage1_metrics is None
    assert result.performance.num_chunks >= 1
    assert result.performance.chunk_latency_ms >= 0.0


def test_pipeline_config_rejects_non_hann_or_non_50_percent_overlap() -> None:
    """Issue #33 mandates Hann window with exact 50% overlap."""
    with pytest.raises(ValueError, match="overlap_ratio must be exactly 0.5"):
        _ = PipelineConfig(overlap_ratio=0.25)
    with pytest.raises(ValueError, match="chunk_window must be 'hann'"):
        _ = PipelineConfig(chunk_window="blackman")


def test_chunk_processing_config_matches_issue_defaults() -> None:
    """Chunk helper config should expose expected overlap/hop values."""
    config = ChunkProcessingConfig(sample_rate=44_100, chunk_duration_sec=0.25)
    assert config.chunk_samples == 11_025
    assert config.overlap_samples in {5_512, 5_513}
    assert config.hop_samples == config.chunk_samples - config.overlap_samples
