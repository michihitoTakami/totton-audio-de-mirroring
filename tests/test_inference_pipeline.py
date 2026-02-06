"""Tests for Stage 1 -> Stage 2 integrated inference pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from totton_audio_de_mirroring.inference.pipeline import (
    PipelineConfig,
    ReferenceStage1Processor,
    _crossfade_append,
    _iterate_chunks,
    run_stage1_stage2_pipeline,
)


def test_iterate_chunks_with_overlap() -> None:
    """Chunk iterator should include overlap with deterministic coverage."""
    signal = np.arange(20, dtype=np.float64)
    chunks = _iterate_chunks(signal, chunk_samples=8, crossfade_samples=2)

    assert len(chunks) == 3
    assert chunks[0].shape[0] == 8
    assert chunks[1].shape[0] == 8
    assert chunks[2].shape[0] == 8
    assert np.allclose(chunks[0][-2:], chunks[1][:2])


def test_crossfade_append_reduces_boundary_jump() -> None:
    """Linear crossfade should avoid hard discontinuity at chunk boundary."""
    left = np.ones(8, dtype=np.float64)
    right = np.zeros(8, dtype=np.float64)
    merged = _crossfade_append(left, right, crossfade_samples=4)

    assert merged.shape[0] == 12
    assert merged[4] > merged[5]
    assert merged[5] > merged[6]
    assert merged[6] > merged[7]
    assert merged[7] > merged[8]


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
        crossfade_duration_sec=0.005,
    )
    processor = ReferenceStage1Processor()
    signal = np.sin(2.0 * np.pi * 440.0 * np.arange(4410, dtype=np.float64) / 44_100.0)

    result = run_stage1_stage2_pipeline(
        signal, stage1_processor=processor, config=config
    )

    assert result.output_signal.ndim == 1
    assert result.output_signal.shape[0] > signal.shape[0] * 12
    assert result.stage1_signal is not None
    assert result.stage1_reference is not None
    assert result.stage1_metrics is not None
    assert result.performance.latency_sec >= 0.0
    assert result.performance.throughput_x_realtime > 0.0


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
        crossfade_duration_sec=0.005,
        stage1_energy_cap=1.0e-10,
    )
    processor = ReferenceStage1Processor()
    noise = np.random.default_rng(0).standard_normal(4410).astype(np.float64) * 0.1

    result = run_stage1_stage2_pipeline(
        noise, stage1_processor=processor, config=config
    )

    assert result.stage1_metrics is not None
    assert result.stage1_metrics.hb_energy_cap_violated
