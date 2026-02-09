"""Tests for Stage 1 -> Stage 2 integration CLI script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scripts.run_stage1_stage2_pipeline import _build_stage1_processor, main


def test_cli_json_e2e_with_reference_stage1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Config-driven CLI should run end-to-end and emit JSON payload."""
    config_dir = tmp_path / "taps"
    config_dir.mkdir()
    for i in range(1, 4):
        (config_dir / f"stage{i}_taps.txt").write_text("1.0\n", encoding="utf-8")

    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pipeline:",
                "  source_sample_rate: 44100",
                "  stage1_sample_rate: 88200",
                "  output_sample_rate: 705600",
                f"  stage2_config_dir: {config_dir.as_posix()}",
                "  stage2_num_stages: 3",
                "  stage2_backend: python",
                "  chunk_duration_sec: 0.02",
                "  overlap_ratio: 0.5",
                "  chunk_window: hann",
                "  stage1_energy_cap: 0.001",
                "  evaluate_stage1_metrics: true",
                "stage1:",
                "  mode: reference",
            ]
        ),
        encoding="utf-8",
    )

    input_signal = np.sin(
        2.0 * np.pi * 440.0 * np.arange(2205, dtype=np.float64) / 44_100.0
    )
    input_path = tmp_path / "input.npy"
    np.save(input_path, input_signal)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage1_stage2_pipeline.py",
            "--config",
            str(config_path),
            "--input-npy",
            str(input_path),
            "--json",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["output_sample_rate"] == 705_600
    assert payload["stage2_num_stages"] == 3
    assert payload["stage2_backend"] == "python"
    assert payload["performance"]["throughput_x_realtime"] > 0.0
    assert payload["performance"]["num_chunks"] >= 1
    assert payload["performance"]["chunk_latency_ms"] >= 0.0
    assert payload["num_output_samples"] > input_signal.shape[0] * 10
    assert "stage1_metrics" in payload


def test_build_stage1_processor_nmse_requires_checkpoint_path() -> None:
    """NMSE mode should reject missing checkpoint path explicitly."""
    with pytest.raises(ValueError, match="requires checkpoint_path"):
        _ = _build_stage1_processor({"mode": "nmse"})


def test_build_stage1_processor_onnx_requires_model_path() -> None:
    """ONNX mode should reject missing model path explicitly."""
    with pytest.raises(ValueError, match="requires model_path"):
        _ = _build_stage1_processor({"mode": "onnx"})
