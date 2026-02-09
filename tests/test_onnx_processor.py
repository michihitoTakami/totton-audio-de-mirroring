"""Tests for ONNX Runtime Stage 1 processor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from totton_audio_de_mirroring.inference.onnx_processor import (
    OnnxStage1Processor,
    load_onnx_stage1_processor,
)
from totton_audio_de_mirroring.models.nmse import STFTConfig


@dataclass(frozen=True)
class _DummyInput:
    name: str


class _DummySession:
    def __init__(self) -> None:
        self._inputs = [_DummyInput(name="input_signal")]

    def get_inputs(self) -> list[Any]:
        return self._inputs

    def run(
        self, _output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        signal = input_feed["input_signal"]
        return [np.full_like(signal, fill_value=np.float32(0.5))]


def test_onnx_stage1_processor_runs_with_valid_input() -> None:
    taps = np.zeros(4097, dtype=np.float64)
    taps[2048] = 1.0
    processor = OnnxStage1Processor(
        session=_DummySession(),
        sample_rate=88_200,
        cutoff_hz=20_000.0,
        energy_cap=1.0e-3,
        envelope_floor=0.2,
        lowpass_taps=taps,
        highpass_taps=taps,
        stft_config=STFTConfig(),
    )
    time = np.arange(1024, dtype=np.float64) / 44_100.0
    signal = np.sin(2.0 * np.pi * 440.0 * time)

    output = processor.process(
        signal=signal,
        source_sample_rate=44_100,
        target_sample_rate=88_200,
    )

    assert output.ndim == 1
    assert output.shape[0] == signal.shape[0] * 2
    assert np.all(np.isfinite(output))


def test_onnx_stage1_processor_rejects_non_2x_ratio() -> None:
    taps = np.zeros(4097, dtype=np.float64)
    taps[2048] = 1.0
    processor = OnnxStage1Processor(
        session=_DummySession(),
        sample_rate=88_200,
        cutoff_hz=20_000.0,
        energy_cap=1.0e-3,
        envelope_floor=0.2,
        lowpass_taps=taps,
        highpass_taps=taps,
        stft_config=STFTConfig(),
    )
    signal = np.ones(256, dtype=np.float64)
    with pytest.raises(ValueError, match="exact 2x"):
        _ = processor.process(
            signal=signal,
            source_sample_rate=44_100,
            target_sample_rate=96_000,
        )


def test_load_onnx_stage1_processor_rejects_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        _ = load_onnx_stage1_processor(
            model_path=tmp_path / "missing.onnx",
            data_config_path=tmp_path / "data.yaml",
        )
