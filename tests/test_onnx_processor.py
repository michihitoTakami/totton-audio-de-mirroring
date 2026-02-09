"""Tests for ONNX Runtime Stage 1 processor."""

from __future__ import annotations

import sys
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


class _DummyOrtSession:
    def __init__(self, _model_path: str, providers: list[str]) -> None:
        self.providers = providers
        self._inputs = [_DummyInput(name="input_magnitude")]

    def get_inputs(self) -> list[Any]:
        return self._inputs

    def run(
        self, _output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        tensor = input_feed["input_magnitude"]
        return [np.asarray(tensor, dtype=np.float32)]


class _DummyOrtModule:
    def __init__(self, available_providers: list[str]) -> None:
        self._available_providers = available_providers
        self.last_session: _DummyOrtSession | None = None

    def get_available_providers(self) -> list[str]:
        return self._available_providers

    def InferenceSession(
        self, model_path: str, providers: list[str]
    ) -> _DummyOrtSession:
        session = _DummyOrtSession(model_path, providers)
        self.last_session = session
        return session


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


def test_load_onnx_stage1_processor_prefers_cuda_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")
    data_config = Path("configs/data_generation.yaml")
    dummy_ort = _DummyOrtModule(
        available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", dummy_ort)

    _ = load_onnx_stage1_processor(
        model_path=model_path,
        data_config_path=data_config,
        device="cuda",
    )

    assert dummy_ort.last_session is not None
    assert dummy_ort.last_session.providers == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_load_onnx_stage1_processor_rejects_silent_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")
    data_config = Path("configs/data_generation.yaml")
    dummy_ort = _DummyOrtModule(available_providers=["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", dummy_ort)

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is not available"):
        _ = load_onnx_stage1_processor(
            model_path=model_path,
            data_config_path=data_config,
            device="cuda",
        )


def test_load_onnx_stage1_processor_allows_cpu_fallback_when_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")
    data_config = Path("configs/data_generation.yaml")
    dummy_ort = _DummyOrtModule(available_providers=["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", dummy_ort)

    _ = load_onnx_stage1_processor(
        model_path=model_path,
        data_config_path=data_config,
        device="cuda",
        allow_cpu_fallback=True,
    )

    assert dummy_ort.last_session is not None
    assert dummy_ort.last_session.providers == ["CPUExecutionProvider"]
