"""Tests for TensorRT Stage 1 processor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from totton_audio_de_mirroring.inference.tensorrt_processor import (
    TensorRtStage1Processor,
    _resolve_io_names,
    load_tensorrt_stage1_processor,
)
from totton_audio_de_mirroring.models.nmse import STFTConfig


class _DummySession:
    def run(self, input_magnitude: np.ndarray) -> np.ndarray:
        return np.asarray(input_magnitude, dtype=np.float32)


def test_tensorrt_stage1_processor_runs_with_valid_input() -> None:
    taps = np.zeros(4097, dtype=np.float64)
    taps[2048] = 1.0
    processor = TensorRtStage1Processor(
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


def test_tensorrt_stage1_processor_rejects_non_2x_ratio() -> None:
    taps = np.zeros(4097, dtype=np.float64)
    taps[2048] = 1.0
    processor = TensorRtStage1Processor(
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


def test_load_tensorrt_stage1_processor_rejects_missing_engine(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="TensorRT engine not found"):
        _ = load_tensorrt_stage1_processor(
            engine_path=tmp_path / "missing.engine",
            data_config_path=tmp_path / "data.yaml",
        )


def test_load_tensorrt_stage1_processor_rejects_non_cuda_device(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"dummy")
    with pytest.raises(ValueError, match="device must be 'cuda'"):
        _ = load_tensorrt_stage1_processor(
            engine_path=engine_path,
            data_config_path=Path("configs/data_generation.yaml"),
            device="cpu",
        )


def test_load_tensorrt_stage1_processor_builds_with_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"dummy")

    monkeypatch.setattr(
        "totton_audio_de_mirroring.inference.tensorrt_processor._load_tensorrt_session",
        lambda _path: _DummySession(),
    )
    monkeypatch.setattr(
        "totton_audio_de_mirroring.inference.tensorrt_processor.torch.cuda.is_available",
        lambda: True,
    )

    processor = load_tensorrt_stage1_processor(
        engine_path=engine_path,
        data_config_path=Path("configs/data_generation.yaml"),
        device="cuda",
    )

    assert isinstance(processor, TensorRtStage1Processor)
    assert processor.envelope_floor == pytest.approx(0.2)


@dataclass(frozen=True)
class _DummyTensorIOMode:
    INPUT: int = 0
    OUTPUT: int = 1


class _DummyNewApiEngine:
    num_io_tensors = 2

    def get_tensor_name(self, index: int) -> str:
        return ["input", "output"][index]

    def get_tensor_mode(self, name: str) -> int:
        return 0 if name == "input" else 1


class _DummyOldApiEngine:
    num_bindings = 2

    def get_binding_name(self, index: int) -> str:
        return ["input", "output"][index]

    def binding_is_input(self, index: int) -> bool:
        return index == 0


def test_resolve_io_names_supports_new_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyTrt:
        TensorIOMode = _DummyTensorIOMode

    import sys

    monkeypatch.setitem(sys.modules, "tensorrt", _DummyTrt)
    input_name, output_name = _resolve_io_names(_DummyNewApiEngine())

    assert input_name == "input"
    assert output_name == "output"


def test_resolve_io_names_supports_old_api() -> None:
    input_name, output_name = _resolve_io_names(_DummyOldApiEngine())

    assert input_name == "input"
    assert output_name == "output"
