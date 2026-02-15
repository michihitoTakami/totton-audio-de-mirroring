"""Tests for TensorRT export script."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest
from scripts import export_to_tensorrt


def test_parse_modes_deduplicates_and_validates() -> None:
    assert export_to_tensorrt._parse_modes("mixed,fp32,mixed") == ["mixed", "fp32"]

    with pytest.raises(ValueError, match="Unsupported mode"):
        _ = export_to_tensorrt._parse_modes("int8")


def test_validate_args_rejects_invalid_profile(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    args = Namespace(
        onnx_path=onnx_path,
        workspace_mb=1024,
        freq_bins=513,
        min_time_frames=64,
        opt_time_frames=32,
        max_time_frames=128,
    )

    with pytest.raises(ValueError, match="opt_time_frames"):
        export_to_tensorrt._validate_args(args)


def test_parse_args_defaults_cover_up_to_two_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["export_to_tensorrt.py"])
    args = export_to_tensorrt.parse_args()

    assert args.min_time_frames == 87
    assert args.opt_time_frames == 345
    assert args.max_time_frames == 690


def test_main_builds_requested_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "out"
    onnx_path.write_bytes(b"onnx")

    args = Namespace(
        onnx_path=onnx_path,
        output_dir=output_dir,
        modes="fp32,mixed",
        workspace_mb=1024,
        freq_bins=513,
        min_time_frames=16,
        opt_time_frames=32,
        max_time_frames=64,
        strict_mixed_io_fp32=True,
    )
    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(export_to_tensorrt, "parse_args", lambda: args)

    def _fake_build_engine(
        *,
        onnx_path: Path,
        output_path: Path,
        mode: export_to_tensorrt.TensorRtBuildMode,
        workspace_mb: int,
        freq_bins: int,
        min_time_frames: int,
        opt_time_frames: int,
        max_time_frames: int,
        strict_mixed_io_fp32: bool,
    ) -> None:
        assert onnx_path.exists()
        assert workspace_mb == 1024
        assert freq_bins == 513
        assert min_time_frames == 16
        assert opt_time_frames == 32
        assert max_time_frames == 64
        assert strict_mixed_io_fp32 is True
        output_path.write_bytes(mode.name.encode("utf-8"))
        calls.append((mode.name, output_path))

    monkeypatch.setattr(export_to_tensorrt, "_build_engine", _fake_build_engine)

    export_to_tensorrt.main()

    assert [name for name, _ in calls] == ["fp32", "mixed"]
    for _, path in calls:
        assert path.exists()


def test_build_engine_applies_fp16_flag_for_mixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    output_path = tmp_path / "out.engine"
    onnx_path.write_bytes(b"dummy")

    events: dict[str, object] = {}

    class _DummyProfile:
        def set_shape(
            self,
            name: str,
            min: tuple[int, ...],
            opt: tuple[int, ...],
            max: tuple[int, ...],
        ) -> None:
            events["shape"] = (name, min, opt, max)

    class _DummyConfig:
        def __init__(self) -> None:
            self.flags: list[object] = []

        def set_memory_pool_limit(self, _pool: object, size: int) -> None:
            events["workspace"] = size

        def set_flag(self, flag: object) -> None:
            self.flags.append(flag)

        def add_optimization_profile(self, _profile: _DummyProfile) -> None:
            events["profile_added"] = True

    class _DummyTensor:
        def __init__(self, name: str) -> None:
            self.name = name
            self.dtype: object | None = None

    class _DummyNetwork:
        def __init__(self) -> None:
            self.input = _DummyTensor("input_magnitude")
            self.output = _DummyTensor("output_mask")

        def get_input(self, _index: int) -> _DummyTensor:
            return self.input

        def get_output(self, _index: int) -> _DummyTensor:
            return self.output

    class _DummyParser:
        num_errors = 0

        def __init__(self, _network: _DummyNetwork, _logger: object) -> None:
            pass

        def parse(self, _payload: bytes) -> bool:
            return True

    class _DummyBuilder:
        def __init__(self, _logger: object) -> None:
            self.network = _DummyNetwork()
            self.config = _DummyConfig()

        def create_network(self, _flags: int) -> _DummyNetwork:
            return self.network

        def create_builder_config(self) -> _DummyConfig:
            return self.config

        def create_optimization_profile(self) -> _DummyProfile:
            return _DummyProfile()

        def build_serialized_network(
            self, _network: _DummyNetwork, _config: _DummyConfig
        ) -> bytes:
            events["flags"] = list(_config.flags)
            return b"engine"

    class _DummyTrt:
        class Logger:
            WARNING = 1

            def __init__(self, _level: int) -> None:
                pass

        class NetworkDefinitionCreationFlag:
            EXPLICIT_BATCH = 0

        class MemoryPoolType:
            WORKSPACE = 0

        class BuilderFlag:
            FP16 = "fp16"
            PREFER_PRECISION_CONSTRAINTS = "prefer"
            OBEY_PRECISION_CONSTRAINTS = "obey"

        float32 = "float32"
        Builder = _DummyBuilder
        OnnxParser = _DummyParser

    monkeypatch.setattr(export_to_tensorrt, "_import_tensorrt", lambda: _DummyTrt)

    export_to_tensorrt._build_engine(
        onnx_path=onnx_path,
        output_path=output_path,
        mode=export_to_tensorrt.TensorRtBuildMode(
            name="mixed", enable_fp16=True, mixed_precision=True
        ),
        workspace_mb=512,
        freq_bins=513,
        min_time_frames=16,
        opt_time_frames=32,
        max_time_frames=64,
        strict_mixed_io_fp32=True,
    )

    assert output_path.exists()
    assert events["workspace"] == 512 * 1024 * 1024
    assert events["profile_added"] is True
    flags = events["flags"]
    assert isinstance(flags, list)
    assert "fp16" in flags
    assert "prefer" in flags
    assert "obey" in flags
