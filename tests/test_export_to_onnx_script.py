"""Tests for ONNX export CLI script."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from scripts import export_to_onnx


class _DummyNmseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = torch.nn.Identity()
        self.unet.down_blocks = [object(), object(), object(), object()]

    def _stft(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.complex(signal.unsqueeze(1), torch.zeros_like(signal).unsqueeze(1))


@dataclass(frozen=True)
class _DummyProcessor:
    model: torch.nn.Module


def test_validate_args_rejects_low_opset() -> None:
    args = Namespace(
        opset_version=16,
        dummy_samples=1,
        tolerance=1.0e-5,
    )
    with pytest.raises(ValueError, match=">= 17"):
        export_to_onnx._validate_args(args)


def test_main_uses_dynamic_axes_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}
    output_path = tmp_path / "model.onnx"
    checkpoint_path = tmp_path / "stage1.pt"
    data_config_path = tmp_path / "data.yaml"
    checkpoint_path.write_bytes(b"dummy")
    data_config_path.write_text("dummy: true\n", encoding="utf-8")

    args = Namespace(
        checkpoint_path=checkpoint_path,
        data_config_path=data_config_path,
        output_path=output_path,
        opset_version=17,
        dummy_samples=512,
        disable_dynamic_axes=False,
        device="cpu",
        check_model=False,
        verify_ort=False,
        tolerance=1.0e-5,
    )
    monkeypatch.setattr(export_to_onnx, "parse_args", lambda: args)
    monkeypatch.setattr(
        export_to_onnx,
        "load_nmse_stage1_processor",
        lambda **_: _DummyProcessor(model=_DummyNmseModel()),
    )

    def _fake_export(**kwargs: Any) -> None:
        called.update(kwargs)
        output_path.write_bytes(b"onnx")

    monkeypatch.setattr(export_to_onnx.torch.onnx, "export", _fake_export)

    export_to_onnx.main()

    assert output_path.exists()
    assert called["opset_version"] == 17
    assert called["dynamic_axes"] == {
        "input_magnitude": {0: "batch_size", 2: "freq_bins", 3: "time_frames"},
        "output_mask": {0: "batch_size", 2: "freq_bins", 3: "time_frames"},
    }
