"""Tests for CAPB waveform-contract export metadata."""

from pathlib import Path

import pytest
import torch
from scripts.export_capb_to_onnx import (
    CAPBControllerOnly,
    _capb_metadata,
    _default_controller_output,
)

from totton_audio_de_mirroring.models.capb import CAPB
from totton_audio_de_mirroring.models.proto_bank import (
    build_prototype_bank_for_profile,
)


def test_capb_metadata_binds_profile_hash_and_precision() -> None:
    bank = build_prototype_bank_for_profile(88_200, "long_sharp_1535_a120")
    model = CAPB(bank=bank, fir_compute_dtype="float32")

    metadata = _capb_metadata(44_100, model)

    assert metadata == {
        "expected_input_rate": "44100",
        "cuda_compute_precision": "strict_fp32",
        "prototype_profile": "long_sharp_1535_a120",
        "prototype_hash": bank.coefficient_hash,
        "fir_compute_dtype": "float32",
        "capb_model_role": "upsampler",
        "controller_control_stride": "64",
        "controller_weight_layout": "batch,prototype,frame",
        "controller_weight_interpolation": (
            "linear,align_corners=false,target=2x_input_samples"
        ),
        "prototype_names": "sharp,mid,gentle",
    }


def test_capb_metadata_rejects_invalid_rate() -> None:
    model = CAPB()

    with pytest.raises(ValueError, match="positive"):
        _capb_metadata(0, model)


def test_controller_only_matches_capb_controller_weights() -> None:
    model = CAPB()
    source = torch.tensor([[0.0, 0.2, -0.4, 0.1] * 32])

    actual = CAPBControllerOnly(model)(source)

    assert torch.equal(actual, model.controller_weights(source))
    assert actual.shape[:2] == (1, model.num_prototypes)
    assert torch.allclose(actual.sum(dim=1), torch.ones_like(actual[:, 0]))


def test_default_controller_output_is_sibling_onnx() -> None:
    assert _default_controller_output(Path("models/capb_stage1.onnx")) == Path(
        "models/capb_stage1_controller.onnx"
    )


def test_default_controller_output_rejects_non_onnx_path() -> None:
    with pytest.raises(ValueError, match=".onnx"):
        _default_controller_output(Path("models/capb_stage1.bin"))
