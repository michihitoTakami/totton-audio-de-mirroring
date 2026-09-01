"""Tests for CAPB waveform-contract export metadata."""

import pytest
from scripts.export_capb_to_onnx import _capb_metadata

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
    }


def test_capb_metadata_rejects_invalid_rate() -> None:
    model = CAPB()

    with pytest.raises(ValueError, match="positive"):
        _capb_metadata(0, model)
