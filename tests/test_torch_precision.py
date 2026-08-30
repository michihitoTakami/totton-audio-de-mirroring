"""Tests for CAPB Torch execution-precision policy."""

import torch

from totton_audio_de_mirroring.torch_precision import configure_torch_precision


def test_cpu_precision_is_always_strict_fp32() -> None:
    record = configure_torch_precision("cpu", allow_tf32=True)

    assert record.device == "cpu"
    assert record.precision_mode == "strict_fp32"
    assert not record.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert not torch.backends.cuda.matmul.allow_tf32


def test_precision_record_reports_deterministic_policy() -> None:
    record = configure_torch_precision("cpu", deterministic=True)

    assert record.deterministic
    assert torch.are_deterministic_algorithms_enabled()

    configure_torch_precision("cpu", deterministic=False)
