"""Tests for distillation training script helpers."""

from argparse import Namespace
from pathlib import Path

import pytest
import torch
from scripts.train_distillation import (
    _apply_overrides,
    _emit_stage1_light_checkpoint,
    _teacher_tag,
    _validate_teacher_checkpoint_teacher_type,
)

from totton_audio_de_mirroring.training.distillation import DistillationConfig


def test_apply_overrides_rejects_conflicting_cuda_flags() -> None:
    args = Namespace(
        pruning_ratio=0.0,
        epochs=None,
        learning_rate=None,
        seed=None,
        device=None,
        energy_cap=None,
        teacher_type=None,
        hb_loss_weight=None,
        preserve_lb_weight=None,
        require_cuda=True,
        allow_cpu=True,
    )
    with pytest.raises(ValueError, match="Specify only one"):
        _ = _apply_overrides(DistillationConfig(require_cuda=False), args)


def test_apply_overrides_rejects_invalid_pruning_ratio() -> None:
    args = Namespace(
        pruning_ratio=1.0,
        epochs=None,
        learning_rate=None,
        seed=None,
        device=None,
        energy_cap=None,
        teacher_type=None,
        hb_loss_weight=None,
        preserve_lb_weight=None,
        require_cuda=False,
        allow_cpu=False,
    )
    with pytest.raises(ValueError, match="pruning_ratio"):
        _ = _apply_overrides(DistillationConfig(require_cuda=False), args)


def test_apply_overrides_updates_learning_rate() -> None:
    args = Namespace(
        pruning_ratio=0.2,
        epochs=None,
        learning_rate=3.0e-4,
        seed=None,
        device=None,
        energy_cap=None,
        teacher_type=None,
        hb_loss_weight=None,
        preserve_lb_weight=None,
        require_cuda=False,
        allow_cpu=False,
    )
    updated = _apply_overrides(DistillationConfig(require_cuda=False), args)
    assert updated.learning_rate == pytest.approx(3.0e-4)


def test_apply_overrides_updates_teacher_policy_fields() -> None:
    args = Namespace(
        pruning_ratio=0.0,
        epochs=None,
        learning_rate=None,
        seed=None,
        device=None,
        energy_cap=None,
        teacher_type="bessel_88k2",
        hb_loss_weight=1.2,
        preserve_lb_weight=1.4,
        require_cuda=False,
        allow_cpu=False,
    )
    updated = _apply_overrides(DistillationConfig(require_cuda=False), args)
    assert updated.teacher_type == "bessel_88k2"
    assert updated.hb_loss_weight == pytest.approx(1.2)
    assert updated.preserve_lb_weight == pytest.approx(1.4)


def test_emit_stage1_light_checkpoint_copies_best(tmp_path: Path) -> None:
    best = tmp_path / "stage1_distill_raw88_best.pt"
    best.write_bytes(b"checkpoint")
    emitted = _emit_stage1_light_checkpoint(
        best_checkpoint=best,
        checkpoint_dir=tmp_path,
        teacher_type="raw_88k2",
    )
    assert emitted == tmp_path / "stage1_light_raw88.pt"
    assert (tmp_path / "stage1_light.pt").read_bytes() == b"checkpoint"
    assert emitted.read_bytes() == b"checkpoint"


def test_teacher_tag_maps_supported_types() -> None:
    assert _teacher_tag("raw_88k2") == "raw88"
    assert _teacher_tag("bessel_88k2") == "bessel"


def test_validate_teacher_checkpoint_type_mismatch(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"training_config": {"teacher_type": "bessel_88k2"}}, checkpoint_path)
    with pytest.raises(RuntimeError, match="Teacher checkpoint type mismatch"):
        _validate_teacher_checkpoint_teacher_type(
            checkpoint_path=checkpoint_path,
            expected_teacher_type="raw_88k2",
        )
