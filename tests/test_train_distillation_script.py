"""Tests for distillation training script helpers."""

from argparse import Namespace
from pathlib import Path

import pytest
from scripts.train_distillation import _apply_overrides, _emit_stage1_light_checkpoint

from totton_audio_de_mirroring.training.distillation import DistillationConfig


def test_apply_overrides_rejects_conflicting_cuda_flags() -> None:
    args = Namespace(
        pruning_ratio=0.0,
        epochs=None,
        learning_rate=None,
        seed=None,
        device=None,
        energy_cap=None,
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
        require_cuda=False,
        allow_cpu=False,
    )
    updated = _apply_overrides(DistillationConfig(require_cuda=False), args)
    assert updated.learning_rate == pytest.approx(3.0e-4)


def test_emit_stage1_light_checkpoint_copies_best(tmp_path: Path) -> None:
    best = tmp_path / "stage1_distill_best.pt"
    best.write_bytes(b"checkpoint")
    emitted = _emit_stage1_light_checkpoint(
        best_checkpoint=best,
        checkpoint_dir=tmp_path,
    )
    assert emitted == tmp_path / "stage1_light.pt"
    assert emitted.read_bytes() == b"checkpoint"
