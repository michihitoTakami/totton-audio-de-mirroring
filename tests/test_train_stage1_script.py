"""Tests for Stage 1 training script helpers."""

from argparse import Namespace
from pathlib import Path

import pytest
from scripts.train_stage1 import _apply_overrides, _resolve_train_config_path

from totton_audio_de_mirroring.training.trainer import TrainingConfig


def test_resolve_train_config_prefers_train_config() -> None:
    args = Namespace(config=None, train_config=Path("b.yaml"))
    assert _resolve_train_config_path(args) == Path("b.yaml")


def test_resolve_train_config_uses_config_alias() -> None:
    args = Namespace(config=Path("a.yaml"), train_config=None)
    assert _resolve_train_config_path(args) == Path("a.yaml")


def test_resolve_train_config_rejects_both_options() -> None:
    args = Namespace(config=Path("a.yaml"), train_config=Path("b.yaml"))
    with pytest.raises(ValueError, match="Specify only one"):
        _ = _resolve_train_config_path(args)


def test_apply_overrides_rejects_conflicting_cuda_flags() -> None:
    args = Namespace(
        epochs=None,
        seed=None,
        learning_rate=None,
        energy_cap=None,
        teacher_type=None,
        hb_loss_weight=None,
        preserve_lb_weight=None,
        subtract_loss_weight=None,
        cap_strict_loss_weight=None,
        edge_loss_weight=None,
        step_loss_weight=None,
        device=None,
        no_amp=False,
        require_cuda=True,
        allow_cpu=True,
    )
    with pytest.raises(ValueError, match="Specify only one"):
        _ = _apply_overrides(TrainingConfig(), args)


def test_apply_overrides_updates_seed() -> None:
    args = Namespace(
        epochs=None,
        seed=1234,
        learning_rate=None,
        energy_cap=None,
        teacher_type=None,
        hb_loss_weight=None,
        preserve_lb_weight=None,
        subtract_loss_weight=None,
        cap_strict_loss_weight=None,
        edge_loss_weight=None,
        step_loss_weight=None,
        device=None,
        no_amp=False,
        require_cuda=False,
        allow_cpu=False,
    )
    updated = _apply_overrides(TrainingConfig(seed=1), args)
    assert updated.seed == 1234


def test_apply_overrides_updates_teacher_policy_fields() -> None:
    args = Namespace(
        epochs=None,
        seed=None,
        learning_rate=None,
        energy_cap=None,
        teacher_type="bessel_88k2",
        hb_loss_weight=1.25,
        preserve_lb_weight=1.5,
        subtract_loss_weight=0.9,
        cap_strict_loss_weight=3.5,
        edge_loss_weight=0.12,
        step_loss_weight=0.08,
        device=None,
        no_amp=False,
        require_cuda=False,
        allow_cpu=False,
    )
    updated = _apply_overrides(TrainingConfig(), args)
    assert updated.teacher_type == "bessel_88k2"
    assert updated.hb_loss_weight == pytest.approx(1.25)
    assert updated.preserve_lb_weight == pytest.approx(1.5)
    assert updated.loss_weights.mask == pytest.approx(1.25)
    assert updated.loss_weights.stft == pytest.approx(1.25)
    assert updated.loss_weights.preserve == pytest.approx(1.5)
    assert updated.loss_weights.subtract == pytest.approx(0.9)
    assert updated.loss_weights.cap_strict == pytest.approx(3.5)
    assert updated.loss_weights.edge == pytest.approx(0.12)
    assert updated.loss_weights.step == pytest.approx(0.08)
