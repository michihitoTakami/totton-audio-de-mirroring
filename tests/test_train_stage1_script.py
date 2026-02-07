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
        device=None,
        no_amp=False,
        require_cuda=False,
        allow_cpu=False,
    )
    updated = _apply_overrides(TrainingConfig(seed=1), args)
    assert updated.seed == 1234
