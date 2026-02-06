"""Tests for Stage 1 training script helpers."""

from argparse import Namespace
from pathlib import Path

import pytest
from scripts.train_stage1 import _resolve_train_config_path


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
