"""Tests for training utilities."""

import textwrap
from pathlib import Path

import pytest

from totton_audio_de_mirroring.training.trainer import (
    TrainingConfig,
    load_training_config,
    select_device,
)


def test_select_device_override_cpu() -> None:
    device = select_device(device_override="cpu")
    assert device.type == "cpu"


def test_training_config_from_dict_parses_weights() -> None:
    config = TrainingConfig.from_dict(
        {
            "epochs": 2,
            "learning_rate": 1.0e-3,
            "loss_weights": {
                "mask": 2.0,
                "stft": 0.5,
                "preserve": 1.5,
                "energy": 0.2,
            },
        }
    )
    assert config.epochs == 2
    assert config.loss_weights.mask == 2.0
    assert config.loss_weights.stft == 0.5


def test_load_training_config_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            epochs: 3
            learning_rate: 0.001
            loss_weights:
              mask: 1.0
              stft: 1.0
              preserve: 1.0
              energy: 1.0
            """
        ).strip(),
        encoding="utf-8",
    )
    config = load_training_config(config_path)
    assert config.epochs == 3
    assert config.learning_rate == pytest.approx(0.001)
