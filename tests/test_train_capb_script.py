"""Tests for the CAPB training command helpers."""

from pathlib import Path

import pytest
from scripts.train_capb import _override_seed

from totton_audio_de_mirroring.data.capb_dataset import CAPBDataConfig
from totton_audio_de_mirroring.training.capb_trainer import CAPBTrainingConfig


def test_seed_override_updates_data_and_training() -> None:
    data = CAPBDataConfig(seed=1)
    training = CAPBTrainingConfig(seed=2, checkpoint_dir=Path("checkpoints"))

    updated_data, updated_training = _override_seed(data, training, 2234)

    assert updated_data.seed == 2234
    assert updated_training.seed == 2234
    assert data.seed == 1
    assert training.seed == 2


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_override_rejects_out_of_range(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        _override_seed(CAPBDataConfig(), CAPBTrainingConfig(), seed)
