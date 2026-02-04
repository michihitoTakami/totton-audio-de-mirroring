import json
from pathlib import Path

import numpy as np
import torch

from totton_audio_de_mirroring.data.dataloader import (
    DataLoaderConfig,
    create_dataloader,
)
from totton_audio_de_mirroring.data.pipeline_config import (
    AugmentationConfig,
    DataPipelineConfig,
    SignalSamplingConfig,
    load_data_config,
    save_data_config,
)
from totton_audio_de_mirroring.models.band_split import BandSplitConfig


def _small_config() -> DataPipelineConfig:
    return DataPipelineConfig(
        num_samples=4,
        source_duration_sec=0.5,
        chunk_duration_sec=0.25,
        seed=123,
        signal_sampling=SignalSamplingConfig(signal_types=("multitone",)),
        augmentation=AugmentationConfig(
            gain_range=(1.0, 1.0),
            polarity_flip_prob=0.0,
            noise_std_range=(0.0, 0.0),
            soft_clip_prob=0.0,
            soft_clip_drive_range=(1.0, 1.0),
        ),
        band_split=BandSplitConfig(num_taps=513, sample_rate=88_200),
    )


def test_dataset_item_shapes() -> None:
    config = _small_config()
    dataset = create_dataloader(config, DataLoaderConfig(batch_size=1)).dataset

    sample = dataset[0]

    length = int(round(config.chunk_duration_sec * config.target_sample_rate))
    assert sample["x_full"].shape == (length,)
    assert sample["low_band"].shape == (length,)
    assert sample["high_band"].shape == (length,)
    assert sample["hb_target"].shape == (length,)
    assert sample["mirror_mask"].ndim == 2
    assert sample["source"].shape == (
        int(round(config.chunk_duration_sec * config.source_sample_rate)),
    )
    assert isinstance(sample["profile"].method, str)
    assert sample["signal_type"] == "multitone"
    assert sample["x_full"].dtype == torch.float32


def test_dataset_reproducible() -> None:
    config = DataPipelineConfig(
        num_samples=2,
        source_duration_sec=0.5,
        chunk_duration_sec=0.25,
        seed=999,
        signal_sampling=SignalSamplingConfig(signal_types=("multitone",)),
        augmentation=AugmentationConfig(
            gain_range=(1.0, 1.0),
            polarity_flip_prob=0.0,
            noise_std_range=(0.0, 0.0),
            soft_clip_prob=0.0,
            soft_clip_drive_range=(1.0, 1.0),
        ),
        band_split=BandSplitConfig(num_taps=513, sample_rate=88_200),
    )
    dataset_a = create_dataloader(config, DataLoaderConfig(batch_size=1)).dataset
    dataset_b = create_dataloader(config, DataLoaderConfig(batch_size=1)).dataset

    sample_a = dataset_a[0]
    sample_b = dataset_b[0]

    assert torch.allclose(sample_a["x_full"], sample_b["x_full"])
    assert torch.allclose(sample_a["hb_target"], sample_b["hb_target"])


def test_dataloader_batches() -> None:
    config = _small_config()
    loader = create_dataloader(
        config,
        DataLoaderConfig(batch_size=2, shuffle=False, num_workers=0, drop_last=False),
    )

    batch = next(iter(loader))
    assert batch["x_full"].shape[0] == 2
    assert batch["hb_target"].shape[0] == 2
    assert batch["mirror_mask"].shape[0] == 2
    assert batch["chunk_start"].shape == (2,)


def test_config_roundtrip_json_yaml(tmp_path: Path) -> None:
    config = _small_config()

    json_path = tmp_path / "config.json"
    yaml_path = tmp_path / "config.yaml"

    save_data_config(config, json_path)
    save_data_config(config, yaml_path)

    config_json = load_data_config(json_path)
    config_yaml = load_data_config(yaml_path)

    assert config_json.chunk_duration_sec == config.chunk_duration_sec
    assert config_yaml.chunk_duration_sec == config.chunk_duration_sec

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw["num_samples"] == config.num_samples


def test_signal_sampling_config_validation() -> None:
    config = SignalSamplingConfig(signal_types=("white_noise",))
    rng = np.random.default_rng(0)
    request = config.signal_types[rng.integers(0, len(config.signal_types))]
    assert request == "white_noise"


def test_config_bool_coercion() -> None:
    config = DataPipelineConfig.from_dict({"random_chunk": "false"})
    assert config.random_chunk is False
