"""DataLoader helpers for mirror suppression datasets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from totton_audio_de_mirroring.data.dataset import (
    DataPipelineConfig,
    MirrorSuppressionDataset,
)


@dataclass(frozen=True)
class DataLoaderConfig:
    """Configuration for PyTorch DataLoader construction.

    Args:
        batch_size: Batch size.
        shuffle: Whether to shuffle samples.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for faster transfers.
        drop_last: Whether to drop incomplete last batch.

    Physical Basis:
        Stable batching enables consistent training throughput and
        deterministic coverage of synthetic samples.
    """

    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = True

    def __post_init__(self) -> None:
        _validate_positive_int(self.batch_size, "batch_size")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")


def create_dataloader(
    dataset_config: DataPipelineConfig,
    loader_config: DataLoaderConfig | None = None,
) -> DataLoader[dict[str, Any]]:
    """Create a DataLoader for mirror suppression data.

    Args:
        dataset_config: Dataset configuration.
        loader_config: DataLoader configuration.

    Returns:
        Configured DataLoader instance.

    Physical Basis:
        Batching and worker parallelism maintain efficient throughput
        without altering the generated signal content.
    """
    if not isinstance(dataset_config, DataPipelineConfig):
        raise ValueError("dataset_config must be a DataPipelineConfig")

    config = loader_config or DataLoaderConfig()
    dataset = MirrorSuppressionDataset(dataset_config)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_samples,
    )


def collate_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate dataset samples into a batch.

    Args:
        samples: Sequence of sample dictionaries.

    Returns:
        Batched dictionary with stacked tensors and metadata lists.

    Physical Basis:
        Collation keeps tensor alignment across the pipeline outputs.
    """
    if not samples:
        raise ValueError("samples must be non-empty")

    tensor_keys = {
        "source",
        "x_full",
        "low_band",
        "high_band",
        "hb_target",
    }

    batch: dict[str, Any] = {}
    for key in tensor_keys:
        batch[key] = torch.stack([sample[key] for sample in samples])

    batch["profile"] = [sample["profile"] for sample in samples]
    batch["signal_type"] = [sample["signal_type"] for sample in samples]
    batch["chunk_start"] = torch.tensor(
        [sample["chunk_start"] for sample in samples], dtype=torch.int64
    )

    return batch


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
