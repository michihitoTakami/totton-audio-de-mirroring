"""Train Stage 1 NMSE with composite losses."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from totton_audio_de_mirroring.data.dataloader import (
    DataLoaderConfig,
    collate_samples,
)
from totton_audio_de_mirroring.data.dataset import MirrorSuppressionDataset
from totton_audio_de_mirroring.data.filters import design_band_split_filters
from totton_audio_de_mirroring.data.pipeline_config import (
    DataPipelineConfig,
    load_data_config,
)
from totton_audio_de_mirroring.models.nmse import NMSE
from totton_audio_de_mirroring.training.trainer import (
    TrainingConfig,
    TrainingResult,
    load_training_config,
    train_stage1,
)


def main() -> None:
    """Entry point for Stage 1 training.

    Physical Basis:
        Training uses CUDA to handle high-rate STFT masking and mirror
        suppression at practical throughput.
    """
    args = _parse_args()
    data_config = _load_data_config(args.data_config)
    training_config = _load_training_config(_resolve_train_config_path(args))
    training_config = _apply_overrides(training_config, args)

    train_loader, val_loader = _create_train_val_loaders(
        data_config=data_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
        seed=training_config.seed,
    )

    lowpass, highpass = design_band_split_filters(
        cutoff_hz=data_config.band_split.cutoff_hz,
        sample_rate=data_config.band_split.sample_rate,
        num_taps=data_config.band_split.num_taps,
        window=data_config.band_split.window,
    )

    nmse = NMSE(
        sample_rate=data_config.target_sample_rate,
        cutoff_hz=data_config.band_split.cutoff_hz,
        stft_config=None,
        energy_cap=training_config.energy_cap,
        envelope_floor=data_config.hb_target.envelope_min,
        lowpass_taps=lowpass,
        highpass_taps=highpass,
    )

    result = train_stage1(
        model=nmse,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=training_config,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
    )
    _log_result(result)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Physical Basis:
        Exposing training parameters via CLI enables controlled sweeps
        without changing model code.
    """
    parser = argparse.ArgumentParser(description="Train Stage 1 NMSE")
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data_generation.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Alias for --train-config (README compatibility).",
    )
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--energy-cap", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/checkpoints"))
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _resolve_train_config_path(args: argparse.Namespace) -> Path | None:
    """Resolve training config path from CLI options.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Selected training config path or None.

    Raises:
        ValueError: If both --config and --train-config are specified.

    Physical Basis:
        A single source of truth for loss and optimizer settings prevents
        ambiguous training behavior.
    """
    if args.config is not None and args.train_config is not None:
        raise ValueError("Specify only one of --config or --train-config.")
    if args.train_config is not None:
        return args.train_config
    return args.config


def _load_data_config(path: Path) -> DataPipelineConfig:
    """Load dataset configuration from disk.

    Args:
        path: Path to data config.

    Returns:
        Parsed DataPipelineConfig.

    Physical Basis:
        Data pipeline config defines the synthetic degradations that
        determine mirror artifact distribution during training.
    """
    if not isinstance(path, Path):
        raise ValueError("data_config must be a Path.")
    try:
        return load_data_config(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load data config: {exc}") from exc


def _load_training_config(path: Path | None) -> TrainingConfig:
    """Load training configuration from disk.

    Args:
        path: Optional path to training config.

    Returns:
        Parsed TrainingConfig.

    Physical Basis:
        Training config governs loss balance and optimization stability.
    """
    if path is None:
        return TrainingConfig()
    try:
        return load_training_config(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load training config: {exc}") from exc


def _apply_overrides(
    config: TrainingConfig,
    args: argparse.Namespace,
) -> TrainingConfig:
    """Apply CLI overrides to TrainingConfig.

    Args:
        config: Base training configuration.
        args: Parsed CLI arguments.

    Returns:
        Updated TrainingConfig.

    Physical Basis:
        Override hooks allow rapid tuning of energy caps and learning rates
        without editing configuration files.
    """
    updated = config
    if args.epochs is not None:
        updated = replace(updated, epochs=args.epochs)
    if args.learning_rate is not None:
        updated = replace(updated, learning_rate=args.learning_rate)
    if args.energy_cap is not None:
        updated = replace(updated, energy_cap=args.energy_cap)
    if args.device is not None:
        updated = replace(updated, device=args.device)
    if args.no_amp:
        updated = replace(updated, use_amp=False)
    if args.require_cuda:
        updated = replace(updated, require_cuda=True)
    if args.allow_cpu:
        updated = replace(updated, require_cuda=False)
    return updated


def _create_train_val_loaders(
    *,
    data_config: DataPipelineConfig,
    batch_size: int,
    num_workers: int,
    validation_split: float,
    seed: int | None,
) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]] | None]:
    """Create train/validation dataloaders from a single dataset.

    Args:
        data_config: Data pipeline configuration.
        batch_size: Batch size.
        num_workers: DataLoader worker count.
        validation_split: Validation split ratio in [0, 1).
        seed: Optional random seed.

    Returns:
        Tuple of (train_loader, val_loader).

    Physical Basis:
        Validation split quantifies suppression generalization while
        preserving consistent synthetic data generation.
    """
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0.0, 1.0).")

    dataset = MirrorSuppressionDataset(data_config)
    total_size = len(dataset)
    if total_size < 2:
        raise ValueError("dataset must contain at least 2 samples for training.")

    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    if train_size <= 0:
        raise ValueError("validation_split is too large for dataset size.")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    if val_size > 0:
        train_set, val_set = random_split(dataset, [train_size, val_size], generator)
    else:
        train_set = dataset
        val_set = None

    loader_config = DataLoaderConfig(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    train_loader: DataLoader[dict[str, object]] = DataLoader(
        train_set,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        num_workers=loader_config.num_workers,
        pin_memory=loader_config.pin_memory,
        drop_last=loader_config.drop_last,
        persistent_workers=loader_config.num_workers > 0,
        collate_fn=collate_samples,
    )

    val_loader: DataLoader[dict[str, object]] | None = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=loader_config.batch_size,
            shuffle=False,
            num_workers=loader_config.num_workers,
            pin_memory=loader_config.pin_memory,
            drop_last=False,
            persistent_workers=loader_config.num_workers > 0,
            collate_fn=collate_samples,
        )

    return train_loader, val_loader


def _log_result(result: TrainingResult) -> None:
    """Print final training summary."""
    print(
        f"training_completed device={result.device} best_val_total={result.best_val_total:.6f}"
    )
    if result.last_checkpoint is not None:
        print(f"last_checkpoint={result.last_checkpoint}")
    if result.best_checkpoint is not None:
        print(f"best_checkpoint={result.best_checkpoint}")


if __name__ == "__main__":
    main()
