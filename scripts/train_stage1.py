"""Train Stage 1 NMSE with composite losses."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from totton_audio_de_mirroring.data.dataloader import (
    DataLoaderConfig,
    create_dataloader,
)
from totton_audio_de_mirroring.data.filters import design_band_split_filters
from totton_audio_de_mirroring.data.pipeline_config import (
    DataPipelineConfig,
    load_data_config,
)
from totton_audio_de_mirroring.models.nmse import NMSE
from totton_audio_de_mirroring.training.trainer import (
    TrainingConfig,
    load_training_config,
    train_stage1,
)


def main() -> None:
    """Entry point for Stage 1 training.

    Physical Basis:
        Training uses CUDA when available to handle high-rate STFT masking
        and mirror suppression at practical throughput.
    """
    args = _parse_args()
    data_config = _load_data_config(args.data_config)
    training_config = _load_training_config(_resolve_train_config_path(args))
    training_config = _apply_overrides(training_config, args)

    loader_config = DataLoaderConfig(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    dataloader = create_dataloader(data_config, loader_config)

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

    losses = train_stage1(nmse, dataloader, training_config)
    _save_checkpoint(nmse, training_config, args.checkpoint_dir, len(losses))


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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--energy-cap", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/checkpoints"))
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
    config: TrainingConfig, args: argparse.Namespace
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
    return updated


def _save_checkpoint(
    model: NMSE,
    config: TrainingConfig,
    checkpoint_dir: Path,
    steps: int,
) -> None:
    """Save a training checkpoint to disk.

    Args:
        model: Trained NMSE model.
        config: Training configuration.
        checkpoint_dir: Directory for checkpoint output.
        steps: Total optimization steps.

    Physical Basis:
        Persisting checkpoints preserves learned mirror suppression
        behavior for later evaluation and inference.
    """
    if not isinstance(checkpoint_dir, Path):
        raise ValueError("checkpoint_dir must be a Path.")
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state": model.state_dict(),
            "training_config": config,
            "steps": steps,
        }
        torch.save(state, checkpoint_dir / "stage1_last.pt")
    except Exception as exc:
        raise RuntimeError(f"Failed to save checkpoint: {exc}") from exc


if __name__ == "__main__":
    main()
