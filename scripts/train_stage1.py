"""Train Stage 1 NMSE with composite losses."""

from __future__ import annotations

import argparse
import gc
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

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
    LossWeights,
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
    training_config = _load_training_config(
        _resolve_train_config_path(args),
        default_teacher_type=data_config.teacher_type,
    )
    training_config = _apply_overrides(training_config, args)
    data_config, training_config = _synchronize_teacher_type(
        data_config=data_config,
        training_config=training_config,
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

    if args.auto_batch_size and args.resume_from is not None:
        raise ValueError("--auto-batch-size cannot be used with --resume-from.")

    if args.auto_batch_size:
        result = _train_with_adaptive_batch_size(
            model=nmse,
            data_config=data_config,
            training_config=training_config,
            initial_batch_size=args.batch_size,
            min_batch_size=args.min_batch_size,
            max_oom_retries=args.max_oom_retries,
            num_workers=args.num_workers,
            validation_split=args.validation_split,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        train_loader, val_loader = _create_train_val_loaders(
            data_config=data_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            validation_split=args.validation_split,
            seed=training_config.seed,
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--energy-cap", type=float, default=None)
    parser.add_argument(
        "--teacher-type",
        type=str,
        choices=["raw_88k2", "bessel_88k2", "raw_176k4", "bessel_176k4"],
        default=None,
    )
    parser.add_argument("--hb-loss-weight", type=float, default=None)
    parser.add_argument("--preserve-lb-weight", type=float, default=None)
    parser.add_argument("--subtract-loss-weight", type=float, default=None)
    parser.add_argument("--cap-strict-loss-weight", type=float, default=None)
    parser.add_argument("--edge-loss-weight", type=float, default=None)
    parser.add_argument("--step-loss-weight", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/checkpoints"))
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--auto-batch-size",
        action="store_true",
        help="Retry training with smaller batch size on CUDA out-of-memory.",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=1,
        help="Minimum batch size when using --auto-batch-size.",
    )
    parser.add_argument(
        "--max-oom-retries",
        type=int,
        default=5,
        help="Maximum number of CUDA OOM retries when using --auto-batch-size.",
    )
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
    train_config = cast(Path | None, args.train_config)
    config_alias = cast(Path | None, args.config)
    if train_config is not None:
        return train_config
    return config_alias


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


def _load_training_config(
    path: Path | None,
    *,
    default_teacher_type: str,
) -> TrainingConfig:
    """Load training configuration from disk.

    Args:
        path: Optional path to training config.

    Returns:
        Parsed TrainingConfig.

    Physical Basis:
        Training config governs loss balance and optimization stability.
    """
    if path is None:
        return TrainingConfig.from_dict({}, default_teacher_type=default_teacher_type)
    try:
        return load_training_config(path, default_teacher_type=default_teacher_type)
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
    if args.require_cuda and args.allow_cpu:
        raise ValueError("Specify only one of --require-cuda or --allow-cpu.")

    updated = config
    if args.epochs is not None:
        updated = replace(updated, epochs=args.epochs)
    if args.seed is not None:
        updated = replace(updated, seed=args.seed)
    if args.learning_rate is not None:
        updated = replace(updated, learning_rate=args.learning_rate)
    if args.energy_cap is not None:
        updated = replace(updated, energy_cap=args.energy_cap)
    if args.teacher_type is not None:
        updated = replace(updated, teacher_type=args.teacher_type)
    if args.hb_loss_weight is not None:
        updated = replace(
            updated,
            hb_loss_weight=args.hb_loss_weight,
            loss_weights=LossWeights(
                mask=args.hb_loss_weight,
                stft=args.hb_loss_weight,
                preserve=updated.loss_weights.preserve,
                energy=updated.loss_weights.energy,
                subtract=updated.loss_weights.subtract,
                cap_strict=updated.loss_weights.cap_strict,
                edge=updated.loss_weights.edge,
                step=updated.loss_weights.step,
            ),
        )
    if args.preserve_lb_weight is not None:
        updated = replace(
            updated,
            preserve_lb_weight=args.preserve_lb_weight,
            loss_weights=LossWeights(
                mask=updated.loss_weights.mask,
                stft=updated.loss_weights.stft,
                preserve=args.preserve_lb_weight,
                energy=updated.loss_weights.energy,
                subtract=updated.loss_weights.subtract,
                cap_strict=updated.loss_weights.cap_strict,
                edge=updated.loss_weights.edge,
                step=updated.loss_weights.step,
            ),
        )
    if args.subtract_loss_weight is not None:
        updated = replace(
            updated,
            loss_weights=replace(
                updated.loss_weights,
                subtract=float(args.subtract_loss_weight),
            ),
        )
    if args.cap_strict_loss_weight is not None:
        updated = replace(
            updated,
            loss_weights=replace(
                updated.loss_weights,
                cap_strict=float(args.cap_strict_loss_weight),
            ),
        )
    if args.edge_loss_weight is not None:
        updated = replace(
            updated,
            loss_weights=replace(
                updated.loss_weights,
                edge=float(args.edge_loss_weight),
            ),
        )
    if args.step_loss_weight is not None:
        updated = replace(
            updated,
            loss_weights=replace(
                updated.loss_weights,
                step=float(args.step_loss_weight),
            ),
        )
    if args.device is not None:
        updated = replace(updated, device=args.device)
    if args.no_amp:
        updated = replace(updated, use_amp=False)
    if args.require_cuda:
        updated = replace(updated, require_cuda=True)
    if args.allow_cpu:
        updated = replace(updated, require_cuda=False)
    return updated


def _synchronize_teacher_type(
    *,
    data_config: DataPipelineConfig,
    training_config: TrainingConfig,
) -> tuple[DataPipelineConfig, TrainingConfig]:
    """Synchronize teacher policy between data and training configs.

    Physical Basis:
        Dataset target generation and loss-policy defaults must share
        the same teacher definition to avoid mixed-policy training runs.
    """
    if training_config.teacher_type == data_config.teacher_type:
        return data_config, training_config
    return (
        replace(data_config, teacher_type=training_config.teacher_type),
        training_config,
    )


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

    train_set: Dataset[dict[str, Any]]
    val_set: Dataset[dict[str, Any]] | None
    if val_size > 0:
        split_train, split_val = random_split(
            dataset,
            [train_size, val_size],
            generator,
        )
        train_set = cast(Dataset[dict[str, Any]], split_train)
        val_set = cast(Dataset[dict[str, Any]], split_val)
    else:
        train_set = cast(Dataset[dict[str, Any]], dataset)
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


def _train_with_adaptive_batch_size(
    *,
    model: nn.Module,
    data_config: DataPipelineConfig,
    training_config: TrainingConfig,
    initial_batch_size: int,
    min_batch_size: int,
    max_oom_retries: int,
    num_workers: int,
    validation_split: float,
    checkpoint_dir: Path,
) -> TrainingResult:
    """Train with batch-size backoff on CUDA OOM.

    Physical Basis:
        Batch size dominates activation memory. Reducing batch size preserves
        model/loss definitions while adapting memory footprint to available VRAM.
    """
    if initial_batch_size <= 0:
        raise ValueError("initial_batch_size must be positive.")
    if min_batch_size <= 0:
        raise ValueError("min_batch_size must be positive.")
    if min_batch_size > initial_batch_size:
        raise ValueError("min_batch_size must be <= initial_batch_size.")
    if max_oom_retries < 0:
        raise ValueError("max_oom_retries must be non-negative.")

    batch_size = initial_batch_size
    retries = 0

    while True:
        attempt_checkpoint_dir = checkpoint_dir / f"bs{batch_size}"
        attempt_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _log_cuda_memory(prefix=f"attempt batch_size={batch_size}")

        train_loader, val_loader = _create_train_val_loaders(
            data_config=data_config,
            batch_size=batch_size,
            num_workers=num_workers,
            validation_split=validation_split,
            seed=training_config.seed,
        )
        try:
            return train_stage1(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                config=training_config,
                checkpoint_dir=attempt_checkpoint_dir,
                resume_from=None,
            )
        except Exception as exc:
            if not _is_cuda_oom_error(exc):
                raise
            if batch_size <= min_batch_size or retries >= max_oom_retries:
                raise RuntimeError(
                    "CUDA OOM persists after adaptive batch-size retries."
                ) from exc

            retries += 1
            next_batch_size = max(min_batch_size, batch_size // 2)
            print(
                "cuda_oom_detected "
                f"retry={retries}/{max_oom_retries} "
                f"batch_size={batch_size}->{next_batch_size}",
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_size = next_batch_size


def _is_cuda_oom_error(exc: BaseException) -> bool:
    """Return True if exception chain indicates CUDA out-of-memory."""
    current: BaseException | None = exc
    while current is not None:
        if "cuda out of memory" in str(current).lower():
            return True
        current = current.__cause__
    return False


def _log_cuda_memory(prefix: str) -> None:
    """Log current free/total CUDA memory when available."""
    if not torch.cuda.is_available():
        return
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return
    free_gb = free_bytes / (1024.0**3)
    total_gb = total_bytes / (1024.0**3)
    print(
        f"cuda_mem {prefix} free_gb={free_gb:.3f} total_gb={total_gb:.3f}",
        flush=True,
    )


def _log_result(result: TrainingResult) -> None:
    """Print final training summary."""
    print(
        f"training_completed device={result.device} best_val_total={result.best_val_total:.6f}",
        flush=True,
    )
    if result.last_checkpoint is not None:
        print(f"last_checkpoint={result.last_checkpoint}", flush=True)
    if result.best_checkpoint is not None:
        print(f"best_checkpoint={result.best_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
