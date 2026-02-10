"""Train lightweight Stage 1 NMSE via teacher-student distillation."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from totton_audio_de_mirroring.data.dataloader import DataLoaderConfig, collate_samples
from totton_audio_de_mirroring.data.dataset import MirrorSuppressionDataset
from totton_audio_de_mirroring.data.filters import design_band_split_filters
from totton_audio_de_mirroring.data.pipeline_config import (
    DataPipelineConfig,
    load_data_config,
)
from totton_audio_de_mirroring.inference.pipeline import load_nmse_stage1_processor
from totton_audio_de_mirroring.models.nmse import NMSE
from totton_audio_de_mirroring.models.nmse_light import NMSELight, NMSELightConfig
from totton_audio_de_mirroring.models.unet import UNet2D
from totton_audio_de_mirroring.training.distillation import (
    DistillationConfig,
    apply_global_magnitude_pruning,
    count_parameters,
    load_distillation_config,
    train_stage1_distillation,
)


def main() -> None:
    """Run Stage 1 distillation training entrypoint.

    Physical Basis:
        Distillation keeps Stage 1 objective alignment while reducing
        student model complexity for deployment throughput.
    """
    args = _parse_args()
    data_config = _load_data_config(args.data_config)
    train_config = _load_train_config(
        args.train_config,
        default_teacher_type=data_config.teacher_type,
    )
    train_config = _apply_overrides(train_config, args)
    data_config, train_config = _synchronize_teacher_type(
        data_config=data_config,
        train_config=train_config,
    )
    _validate_teacher_checkpoint_teacher_type(
        checkpoint_path=args.teacher_checkpoint,
        expected_teacher_type=train_config.teacher_type,
    )
    teacher_tag = _teacher_tag(train_config.teacher_type)
    checkpoint_prefix = f"stage1_distill_{teacher_tag}"

    train_loader, val_loader = _create_train_val_loaders(
        data_config=data_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
        seed=train_config.seed,
    )
    teacher = _load_teacher_model(
        checkpoint_path=args.teacher_checkpoint,
        data_config_path=args.data_config,
        device="cpu",
    )
    student, model_config = _build_student_model(
        model_name=args.student_model,
        data_config=data_config,
        energy_cap=train_config.energy_cap,
        base_channels=args.base_channels,
        num_downsamples=args.num_downsamples,
        channel_multiplier=args.channel_multiplier,
    )
    if args.pruning_ratio > 0.0:
        apply_global_magnitude_pruning(student, amount=args.pruning_ratio)

    print(
        f"teacher_params={count_parameters(teacher)} "
        f"student_params={count_parameters(student)}",
        flush=True,
    )

    result = train_stage1_distillation(
        teacher=teacher,
        student=student,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=train_config,
        checkpoint_dir=args.checkpoint_dir,
        model_config=model_config,
        checkpoint_prefix=checkpoint_prefix,
    )
    print(f"training_completed device={result.device}", flush=True)
    if result.last_checkpoint is not None:
        print(f"last_checkpoint={result.last_checkpoint}", flush=True)
    if result.best_checkpoint is not None:
        print(f"best_checkpoint={result.best_checkpoint}", flush=True)
        stage1_light_path = _emit_stage1_light_checkpoint(
            best_checkpoint=result.best_checkpoint,
            checkpoint_dir=args.checkpoint_dir,
            teacher_type=train_config.teacher_type,
        )
        print(f"stage1_light_checkpoint={stage1_light_path}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1 distillation student.")
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data_generation.yaml"),
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/training_distillation_stage1.yaml"),
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("data/checkpoints/stage1_best.pt"),
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="nmse_light",
        choices=["nmse_light", "nmse"],
    )
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--num-downsamples", type=int, default=None)
    parser.add_argument("--channel-multiplier", type=int, default=None)
    parser.add_argument("--pruning-ratio", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--energy-cap", type=float, default=None)
    parser.add_argument(
        "--teacher-type",
        type=str,
        choices=["raw_88k2", "bessel_88k2"],
        default=None,
    )
    parser.add_argument("--hb-loss-weight", type=float, default=None)
    parser.add_argument("--preserve-lb-weight", type=float, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/checkpoints/distillation"),
    )
    return parser.parse_args()


def _load_data_config(path: Path) -> DataPipelineConfig:
    if not isinstance(path, Path):
        raise ValueError("data_config must be a pathlib.Path.")
    try:
        return load_data_config(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load data config: {exc}") from exc


def _emit_stage1_light_checkpoint(
    *,
    best_checkpoint: Path,
    checkpoint_dir: Path,
    teacher_type: str,
) -> Path:
    """Copy best distillation checkpoint to teacher-aware stage1_light naming.

    Physical Basis:
        Teacher-scoped naming prevents accidental mixing of raw/bessel
        distillation artifacts in A/B evaluations.
    """
    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint}")
    teacher_tag = _teacher_tag(teacher_type)
    stage1_light_path = checkpoint_dir / f"stage1_light_{teacher_tag}.pt"
    shutil.copy2(best_checkpoint, stage1_light_path)
    # Keep legacy alias for backward compatibility with existing scripts.
    shutil.copy2(best_checkpoint, checkpoint_dir / "stage1_light.pt")
    return stage1_light_path


def _load_train_config(
    path: Path,
    *,
    default_teacher_type: str,
) -> DistillationConfig:
    if not isinstance(path, Path):
        raise ValueError("train_config must be a pathlib.Path.")
    try:
        return load_distillation_config(
            path,
            default_teacher_type=default_teacher_type,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load distillation config: {exc}") from exc


def _apply_overrides(
    config: DistillationConfig,
    args: argparse.Namespace,
) -> DistillationConfig:
    if args.require_cuda and args.allow_cpu:
        raise ValueError("Specify only one of --require-cuda or --allow-cpu.")
    if args.pruning_ratio < 0.0 or args.pruning_ratio >= 1.0:
        raise ValueError("pruning_ratio must be in [0.0, 1.0).")
    updated = config
    if args.epochs is not None:
        updated = replace(updated, epochs=args.epochs)
    if args.learning_rate is not None:
        updated = replace(updated, learning_rate=args.learning_rate)
    if args.seed is not None:
        updated = replace(updated, seed=args.seed)
    if args.device is not None:
        updated = replace(updated, device=args.device)
    if args.energy_cap is not None:
        updated = replace(updated, energy_cap=args.energy_cap)
    if args.teacher_type is not None:
        updated = replace(updated, teacher_type=args.teacher_type)
    if args.hb_loss_weight is not None:
        updated = replace(updated, hb_loss_weight=args.hb_loss_weight)
    if args.preserve_lb_weight is not None:
        updated = replace(updated, preserve_lb_weight=args.preserve_lb_weight)
    if args.require_cuda:
        updated = replace(updated, require_cuda=True)
    if args.allow_cpu:
        updated = replace(updated, require_cuda=False)
    return updated


def _synchronize_teacher_type(
    *,
    data_config: DataPipelineConfig,
    train_config: DistillationConfig,
) -> tuple[DataPipelineConfig, DistillationConfig]:
    """Synchronize teacher policy between data and distillation configs.

    Physical Basis:
        Teacher reference generation and distillation policy defaults
        must stay aligned to avoid mixed-target optimization.
    """
    if train_config.teacher_type == data_config.teacher_type:
        return data_config, train_config
    return (
        replace(data_config, teacher_type=train_config.teacher_type),
        train_config,
    )


def _create_train_val_loaders(
    *,
    data_config: DataPipelineConfig,
    batch_size: int,
    num_workers: int,
    validation_split: float,
    seed: int | None,
) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]] | None]:
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


def _load_teacher_model(
    *,
    checkpoint_path: Path,
    data_config_path: Path,
    device: str,
) -> nn.Module:
    processor = load_nmse_stage1_processor(
        checkpoint_path=checkpoint_path,
        data_config_path=data_config_path,
        device=device,
    )
    model = processor.model
    model.eval()
    return model


def _validate_teacher_checkpoint_teacher_type(
    *,
    checkpoint_path: Path,
    expected_teacher_type: str,
) -> None:
    """Validate teacher checkpoint policy metadata against training policy.

    Physical Basis:
        Distillation requires teacher policy consistency; mixing raw and
        bessel teachers invalidates matched-condition comparisons.
    """
    checkpoint_teacher_type = _load_teacher_type_from_checkpoint(checkpoint_path)
    if checkpoint_teacher_type != expected_teacher_type:
        raise RuntimeError(
            "Teacher checkpoint type mismatch: "
            f"expected {expected_teacher_type!r}, got {checkpoint_teacher_type!r} "
            f"from {checkpoint_path}."
        )


def _load_teacher_type_from_checkpoint(checkpoint_path: Path) -> str:
    """Load teacher_type metadata from teacher checkpoint.

    Physical Basis:
        Training config metadata encodes the teacher policy used for
        high-band suppression targets and must be preserved for provenance.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load teacher checkpoint metadata: {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Invalid teacher checkpoint root type: {checkpoint_path}: {type(checkpoint)!r}."
        )
    training_config = checkpoint.get("training_config")
    if not isinstance(training_config, dict):
        raise RuntimeError(
            f"Missing training_config in teacher checkpoint: {checkpoint_path}."
        )
    teacher_type = training_config.get("teacher_type")
    if not isinstance(teacher_type, str):
        raise RuntimeError(
            f"Missing training_config.teacher_type in teacher checkpoint: {checkpoint_path}."
        )
    return teacher_type


def _teacher_tag(teacher_type: str) -> str:
    """Convert teacher type into artifact-friendly short tag.

    Physical Basis:
        Stable short tags prevent naming collisions between teacher policies
        across baseline/distillation runs.
    """
    if teacher_type == "raw_88k2":
        return "raw88"
    if teacher_type == "bessel_88k2":
        return "bessel"
    raise ValueError(f"Unsupported teacher_type: {teacher_type!r}.")


def _build_student_model(
    *,
    model_name: str,
    data_config: DataPipelineConfig,
    energy_cap: float,
    base_channels: int | None,
    num_downsamples: int | None,
    channel_multiplier: int | None,
) -> tuple[nn.Module, dict[str, Any]]:
    lowpass, highpass = design_band_split_filters(
        cutoff_hz=data_config.band_split.cutoff_hz,
        sample_rate=data_config.band_split.sample_rate,
        num_taps=data_config.band_split.num_taps,
        window=data_config.band_split.window,
    )
    if model_name == "nmse_light":
        config = NMSELightConfig(
            base_channels=base_channels if base_channels is not None else 40,
            num_downsamples=num_downsamples if num_downsamples is not None else 3,
            channel_multiplier=(
                channel_multiplier if channel_multiplier is not None else 2
            ),
        )
        model: nn.Module = NMSELight(
            sample_rate=data_config.target_sample_rate,
            cutoff_hz=data_config.band_split.cutoff_hz,
            energy_cap=energy_cap,
            envelope_floor=data_config.hb_target.envelope_min,
            lowpass_taps=lowpass,
            highpass_taps=highpass,
            model_config=config,
        )
        return model, config.to_checkpoint_dict()

    nmse_base_channels = base_channels if base_channels is not None else 32
    nmse_num_downsamples = num_downsamples if num_downsamples is not None else 4
    nmse_channel_multiplier = (
        channel_multiplier if channel_multiplier is not None else 2
    )
    unet = UNet2D(
        base_channels=nmse_base_channels,
        num_downsamples=nmse_num_downsamples,
        channel_multiplier=nmse_channel_multiplier,
    )
    model = NMSE(
        sample_rate=data_config.target_sample_rate,
        cutoff_hz=data_config.band_split.cutoff_hz,
        stft_config=None,
        unet=unet,
        energy_cap=energy_cap,
        envelope_floor=data_config.hb_target.envelope_min,
        lowpass_taps=lowpass,
        highpass_taps=highpass,
    )
    return model, {
        "model_type": "nmse",
        "base_channels": nmse_base_channels,
        "num_downsamples": nmse_num_downsamples,
        "channel_multiplier": nmse_channel_multiplier,
        "activation": "leaky_relu",
        "use_batch_norm": True,
        "output_activation": "sigmoid",
    }


if __name__ == "__main__":
    main()
