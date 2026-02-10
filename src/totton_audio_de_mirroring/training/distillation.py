"""Knowledge distillation utilities for Stage 1 model compression."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from totton_audio_de_mirroring.training.losses import (
    LossMode,
    LossWeights,
    RingingLossConfig,
    STFTLossConfig,
    compute_losses,
)
from totton_audio_de_mirroring.training.runtime import ensure_dir, save_checkpoint
from totton_audio_de_mirroring.training.trainer import select_device

TeacherType = Literal["raw_88k2", "bessel_88k2"]
ALLOWED_TEACHER_TYPES: tuple[TeacherType, ...] = ("raw_88k2", "bessel_88k2")


@dataclass(frozen=True)
class DistillationConfig:
    """Configuration for Stage 1 teacher-student distillation.

    Args:
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        grad_clip: Optional gradient clipping norm.
        use_amp: Enable automatic mixed precision on CUDA.
        log_interval: Logging interval in optimization steps.
        mask_config: STFT setup for mask/preserve losses.
        stft_configs: Multi-resolution STFT losses for task supervision.
        task_loss_weights: Loss weights for Stage 1 task objective.
        ringing_loss_config: Configuration for ringing auxiliary losses.
        teacher_type: Stage1 teacher policy (`raw_88k2` or `bessel_88k2`).
        hb_loss_weight: Default weight for high-band objective terms.
        preserve_lb_weight: Default weight for low-band preservation term.
        energy_cap: High-band energy cap used by task loss.
        mask_mode: Loss mode used by mask objective.
        distillation_weight: Weight of teacher-student consistency loss.
        task_weight: Weight of Stage 1 task loss.
        distillation_mode: Distillation criterion ("l1" or "l2").
        device: Optional explicit device override.
        require_cuda: Whether CUDA is required.
        seed: Optional RNG seed.

    Physical Basis:
        Distillation transfers suppression behavior from a larger teacher to
        a compact student while retaining Stage 1 hard constraints.
    """

    epochs: int = 20
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    use_amp: bool = True
    log_interval: int = 20
    mask_config: STFTLossConfig = STFTLossConfig(
        n_fft=2048,
        hop_length=512,
        win_length=2048,
    )
    stft_configs: tuple[STFTLossConfig, ...] = (
        STFTLossConfig(n_fft=1024, hop_length=256, win_length=1024),
        STFTLossConfig(n_fft=2048, hop_length=512, win_length=2048),
    )
    task_loss_weights: LossWeights = LossWeights(
        mask=1.0,
        stft=1.0,
        preserve=1.0,
        energy=1.0,
        subtract=1.0,
        cap_strict=4.0,
        edge=0.05,
        step=0.05,
    )
    ringing_loss_config: RingingLossConfig = RingingLossConfig()
    teacher_type: TeacherType = "raw_88k2"
    hb_loss_weight: float = 1.0
    preserve_lb_weight: float = 1.0
    energy_cap: float = 1.0e-3
    mask_mode: LossMode = "l1"
    distillation_weight: float = 1.0
    task_weight: float = 1.0
    distillation_mode: LossMode = "l2"
    device: str | None = None
    require_cuda: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.grad_clip is not None and self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive when set.")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive.")
        if self.teacher_type not in ALLOWED_TEACHER_TYPES:
            raise ValueError(
                "teacher_type must be one of "
                f"{ALLOWED_TEACHER_TYPES}, got {self.teacher_type!r}."
            )
        if self.hb_loss_weight <= 0.0:
            raise ValueError("hb_loss_weight must be positive.")
        if self.preserve_lb_weight <= 0.0:
            raise ValueError("preserve_lb_weight must be positive.")
        if self.energy_cap <= 0.0:
            raise ValueError("energy_cap must be positive.")
        if self.distillation_weight < 0.0:
            raise ValueError("distillation_weight must be non-negative.")
        if self.task_weight < 0.0:
            raise ValueError("task_weight must be non-negative.")
        if not self.stft_configs:
            raise ValueError("stft_configs must not be empty.")
        if self.distillation_mode not in {"l1", "l2"}:
            raise ValueError("distillation_mode must be 'l1' or 'l2'.")

    @staticmethod
    def from_dict(
        raw: Mapping[str, Any],
        *,
        default_teacher_type: str | None = None,
    ) -> DistillationConfig:
        """Create DistillationConfig from JSON/YAML mapping."""
        if not isinstance(raw, Mapping):
            raise ValueError("raw must be a mapping.")

        teacher_type = _parse_teacher_type(
            raw.get("teacher_type", default_teacher_type or "raw_88k2")
        )
        hb_loss_weight = float(
            raw.get(
                "hb_loss_weight",
                _default_hb_loss_weight_for_teacher(teacher_type),
            )
        )
        preserve_lb_weight = float(
            raw.get(
                "preserve_lb_weight",
                _default_preserve_lb_weight_for_teacher(teacher_type),
            )
        )
        stft_list = raw.get("stft_configs")
        stft_configs = (
            tuple(_parse_stft_config(item) for item in stft_list)
            if isinstance(stft_list, list)
            else (
                STFTLossConfig(n_fft=1024, hop_length=256, win_length=1024),
                STFTLossConfig(n_fft=2048, hop_length=512, win_length=2048),
            )
        )
        return DistillationConfig(
            epochs=int(raw.get("epochs", 20)),
            learning_rate=float(raw.get("learning_rate", 1.0e-4)),
            weight_decay=float(raw.get("weight_decay", 0.0)),
            grad_clip=_optional_float(raw.get("grad_clip", 1.0)),
            use_amp=_parse_bool(raw.get("use_amp", True)),
            log_interval=int(raw.get("log_interval", 20)),
            mask_config=_parse_stft_config(raw.get("mask_config", {})),
            stft_configs=stft_configs,
            task_loss_weights=_parse_loss_weights(
                raw.get("task_loss_weights", {}),
                hb_loss_weight=hb_loss_weight,
                preserve_lb_weight=preserve_lb_weight,
                teacher_type=teacher_type,
            ),
            ringing_loss_config=_parse_ringing_loss_config(
                raw.get("ringing_loss_config", {})
            ),
            teacher_type=teacher_type,
            hb_loss_weight=hb_loss_weight,
            preserve_lb_weight=preserve_lb_weight,
            energy_cap=float(
                raw.get(
                    "energy_cap",
                    _default_energy_cap_for_teacher(teacher_type),
                )
            ),
            mask_mode=_parse_loss_mode(raw.get("mask_mode", "l1")),
            distillation_weight=float(raw.get("distillation_weight", 1.0)),
            task_weight=float(raw.get("task_weight", 1.0)),
            distillation_mode=_parse_loss_mode(raw.get("distillation_mode", "l2")),
            device=_optional_str(raw.get("device")),
            require_cuda=_parse_bool(raw.get("require_cuda", True)),
            seed=_optional_int(raw.get("seed")),
        )


@dataclass(frozen=True)
class DistillationEpochMetrics:
    """Aggregated one-epoch distillation metrics."""

    total: float
    task: float
    distill: float
    samples: int
    steps: int
    throughput_samples_per_sec: float
    throughput_steps_per_sec: float


@dataclass(frozen=True)
class DistillationResult:
    """Result object for distillation training."""

    device: str
    train_history: tuple[DistillationEpochMetrics, ...]
    val_history: tuple[DistillationEpochMetrics, ...]
    best_val_total: float
    last_checkpoint: Path | None
    best_checkpoint: Path | None


def load_distillation_config(
    path: Path,
    *,
    default_teacher_type: str | None = None,
) -> DistillationConfig:
    """Load distillation config from JSON or YAML."""
    if not isinstance(path, Path):
        raise ValueError("path must be a pathlib.Path.")
    if not path.exists():
        raise FileNotFoundError(f"Distillation config not found: {path}")

    try:
        if path.suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix in {".yaml", ".yml"}:
            import yaml  # type: ignore

            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise ValueError("Config file must be .json, .yaml, or .yml")
    except Exception as exc:
        raise RuntimeError(f"Failed to load distillation config: {exc}") from exc
    if raw is None:
        raw = {}
    return DistillationConfig.from_dict(
        raw,
        default_teacher_type=default_teacher_type,
    )


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def apply_global_magnitude_pruning(model: nn.Module, amount: float) -> None:
    """Apply global magnitude pruning to convolution and linear layers.

    Args:
        model: Target model.
        amount: Fraction to prune in [0, 1).

    Raises:
        ValueError: If amount is outside valid range.

    Physical Basis:
        Magnitude pruning removes low-importance weights and lowers compute
        burden while preserving the learned suppression structure.
    """
    if amount < 0.0 or amount >= 1.0:
        raise ValueError(f"amount must be in [0.0, 1.0), got {amount}.")
    if amount == 0.0:
        return

    targets: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, nn.Conv1d | nn.Conv2d | nn.ConvTranspose2d | nn.Linear):
            targets.append((module, "weight"))

    if not targets:
        raise ValueError("No prunable modules found in model.")

    prune.global_unstructured(
        targets,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, _ in targets:
        prune.remove(module, "weight")


def train_stage1_distillation(
    *,
    teacher: nn.Module,
    student: nn.Module,
    train_dataloader: DataLoader[dict[str, Any]],
    config: DistillationConfig,
    checkpoint_dir: Path | None = None,
    val_dataloader: DataLoader[dict[str, Any]] | None = None,
    model_config: Mapping[str, Any] | None = None,
    checkpoint_prefix: str = "stage1_distill",
) -> DistillationResult:
    """Train lightweight Stage 1 student using teacher distillation.

    Physical Basis:
        Student learns both task-aligned suppression targets and teacher
        behavior, reducing model size without changing Stage 1 signal flow.
    """
    if train_dataloader is None:
        raise ValueError("train_dataloader must be provided.")
    if len(checkpoint_prefix.strip()) == 0:
        raise ValueError("checkpoint_prefix must not be empty.")
    _set_seed(config.seed)
    device = select_device(
        device_override=config.device, require_cuda=config.require_cuda
    )

    teacher = teacher.to(device)
    student = student.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.use_amp and device.type == "cuda"
    )

    if checkpoint_dir is not None:
        ensure_dir(checkpoint_dir)

    best_val_total = float("inf")
    train_history: list[DistillationEpochMetrics] = []
    val_history: list[DistillationEpochMetrics] = []
    best_checkpoint: Path | None = None
    last_checkpoint: Path | None = None

    for epoch in range(config.epochs):
        student.train()
        train_metrics = _run_epoch(
            teacher=teacher,
            student=student,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            device=device,
            training=True,
        )
        train_history.append(train_metrics)

        if val_dataloader is not None:
            student.eval()
            with torch.no_grad():
                val_metrics = _run_epoch(
                    teacher=teacher,
                    student=student,
                    dataloader=val_dataloader,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    device=device,
                    training=False,
                )
            val_history.append(val_metrics)
            monitor_value = val_metrics.total
        else:
            monitor_value = train_metrics.total

        if checkpoint_dir is not None:
            state = _build_checkpoint_state(
                student=student,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                epoch=epoch,
                best_val_total=best_val_total,
                train_history=train_history,
                val_history=val_history,
                device=device,
                model_config=model_config,
            )
            last_checkpoint = checkpoint_dir / f"{checkpoint_prefix}_last.pt"
            save_checkpoint(last_checkpoint, state)

            if monitor_value < best_val_total:
                best_val_total = monitor_value
                state["best_val_total"] = best_val_total
                best_checkpoint = checkpoint_dir / f"{checkpoint_prefix}_best.pt"
                save_checkpoint(best_checkpoint, state)

        if (epoch + 1) % config.log_interval == 0 or epoch == 0:
            print(
                f"epoch={epoch + 1}/{config.epochs} "
                f"train_total={train_metrics.total:.6f} "
                f"train_task={train_metrics.task:.6f} "
                f"train_distill={train_metrics.distill:.6f}",
                flush=True,
            )

    return DistillationResult(
        device=str(device),
        train_history=tuple(train_history),
        val_history=tuple(val_history),
        best_val_total=best_val_total,
        last_checkpoint=last_checkpoint,
        best_checkpoint=best_checkpoint,
    )


def _run_epoch(
    *,
    teacher: nn.Module,
    student: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: DistillationConfig,
    device: torch.device,
    training: bool,
) -> DistillationEpochMetrics:
    start = time.perf_counter()
    total_loss_sum = 0.0
    task_loss_sum = 0.0
    distill_loss_sum = 0.0
    total_samples = 0
    total_steps = 0

    for batch in dataloader:
        hb_in = _batch_tensor(batch, "high_band", device)
        hb_target = _batch_tensor(batch, "hb_target", device)
        mirror_mask = _batch_tensor(batch, "mirror_mask", device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            with torch.no_grad():
                hb_teacher = cast(Any, teacher).forward_highband(hb_in)
            hb_student = cast(Any, student).forward_highband(hb_in)

            task_terms = compute_losses(
                hb_in=hb_in,
                hb_target=hb_target,
                hb_pred=hb_student,
                mirror_mask=mirror_mask,
                mask_config=config.mask_config,
                stft_configs=config.stft_configs,
                weights=config.task_loss_weights,
                energy_cap=config.energy_cap,
                ringing_config=config.ringing_loss_config,
                mask_mode=config.mask_mode,
            )
            distill_loss = _distillation_loss(
                hb_student=hb_student,
                hb_teacher=hb_teacher,
                mode=config.distillation_mode,
            )
            total_loss = (
                config.task_weight * task_terms.total
                + config.distillation_weight * distill_loss
            )

        if training:
            scaler.scale(total_loss).backward()
            if config.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        batch_size = int(hb_in.shape[0])
        total_samples += batch_size
        total_steps += 1
        total_loss_sum += float(total_loss.detach().item()) * batch_size
        task_loss_sum += float(task_terms.total.detach().item()) * batch_size
        distill_loss_sum += float(distill_loss.detach().item()) * batch_size

    elapsed = max(time.perf_counter() - start, 1.0e-6)
    if total_samples <= 0:
        raise RuntimeError("Empty dataloader is not supported.")
    return DistillationEpochMetrics(
        total=total_loss_sum / total_samples,
        task=task_loss_sum / total_samples,
        distill=distill_loss_sum / total_samples,
        samples=total_samples,
        steps=total_steps,
        throughput_samples_per_sec=float(total_samples / elapsed),
        throughput_steps_per_sec=float(total_steps / elapsed),
    )


def _distillation_loss(
    *,
    hb_student: torch.Tensor,
    hb_teacher: torch.Tensor,
    mode: LossMode,
) -> torch.Tensor:
    if hb_student.shape != hb_teacher.shape:
        raise ValueError("hb_student and hb_teacher must share shape.")
    if mode == "l1":
        return torch.mean(torch.abs(hb_student - hb_teacher))
    if mode == "l2":
        return torch.mean((hb_student - hb_teacher) ** 2)
    raise ValueError(f"Unsupported distillation mode: {mode}.")


def _batch_tensor(
    batch: Mapping[str, Any],
    key: str,
    device: torch.device,
) -> torch.Tensor:
    raw = batch.get(key)
    if not isinstance(raw, torch.Tensor):
        raise ValueError(f"batch[{key!r}] must be a torch.Tensor.")
    return raw.to(device)


def _build_checkpoint_state(
    *,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: DistillationConfig,
    epoch: int,
    best_val_total: float,
    train_history: Sequence[DistillationEpochMetrics],
    val_history: Sequence[DistillationEpochMetrics],
    device: torch.device,
    model_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "model_state": student.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        "training_config": asdict(config),
        "epoch": epoch,
        "best_val_total": best_val_total,
        "device": str(device),
        "train_history": [asdict(entry) for entry in train_history],
        "val_history": [asdict(entry) for entry in val_history],
    }
    if model_config is not None:
        state["model_config"] = dict(model_config)
    return state


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_stft_config(raw: Any) -> STFTLossConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("STFT config must be a mapping.")
    return STFTLossConfig(
        n_fft=int(raw.get("n_fft", 1024)),
        hop_length=int(raw.get("hop_length", 256)),
        win_length=int(raw.get("win_length", 1024)),
        center=_parse_bool(raw.get("center", True)),
    )


def _parse_loss_weights(
    raw: Any,
    *,
    hb_loss_weight: float,
    preserve_lb_weight: float,
    teacher_type: TeacherType,
) -> LossWeights:
    if not isinstance(raw, Mapping):
        raise ValueError("task_loss_weights must be a mapping.")
    return LossWeights(
        mask=float(raw.get("mask", hb_loss_weight)),
        stft=float(raw.get("stft", hb_loss_weight)),
        preserve=float(raw.get("preserve", preserve_lb_weight)),
        energy=float(raw.get("energy", 1.0)),
        subtract=float(raw.get("subtract", _default_subtractive_weight(teacher_type))),
        cap_strict=float(
            raw.get("cap_strict", _default_cap_strict_weight(teacher_type))
        ),
        edge=float(raw.get("edge", 0.05)),
        step=float(raw.get("step", 0.05)),
    )


def _parse_ringing_loss_config(raw: Any) -> RingingLossConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("ringing_loss_config must be a mapping.")
    return RingingLossConfig(
        edge_weight_cap=float(raw.get("edge_weight_cap", 4.0)),
        step_window_size=int(raw.get("step_window_size", 33)),
        eps=float(raw.get("eps", 1.0e-5)),
    )


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected boolean-like value, got {value!r}.")


def _parse_teacher_type(value: Any) -> TeacherType:
    if not isinstance(value, str):
        raise ValueError(
            f"teacher_type must be one of {ALLOWED_TEACHER_TYPES}, got {value!r}."
        )
    normalized = value.strip().lower()
    if normalized == "raw_88k2":
        return "raw_88k2"
    if normalized == "bessel_88k2":
        return "bessel_88k2"
    raise ValueError(
        f"teacher_type must be one of {ALLOWED_TEACHER_TYPES}, got {value!r}."
    )


def _default_energy_cap_for_teacher(teacher_type: TeacherType) -> float:
    if teacher_type == "raw_88k2":
        return 1.0e-3
    return 1.0


def _default_hb_loss_weight_for_teacher(teacher_type: TeacherType) -> float:
    del teacher_type
    return 1.0


def _default_preserve_lb_weight_for_teacher(teacher_type: TeacherType) -> float:
    del teacher_type
    return 1.0


def _default_subtractive_weight(teacher_type: TeacherType) -> float:
    if teacher_type == "raw_88k2":
        return 1.0
    return 0.0


def _default_cap_strict_weight(teacher_type: TeacherType) -> float:
    if teacher_type == "raw_88k2":
        return 4.0
    return 0.0


def _parse_loss_mode(value: Any) -> LossMode:
    text = str(value).strip().lower()
    if text in {"l1", "l2"}:
        return cast(LossMode, text)
    raise ValueError(f"Unsupported loss mode: {value!r}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
