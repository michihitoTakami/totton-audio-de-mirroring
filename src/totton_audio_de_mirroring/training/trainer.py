"""Training loop utilities for Stage 1 NMSE."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from totton_audio_de_mirroring.training.losses import (
    LossMode,
    LossTerms,
    LossWeights,
    STFTLossConfig,
    _broadcast_mask,
    _stft_magnitude,
    compute_losses,
)
from totton_audio_de_mirroring.training.runtime import (
    EpochMetrics,
    TrainingResult,
    build_checkpoint_state,
    compute_lowband_metrics,
    ensure_dir,
    gpu_peak_memory_mb,
    restore_checkpoint,
    save_checkpoint,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for NMSE training.

    Args:
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        grad_clip: Optional gradient clipping norm.
        use_amp: Whether to use automatic mixed precision on CUDA.
        log_interval: Steps between logging updates.
        mask_config: STFT config for mask/preserve losses.
        stft_configs: STFT configs for multi-resolution loss.
        loss_weights: Composite loss weights.
        energy_cap: Energy cap used for loss penalty.
        mask_mode: Loss mode for mask loss.
        device: Optional device override (e.g., "cuda", "cpu").
        seed: Optional random seed.
        require_cuda: Require GPU execution.
        scheduler_gamma: Optional exponential scheduler gamma.
        log_memory: Whether to report GPU peak memory.

    Physical Basis:
        Training config aligns the optimization loop with mirror suppression
        objectives while keeping safety constraints explicit.
    """

    epochs: int = 10
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    use_amp: bool = True
    log_interval: int = 20
    mask_config: STFTLossConfig = STFTLossConfig(
        n_fft=2048, hop_length=512, win_length=2048
    )
    stft_configs: tuple[STFTLossConfig, ...] = (
        STFTLossConfig(n_fft=1024, hop_length=256, win_length=1024),
        STFTLossConfig(n_fft=2048, hop_length=512, win_length=2048),
    )
    loss_weights: LossWeights = LossWeights()
    energy_cap: float = 1.0
    mask_mode: LossMode = "l1"
    device: str | None = None
    seed: int | None = None
    require_cuda: bool = True
    scheduler_gamma: float | None = None
    log_memory: bool = True

    def __post_init__(self) -> None:
        _validate_positive_int(self.epochs, "epochs")
        _validate_positive_float(self.learning_rate, "learning_rate")
        _validate_non_negative(self.weight_decay, "weight_decay")
        if self.grad_clip is not None:
            _validate_positive_float(self.grad_clip, "grad_clip")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive.")
        if not self.stft_configs:
            raise ValueError("stft_configs must be non-empty.")
        _validate_positive_float(self.energy_cap, "energy_cap")
        if self.scheduler_gamma is not None:
            _validate_positive_float(self.scheduler_gamma, "scheduler_gamma")
            if self.scheduler_gamma > 1.0:
                raise ValueError("scheduler_gamma must be <= 1.0.")

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> TrainingConfig:
        """Build TrainingConfig from a mapping.

        Args:
            raw: Raw mapping from JSON/YAML.

        Returns:
            Parsed TrainingConfig instance.

        Physical Basis:
            External configuration keeps loss balance tunable without
            modifying code, enabling safer iterative experiments.
        """
        if not isinstance(raw, Mapping):
            raise ValueError("raw must be a mapping.")

        mask_cfg = _parse_stft_config(raw.get("mask_config", {}))
        stft_list = raw.get("stft_configs")
        stft_cfgs = (
            tuple(_parse_stft_config(cfg) for cfg in stft_list)
            if isinstance(stft_list, list)
            else (
                STFTLossConfig(n_fft=1024, hop_length=256, win_length=1024),
                STFTLossConfig(n_fft=2048, hop_length=512, win_length=2048),
            )
        )
        weights = _parse_loss_weights(raw.get("loss_weights", {}))

        return TrainingConfig(
            epochs=int(raw.get("epochs", 10)),
            learning_rate=float(raw.get("learning_rate", 1.0e-4)),
            weight_decay=float(raw.get("weight_decay", 0.0)),
            grad_clip=_optional_float(raw.get("grad_clip", 1.0)),
            use_amp=_parse_bool(raw.get("use_amp", True)),
            log_interval=int(raw.get("log_interval", 20)),
            mask_config=mask_cfg,
            stft_configs=stft_cfgs,
            loss_weights=weights,
            energy_cap=float(raw.get("energy_cap", 1.0)),
            mask_mode=_parse_mask_mode(raw.get("mask_mode", "l1")),
            device=raw.get("device"),
            seed=_optional_int(raw.get("seed")),
            require_cuda=_parse_bool(raw.get("require_cuda", True)),
            scheduler_gamma=_optional_float(raw.get("scheduler_gamma")),
            log_memory=_parse_bool(raw.get("log_memory", True)),
        )


def load_training_config(path: Path) -> TrainingConfig:
    """Load TrainingConfig from JSON or YAML.

    Args:
        path: Config file path.

    Returns:
        Parsed TrainingConfig.

    Physical Basis:
        External config files enable repeatable experiments across
        training runs and hardware setups.
    """
    if not isinstance(path, Path):
        raise ValueError("path must be a pathlib.Path.")
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")

    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix in {".yaml", ".yml"}:
            import yaml  # type: ignore

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise ValueError("Config file must be .json, .yaml, or .yml")
    except Exception as exc:
        raise RuntimeError(f"Failed to load training config: {exc}") from exc

    return TrainingConfig.from_dict(data or {})


def select_device(
    prefer_cuda: bool = True,
    device_override: str | None = None,
    *,
    require_cuda: bool = False,
) -> torch.device:
    """Select a torch device with optional CUDA requirement.

    Args:
        prefer_cuda: Prefer CUDA when available.
        device_override: Optional explicit device string.
        require_cuda: If True, require CUDA device.

    Returns:
        Selected torch.device.

    Raises:
        RuntimeError: If CUDA is required but unavailable or overridden to CPU.

    Physical Basis:
        GPU acceleration is required for practical training throughput
        at high sample rates and long receptive fields.
    """
    if device_override is not None:
        selected = torch.device(device_override)
    elif prefer_cuda and torch.cuda.is_available():
        selected = torch.device("cuda")
    else:
        selected = torch.device("cpu")

    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but CUDA is not available.")
    if require_cuda and selected.type != "cuda":
        raise RuntimeError(
            "GPU training is required, but CUDA device was not selected."
        )
    return selected


def train_stage1(
    model: nn.Module,
    train_dataloader: DataLoader[dict[str, Any]],
    config: TrainingConfig,
    *,
    val_dataloader: DataLoader[dict[str, Any]] | None = None,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
) -> TrainingResult:
    """Train NMSE Stage 1 with composite losses.

    Args:
        model: NMSE model with forward_highband method.
        train_dataloader: DataLoader yielding high-band training batches.
        config: Training configuration.
        val_dataloader: Optional validation DataLoader.
        checkpoint_dir: Optional checkpoint output directory.
        resume_from: Optional checkpoint path for resume.

    Returns:
        TrainingResult with histories and checkpoint paths.

    Raises:
        ValueError: If model or dataloaders are invalid.

    Physical Basis:
        Training optimizes mirror suppression while preserving non-mirror
        components and enforcing high-band energy safety.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch.nn.Module.")
    if not hasattr(model, "forward_highband"):
        raise ValueError("model must implement forward_highband().")
    if train_dataloader is None:
        raise ValueError("train_dataloader must be provided.")

    device = select_device(
        device_override=config.device, require_cuda=config.require_cuda
    )
    _set_seed(config.seed)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.scheduler_gamma,
        )
        if config.scheduler_gamma is not None
        else None
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.use_amp and device.type == "cuda",
    )

    if checkpoint_dir is not None:
        ensure_dir(checkpoint_dir)

    train_history: list[EpochMetrics] = []
    val_history: list[EpochMetrics] = []
    best_val_total = math.inf
    best_checkpoint: Path | None = None
    last_checkpoint: Path | None = None

    start_epoch = 0
    if resume_from is not None:
        start_epoch, best_val_total = restore_checkpoint(
            resume_from=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

    try:
        for epoch in range(start_epoch, config.epochs):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            train_epoch = _run_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                config=config,
                epoch=epoch,
                training=True,
            )
            train_history.append(train_epoch)

            val_epoch: EpochMetrics | None = None
            if val_dataloader is not None:
                val_epoch = _run_epoch(
                    model=model,
                    dataloader=val_dataloader,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    config=config,
                    epoch=epoch,
                    training=False,
                )
                val_history.append(val_epoch)

            if scheduler is not None:
                scheduler.step()

            monitor_total = (
                val_epoch.total if val_epoch is not None else train_epoch.total
            )
            is_best = monitor_total < best_val_total
            if is_best:
                best_val_total = monitor_total

            _log_epoch_summary(
                epoch=epoch,
                train_epoch=train_epoch,
                val_epoch=val_epoch,
                lr=optimizer.param_groups[0]["lr"],
                device=device,
            )

            if checkpoint_dir is not None:
                checkpoint_state = build_checkpoint_state(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    epoch=epoch,
                    best_val_total=best_val_total,
                    train_history=train_history,
                    val_history=val_history,
                    device=device,
                )
                last_checkpoint = checkpoint_dir / "stage1_last.pt"
                save_checkpoint(last_checkpoint, checkpoint_state)
                if is_best:
                    best_checkpoint = checkpoint_dir / "stage1_best.pt"
                    save_checkpoint(best_checkpoint, checkpoint_state)
    except Exception as exc:
        if checkpoint_dir is not None:
            emergency_state = build_checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                epoch=max(start_epoch, len(train_history) - 1),
                best_val_total=best_val_total,
                train_history=train_history,
                val_history=val_history,
                device=device,
            )
            save_checkpoint(checkpoint_dir / "stage1_emergency.pt", emergency_state)
        raise RuntimeError(f"Training failed: {exc}") from exc

    return TrainingResult(
        device=str(device),
        train_history=tuple(train_history),
        val_history=tuple(val_history),
        best_val_total=best_val_total,
        last_checkpoint=last_checkpoint,
        best_checkpoint=best_checkpoint,
    )


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
    *,
    training: bool,
) -> EpochMetrics:
    """Run one train/validation epoch and aggregate metrics."""
    totals = {
        "total": 0.0,
        "mask": 0.0,
        "stft": 0.0,
        "preserve": 0.0,
        "energy": 0.0,
        "mirror_reduction_db": 0.0,
        "touch_l1": 0.0,
        "energy_cap_violation": 0.0,
        "lb_mag_mae": 0.0,
        "lb_phase_mae": 0.0,
    }
    lb_count = 0
    sample_count = 0
    step_count = 0

    start = time.perf_counter()
    model.train(mode=training)

    for step, batch in enumerate(dataloader):
        hb_in, hb_target, mirror_mask = _prepare_batch(batch, device)
        batch_size = hb_in.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                hb_pred = _forward_highband(model, hb_in)
                terms = compute_losses(
                    hb_in,
                    hb_target,
                    hb_pred,
                    mirror_mask,
                    mask_config=config.mask_config,
                    stft_configs=config.stft_configs,
                    weights=config.loss_weights,
                    energy_cap=config.energy_cap,
                    mask_mode=config.mask_mode,
                )
            scaler.scale(terms.total).backward()
            if config.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                hb_pred = _forward_highband(model, hb_in)
                terms = compute_losses(
                    hb_in,
                    hb_target,
                    hb_pred,
                    mirror_mask,
                    mask_config=config.mask_config,
                    stft_configs=config.stft_configs,
                    weights=config.loss_weights,
                    energy_cap=config.energy_cap,
                    mask_mode=config.mask_mode,
                )

        batch_metrics = _compute_batch_metrics(
            model=model,
            batch=batch,
            hb_in=hb_in,
            hb_pred=hb_pred,
            mirror_mask=mirror_mask,
            device=device,
            mask_config=config.mask_config,
            energy_cap=config.energy_cap,
            compute_low_band=not training,
        )

        totals["total"] += terms.total.detach().item()
        totals["mask"] += terms.mask.detach().item()
        totals["stft"] += terms.stft.detach().item()
        totals["preserve"] += terms.preserve.detach().item()
        totals["energy"] += terms.energy.detach().item()
        totals["mirror_reduction_db"] += batch_metrics["mirror_reduction_db"]
        totals["touch_l1"] += batch_metrics["touch_l1"]
        totals["energy_cap_violation"] += batch_metrics["energy_cap_violation"]
        if batch_metrics["lb_available"]:
            totals["lb_mag_mae"] += batch_metrics["lb_mag_mae"]
            totals["lb_phase_mae"] += batch_metrics["lb_phase_mae"]
            lb_count += 1

        step_count += 1
        sample_count += batch_size

        if training and step % config.log_interval == 0:
            _log_step_progress(epoch=epoch, step=step, terms=terms)

    if step_count == 0:
        raise ValueError("dataloader produced zero batches.")

    elapsed = max(time.perf_counter() - start, 1.0e-9)
    peak_memory = gpu_peak_memory_mb(device, enabled=config.log_memory)

    return EpochMetrics(
        total=totals["total"] / step_count,
        mask=totals["mask"] / step_count,
        stft=totals["stft"] / step_count,
        preserve=totals["preserve"] / step_count,
        energy=totals["energy"] / step_count,
        mirror_reduction_db=totals["mirror_reduction_db"] / step_count,
        touch_l1=totals["touch_l1"] / step_count,
        energy_cap_violation=totals["energy_cap_violation"] / step_count,
        lb_mag_mae=(totals["lb_mag_mae"] / lb_count) if lb_count > 0 else 0.0,
        lb_phase_mae=(totals["lb_phase_mae"] / lb_count) if lb_count > 0 else 0.0,
        samples=sample_count,
        steps=step_count,
        throughput_samples_per_sec=float(sample_count / elapsed),
        throughput_steps_per_sec=float(step_count / elapsed),
        gpu_peak_memory_mb=peak_memory,
    )


def _compute_batch_metrics(
    *,
    model: nn.Module,
    batch: Mapping[str, Any],
    hb_in: torch.Tensor,
    hb_pred: torch.Tensor,
    mirror_mask: torch.Tensor,
    device: torch.device,
    mask_config: STFTLossConfig,
    energy_cap: float,
    compute_low_band: bool,
) -> dict[str, float | bool]:
    """Compute monitor-only batch metrics."""
    eps = 1.0e-8
    hb_in_mag = _stft_magnitude(hb_in, mask_config)
    hb_pred_mag = _stft_magnitude(hb_pred, mask_config)

    mirror = _broadcast_mask(mirror_mask, hb_in_mag.shape).to(
        device=hb_in_mag.device,
        dtype=hb_in_mag.dtype,
    )
    mirror = torch.clamp(mirror, 0.0, 1.0)

    in_energy = torch.sum((hb_in_mag**2) * mirror, dim=(-2, -1))
    out_energy = torch.sum((hb_pred_mag**2) * mirror, dim=(-2, -1))
    mirror_reduction_db = torch.mean(
        10.0 * torch.log10((in_energy + eps) / (out_energy + eps))
    ).item()

    touch = torch.mean(torch.abs(hb_pred_mag - hb_in_mag) * (1.0 - mirror)).item()
    hb_energy = torch.sum(hb_pred_mag**2, dim=(-2, -1))
    energy_violation = torch.mean(torch.clamp(hb_energy - energy_cap, min=0.0)).item()

    lb_mag_mae = 0.0
    lb_phase_mae = 0.0
    lb_available = False
    if compute_low_band and "x_full" in batch:
        x_full = _to_device_tensor(batch["x_full"], device)
        if x_full is not None:
            with torch.no_grad():
                y_full = model(x_full)
            sample_rate = _get_model_float_attr(model, "sample_rate")
            cutoff_hz = _get_model_float_attr(model, "cutoff_hz")
            if sample_rate is not None and cutoff_hz is not None:
                lb_mag_mae, lb_phase_mae = compute_lowband_metrics(
                    x_full=x_full,
                    y_full=y_full,
                    sample_rate=sample_rate,
                    cutoff_hz=cutoff_hz,
                    stft_config=mask_config,
                )
                lb_available = True

    return {
        "mirror_reduction_db": float(mirror_reduction_db),
        "touch_l1": float(touch),
        "energy_cap_violation": float(energy_violation),
        "lb_mag_mae": float(lb_mag_mae),
        "lb_phase_mae": float(lb_phase_mae),
        "lb_available": lb_available,
    }


def _prepare_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract and move batch tensors to device."""
    if "high_band" not in batch or "hb_target" not in batch:
        raise ValueError("batch must contain high_band and hb_target.")
    if "mirror_mask" not in batch:
        raise ValueError("batch must contain mirror_mask.")

    hb_in = _to_device_tensor(batch["high_band"], device)
    hb_target = _to_device_tensor(batch["hb_target"], device)
    mirror_mask = _to_device_tensor(batch["mirror_mask"], device)

    if hb_in is None or hb_target is None or mirror_mask is None:
        raise ValueError("batch tensors must be torch.Tensor.")
    return hb_in, hb_target, mirror_mask


def _forward_highband(model: nn.Module, hb_in: torch.Tensor) -> torch.Tensor:
    """Forward high-band data through the NMSE model."""
    forward_fn = cast(Callable[[torch.Tensor], torch.Tensor], model.forward_highband)
    return forward_fn(hb_in)


def _log_step_progress(epoch: int, step: int, terms: LossTerms) -> None:
    """Log step progress."""
    print(
        " ".join(
            [
                f"epoch={epoch}",
                f"step={step}",
                f"total={terms.total.item():.6f}",
                f"mask={terms.mask.item():.6f}",
                f"stft={terms.stft.item():.6f}",
                f"preserve={terms.preserve.item():.6f}",
                f"energy={terms.energy.item():.6f}",
            ]
        )
    )


def _log_epoch_summary(
    *,
    epoch: int,
    train_epoch: EpochMetrics,
    val_epoch: EpochMetrics | None,
    lr: float,
    device: torch.device,
) -> None:
    """Log epoch summary."""
    train_line = (
        f"epoch={epoch} split=train total={train_epoch.total:.6f} "
        f"mirror_db={train_epoch.mirror_reduction_db:.3f} "
        f"touch={train_epoch.touch_l1:.6f} "
        f"energy_violation={train_epoch.energy_cap_violation:.6f} "
        f"samples_per_sec={train_epoch.throughput_samples_per_sec:.2f} "
        f"steps_per_sec={train_epoch.throughput_steps_per_sec:.2f} "
        f"gpu_peak_mb={train_epoch.gpu_peak_memory_mb:.2f}"
    )
    print(train_line)

    if val_epoch is not None:
        val_line = (
            f"epoch={epoch} split=val total={val_epoch.total:.6f} "
            f"mirror_db={val_epoch.mirror_reduction_db:.3f} "
            f"touch={val_epoch.touch_l1:.6f} "
            f"energy_violation={val_epoch.energy_cap_violation:.6f} "
            f"lb_mag_mae={val_epoch.lb_mag_mae:.6f} "
            f"lb_phase_mae={val_epoch.lb_phase_mae:.6f}"
        )
        print(val_line)

    device_line = f"epoch={epoch} lr={lr:.8f} device={device}"
    if device.type == "cuda":
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        device_line += f" gpu={name} cuda_index={index}"
    print(device_line)


def _to_device_tensor(value: Any, device: torch.device) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    return value.to(device, non_blocking=True)


def _get_model_float_attr(model: nn.Module, name: str) -> float | None:
    value = getattr(model, name, None)
    if isinstance(value, int | float):
        return float(value)
    return None


def _set_seed(seed: int | None) -> None:
    """Set random seeds for reproducibility."""
    if seed is None:
        return
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_stft_config(raw: Mapping[str, Any]) -> STFTLossConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("STFT config must be a mapping.")
    return STFTLossConfig(
        n_fft=int(raw.get("n_fft", 2048)),
        hop_length=int(raw.get("hop_length", 512)),
        win_length=int(raw.get("win_length", 2048)),
        center=bool(raw.get("center", True)),
    )


def _parse_loss_weights(raw: Mapping[str, Any]) -> LossWeights:
    if not isinstance(raw, Mapping):
        raise ValueError("loss_weights must be a mapping.")
    return LossWeights(
        mask=float(raw.get("mask", 1.0)),
        stft=float(raw.get("stft", 1.0)),
        preserve=float(raw.get("preserve", 1.0)),
        energy=float(raw.get("energy", 1.0)),
    )


def _parse_mask_mode(value: Any) -> LossMode:
    if value not in {"l1", "l2"}:
        raise ValueError("mask_mode must be 'l1' or 'l2'.")
    return cast(LossMode, value)


def _parse_bool(value: Any) -> bool:
    """Parse common bool representations."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Invalid bool string: {value}")
    if isinstance(value, int):
        if value in {0, 1}:
            return bool(value)
        raise ValueError(f"Invalid bool integer: {value}")
    raise ValueError(f"Invalid bool value type: {type(value).__name__}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_positive_float(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _validate_non_negative(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
