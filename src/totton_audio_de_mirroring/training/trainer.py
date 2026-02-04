"""Training loop utilities for Stage 1 NMSE."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from totton_audio_de_mirroring.training.losses import (
    LossMode,
    LossTerms,
    LossWeights,
    STFTLossConfig,
    compute_losses,
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
            use_amp=bool(raw.get("use_amp", True)),
            log_interval=int(raw.get("log_interval", 20)),
            mask_config=mask_cfg,
            stft_configs=stft_cfgs,
            loss_weights=weights,
            energy_cap=float(raw.get("energy_cap", 1.0)),
            mask_mode=_parse_mask_mode(raw.get("mask_mode", "l1")),
            device=raw.get("device"),
            seed=_optional_int(raw.get("seed")),
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
    prefer_cuda: bool = True, device_override: str | None = None
) -> torch.device:
    """Select a torch device with optional CUDA preference.

    Args:
        prefer_cuda: Prefer CUDA when available.
        device_override: Optional explicit device string.

    Returns:
        Selected torch.device.

    Physical Basis:
        GPU acceleration is required for practical training throughput
        at high sample rates and long receptive fields.
    """
    if device_override is not None:
        return torch.device(device_override)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_stage1(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    config: TrainingConfig,
) -> list[LossTerms]:
    """Train NMSE Stage 1 with composite losses.

    Args:
        model: NMSE model with forward_highband method.
        dataloader: DataLoader yielding high-band batches.
        config: Training configuration.

    Returns:
        List of LossTerms per optimization step.

    Raises:
        ValueError: If model or dataloader are invalid.

    Physical Basis:
        Training optimizes mirror suppression while preserving non-mirror
        components and enforcing high-band energy safety.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch.nn.Module.")
    if not hasattr(model, "forward_highband"):
        raise ValueError("model must implement forward_highband().")
    if dataloader is None:
        raise ValueError("dataloader must be provided.")

    device = select_device(device_override=config.device)
    _set_seed(config.seed)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")

    losses: list[LossTerms] = []
    for epoch in range(config.epochs):
        epoch_losses = _train_epoch(
            model,
            dataloader,
            optimizer,
            scaler,
            device,
            config,
            epoch,
        )
        losses.extend(epoch_losses)

    return losses


def _train_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
) -> list[LossTerms]:
    """Train a single epoch.

    Args:
        model: NMSE model.
        dataloader: Training DataLoader.
        optimizer: Optimizer instance.
        scaler: AMP grad scaler.
        device: Device for training.
        config: Training configuration.
        epoch: Current epoch index.

    Returns:
        Loss terms for each step in the epoch.

    Physical Basis:
        Epoch-wise iteration ensures stable convergence while respecting
        the mirror suppression constraints.
    """
    step_losses: list[LossTerms] = []
    for step, batch in enumerate(dataloader):
        hb_in, hb_target, mirror_mask = _prepare_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
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

        if step % config.log_interval == 0:
            _log_progress(epoch, step, terms)
        step_losses.append(terms)

    return step_losses


def _prepare_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract and move batch tensors to device.

    Args:
        batch: Batch dictionary from DataLoader.
        device: Device for tensors.

    Returns:
        Tuple of (hb_in, hb_target, mirror_mask) tensors.

    Physical Basis:
        Moving only required tensors keeps data flow minimal while
        preserving mirror mask alignment.
    """
    if "high_band" not in batch or "hb_target" not in batch:
        raise ValueError("batch must contain high_band and hb_target.")
    if "mirror_mask" not in batch:
        raise ValueError("batch must contain mirror_mask.")

    hb_in = batch["high_band"].to(device, non_blocking=True)
    hb_target = batch["hb_target"].to(device, non_blocking=True)
    mirror_mask = batch["mirror_mask"].to(device, non_blocking=True)

    return hb_in, hb_target, mirror_mask


def _forward_highband(model: nn.Module, hb_in: torch.Tensor) -> torch.Tensor:
    """Forward high-band data through the NMSE model.

    Args:
        model: NMSE model with forward_highband method.
        hb_in: High-band input tensor.

    Returns:
        Predicted high-band output.

    Physical Basis:
        High-band-only forward keeps low-band bypass intact and focuses
        learning on mirror suppression.
    """
    forward_fn = model.forward_highband
    return forward_fn(hb_in)


def _log_progress(epoch: int, step: int, terms: LossTerms) -> None:
    """Log training progress.

    Args:
        epoch: Epoch index.
        step: Step index.
        terms: Loss terms for the step.

    Physical Basis:
        Lightweight logging tracks convergence without perturbing training.
    """
    message = (
        f"epoch={epoch} step={step} total={terms.total.item():.6f} "
        f"mask={terms.mask.item():.6f} stft={terms.stft.item():.6f} "
        f"preserve={terms.preserve.item():.6f} energy={terms.energy.item():.6f}"
    )
    print(message)


def _set_seed(seed: int | None) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Seed value or None.

    Physical Basis:
        Deterministic seeding aids in comparing mirror suppression behavior.
    """
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
    return value


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
