"""Runtime data structures and checkpoint helpers for training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from totton_audio_de_mirroring.training.losses import STFTLossConfig


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated metrics for one epoch."""

    total: float
    mask: float
    stft: float
    preserve: float
    energy: float
    subtract: float
    cap_strict: float
    edge: float
    step: float
    contrib_mask: float
    contrib_stft: float
    contrib_preserve: float
    contrib_energy: float
    contrib_subtract: float
    contrib_cap_strict: float
    contrib_edge: float
    contrib_step: float
    mirror_reduction_db: float
    touch_l1: float
    energy_cap_violation: float
    lb_mag_mae: float
    lb_phase_mae: float
    samples: int
    steps: int
    throughput_samples_per_sec: float
    throughput_steps_per_sec: float
    gpu_peak_memory_mb: float


@dataclass(frozen=True)
class TrainingResult:
    """Outputs from Stage 1 training."""

    device: str
    train_history: tuple[EpochMetrics, ...]
    val_history: tuple[EpochMetrics, ...]
    best_val_total: float
    last_checkpoint: Path | None
    best_checkpoint: Path | None


def build_checkpoint_state(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ExponentialLR | None,
    scaler: torch.amp.GradScaler,
    config: Any,
    epoch: int,
    best_val_total: float,
    train_history: list[EpochMetrics],
    val_history: list[EpochMetrics],
    device: torch.device,
) -> dict[str, Any]:
    """Build checkpoint dictionary.

    Physical Basis:
        Persisting optimizer/scaler/runtime state keeps learning dynamics
        reproducible when training resumes.
    """
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        "training_config": asdict(config),
        "epoch": epoch,
        "best_val_total": best_val_total,
        "device": str(device),
        "train_history": [asdict(metrics) for metrics in train_history],
        "val_history": [asdict(metrics) for metrics in val_history],
    }


def save_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    """Save checkpoint with guarded IO error handling."""
    try:
        ensure_dir(path.parent)
        torch.save(dict(state), path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save checkpoint at {path}: {exc}") from exc


def restore_checkpoint(
    *,
    resume_from: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ExponentialLR | None,
    scaler: torch.amp.GradScaler,
) -> tuple[int, float]:
    """Restore model/optimizer/scaler state.

    Returns:
        Tuple of (start_epoch, best_val_total).
    """
    if not resume_from.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

    try:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {exc}") from exc

    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler.is_enabled() and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val_total = float(checkpoint.get("best_val_total", float("inf")))
    return start_epoch, best_val_total


def gpu_peak_memory_mb(device: torch.device, *, enabled: bool) -> float:
    """Get current GPU peak memory in MB."""
    if not enabled or device.type != "cuda":
        return 0.0
    allocated = torch.cuda.max_memory_allocated(device)
    return float(allocated / (1024**2))


def ensure_dir(path: Path) -> None:
    """Create directory recursively with IO error translation."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to create directory {path}: {exc}") from exc


def compute_lowband_metrics(
    *,
    x_full: torch.Tensor,
    y_full: torch.Tensor,
    sample_rate: float,
    cutoff_hz: float,
    stft_config: STFTLossConfig,
) -> tuple[float, float]:
    """Compute low-band magnitude and phase deviations via STFT mask."""
    x_2d = _ensure_2d_audio(x_full)
    y_2d = _ensure_2d_audio(y_full)
    if x_2d.shape != y_2d.shape:
        raise ValueError("x_full and y_full must have the same shape.")

    window = torch.hann_window(
        stft_config.win_length,
        periodic=True,
        device=x_2d.device,
        dtype=x_2d.dtype,
    )
    x_spec = torch.stft(
        x_2d,
        n_fft=stft_config.n_fft,
        hop_length=stft_config.hop_length,
        win_length=stft_config.win_length,
        center=stft_config.center,
        window=window,
        return_complex=True,
    )
    y_spec = torch.stft(
        y_2d,
        n_fft=stft_config.n_fft,
        hop_length=stft_config.hop_length,
        win_length=stft_config.win_length,
        center=stft_config.center,
        window=window,
        return_complex=True,
    )

    freqs = torch.linspace(
        0.0,
        sample_rate / 2.0,
        x_spec.shape[-2],
        device=x_spec.device,
        dtype=x_spec.real.dtype,
    )
    low_mask = freqs <= cutoff_hz
    low_mask_3d = low_mask.view(1, -1, 1).to(dtype=x_spec.real.dtype)

    x_mag = torch.abs(x_spec)
    y_mag = torch.abs(y_spec)
    expanded_mask = low_mask_3d.expand_as(x_mag)
    mask_norm = torch.sum(expanded_mask)
    mag_mae = (torch.sum(torch.abs(y_mag - x_mag) * expanded_mask) / mask_norm).item()

    phase_diff = torch.angle(y_spec) - torch.angle(x_spec)
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    phase_mae = (torch.sum(torch.abs(phase_diff) * expanded_mask) / mask_norm).item()
    return float(mag_mae), float(phase_mae)


def _ensure_2d_audio(value: torch.Tensor) -> torch.Tensor:
    """Flatten input to (batch, time)."""
    if value.ndim == 2:
        return value
    if value.ndim == 3:
        batch, channels, time_dim = value.shape
        return value.reshape(batch * channels, time_dim)
    raise ValueError(f"Expected 2D/3D tensor, got {value.ndim}D")
