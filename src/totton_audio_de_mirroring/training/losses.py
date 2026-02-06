"""Loss functions for mirror suppression training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

LossMode = Literal["l1", "l2"]


@dataclass(frozen=True)
class STFTLossConfig:
    """Configuration for STFT-based loss computation.

    Args:
        n_fft: FFT size.
        hop_length: Hop length between frames.
        win_length: Window length.
        center: Whether to center frames with padding.

    Physical Basis:
        STFT resolution controls which mirror structures the loss can
        emphasize while preserving time-frequency localization.
    """

    n_fft: int
    hop_length: int
    win_length: int
    center: bool = True

    def __post_init__(self) -> None:
        _validate_positive_int(self.n_fft, "n_fft")
        _validate_positive_int(self.hop_length, "hop_length")
        _validate_positive_int(self.win_length, "win_length")
        if self.win_length > self.n_fft:
            raise ValueError("win_length must be <= n_fft.")
        if self.hop_length > self.win_length:
            raise ValueError("hop_length must be <= win_length.")


@dataclass(frozen=True)
class LossWeights:
    """Weights for composite mirror suppression loss.

    Args:
        mask: Weight for mask loss.
        stft: Weight for multi-resolution STFT loss.
        preserve: Weight for preservation loss.
        energy: Weight for energy cap penalty.

    Physical Basis:
        Weighted combination balances suppression accuracy, time-frequency
        fidelity, and safety constraints in the high band.
    """

    mask: float = 1.0
    stft: float = 1.0
    preserve: float = 1.0
    energy: float = 1.0

    def __post_init__(self) -> None:
        _validate_non_negative(self.mask, "mask")
        _validate_non_negative(self.stft, "stft")
        _validate_non_negative(self.preserve, "preserve")
        _validate_non_negative(self.energy, "energy")


@dataclass(frozen=True)
class LossTerms:
    """Loss terms for mirror suppression training.

    Args:
        total: Weighted total loss.
        mask: Mask prediction loss.
        stft: Multi-resolution STFT loss.
        preserve: Preservation loss outside mirror bins.
        energy: Energy cap penalty.

    Physical Basis:
        Tracking each term ensures mirror suppression while maintaining
        time-domain integrity and high-band safety.
    """

    total: torch.Tensor
    mask: torch.Tensor
    stft: torch.Tensor
    preserve: torch.Tensor
    energy: torch.Tensor


def compute_losses(
    hb_in: torch.Tensor,
    hb_target: torch.Tensor,
    hb_pred: torch.Tensor,
    mirror_mask: torch.Tensor,
    *,
    mask_config: STFTLossConfig,
    stft_configs: Sequence[STFTLossConfig],
    weights: LossWeights,
    energy_cap: float,
    mask_mode: LossMode = "l1",
    eps: float = 1.0e-8,
) -> LossTerms:
    """Compute composite loss for mirror suppression.

    Args:
        hb_in: High-band input signal (batch, time).
        hb_target: High-band target signal (batch, time).
        hb_pred: Predicted high-band output (batch, time).
        mirror_mask: Mirror detection mask (batch, freq, time) or (freq, time).
        mask_config: STFT config for mask and preservation losses.
        stft_configs: STFT configs for multi-resolution loss.
        weights: Loss weights.
        energy_cap: Maximum allowed high-band energy (sum of mag^2).
        mask_mode: Loss mode for mask loss ("l1" or "l2").
        eps: Small constant for numerical stability.

    Returns:
        LossTerms containing each component and total.

    Physical Basis:
        Mask loss aligns suppression with target attenuation, STFT loss
        enforces spectral fidelity, preservation loss protects non-mirror
        bins, and energy loss enforces safety caps.
    """
    _validate_signal_2d(hb_in, "hb_in")
    _validate_signal_2d(hb_target, "hb_target")
    _validate_signal_2d(hb_pred, "hb_pred")
    if hb_in.shape != hb_target.shape or hb_in.shape != hb_pred.shape:
        raise ValueError("hb_in, hb_target, and hb_pred must share shape.")
    if not stft_configs:
        raise ValueError("stft_configs must be non-empty.")
    _validate_positive_float(energy_cap, "energy_cap")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    hb_in_mag = _stft_magnitude(hb_in, mask_config)
    hb_target_mag = _stft_magnitude(hb_target, mask_config)
    hb_pred_mag = _stft_magnitude(hb_pred, mask_config)

    pred_mask = compute_target_mask(hb_in_mag, hb_pred_mag, eps=eps)
    target_mask = compute_target_mask(hb_in_mag, hb_target_mag, eps=eps)
    loss_mask = mask_loss(pred_mask, target_mask, mode=mask_mode)
    loss_preserve = preserve_loss(hb_pred_mag, hb_in_mag, mirror_mask)
    loss_stft = multi_resolution_stft_loss(hb_pred, hb_target, stft_configs)
    loss_energy = energy_cap_loss(hb_pred_mag, energy_cap)

    total = (
        weights.mask * loss_mask
        + weights.stft * loss_stft
        + weights.preserve * loss_preserve
        + weights.energy * loss_energy
    )

    return LossTerms(
        total=total,
        mask=loss_mask,
        stft=loss_stft,
        preserve=loss_preserve,
        energy=loss_energy,
    )


def compute_target_mask(
    hb_in_mag: torch.Tensor,
    hb_target_mag: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Compute target suppression mask from magnitudes.

    Args:
        hb_in_mag: Input STFT magnitude (batch, freq, time).
        hb_target_mag: Target STFT magnitude (batch, freq, time).
        eps: Small constant to avoid division by zero.

    Returns:
        Target mask in [0, 1].

    Physical Basis:
        The ratio of target to input magnitudes encodes how strongly
        mirror components should be attenuated.
    """
    _validate_mag_tensor(hb_in_mag, "hb_in_mag")
    _validate_mag_tensor(hb_target_mag, "hb_target_mag")
    if hb_in_mag.shape != hb_target_mag.shape:
        raise ValueError("Magnitude tensors must share shape.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    mask = hb_target_mag / (hb_in_mag + eps)
    return torch.clamp(mask, 0.0, 1.0)


def mask_loss(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    mode: LossMode = "l1",
) -> torch.Tensor:
    """Compute mask prediction loss.

    Args:
        pred_mask: Predicted mask (batch, freq, time).
        target_mask: Target mask (batch, freq, time).
        mode: Loss mode ("l1" or "l2").

    Returns:
        Scalar loss value.

    Physical Basis:
        Penalizing mask mismatch focuses learning on suppressing only
        mirror-related energy.
    """
    _validate_mag_tensor(pred_mask, "pred_mask")
    _validate_mag_tensor(target_mask, "target_mask")
    if pred_mask.shape != target_mask.shape:
        raise ValueError("pred_mask and target_mask must share shape.")
    if mode == "l1":
        return torch.mean(torch.abs(pred_mask - target_mask))
    if mode == "l2":
        return torch.mean((pred_mask - target_mask) ** 2)
    raise ValueError(f"Unsupported mask loss mode: {mode}.")


def multi_resolution_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    configs: Sequence[STFTLossConfig],
) -> torch.Tensor:
    """Compute multi-resolution STFT magnitude loss.

    Args:
        pred: Predicted high-band signal (batch, time).
        target: Target high-band signal (batch, time).
        configs: Sequence of STFT configurations.

    Returns:
        Scalar loss value.

    Physical Basis:
        Multi-resolution STFT loss captures both fine and coarse mirror
        artifact structures without altering time-domain identity.
    """
    _validate_signal_2d(pred, "pred")
    _validate_signal_2d(target, "target")
    if pred.shape != target.shape:
        raise ValueError("pred and target must share shape.")
    if not configs:
        raise ValueError("configs must be non-empty.")

    losses: list[torch.Tensor] = []
    for cfg in configs:
        pred_mag = _stft_magnitude(pred, cfg)
        target_mag = _stft_magnitude(target, cfg)
        losses.append(torch.mean(torch.abs(pred_mag - target_mag)))
    return torch.stack(losses).mean()


def preserve_loss(
    pred_mag: torch.Tensor,
    input_mag: torch.Tensor,
    mirror_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize changes outside mirror detection regions.

    Args:
        pred_mag: Predicted STFT magnitude (batch, freq, time).
        input_mag: Input STFT magnitude (batch, freq, time).
        mirror_mask: Mirror detection mask (batch, freq, time) or (freq, time).

    Returns:
        Scalar loss value.

    Physical Basis:
        Non-mirror bins should remain unchanged to preserve time-domain
        fidelity and avoid unnecessary suppression.
    """
    _validate_mag_tensor(pred_mag, "pred_mag")
    _validate_mag_tensor(input_mag, "input_mag")
    if pred_mag.shape != input_mag.shape:
        raise ValueError("pred_mag and input_mag must share shape.")

    mask = _broadcast_mask(mirror_mask, pred_mag.shape)
    mask = mask.to(device=pred_mag.device, dtype=pred_mag.dtype)
    outside = 1.0 - torch.clamp(mask, 0.0, 1.0)
    diff = torch.abs(pred_mag - input_mag)
    return torch.mean(diff * outside)


def energy_cap_loss(pred_mag: torch.Tensor, energy_cap: float) -> torch.Tensor:
    """Penalize violations of the high-band energy cap.

    Args:
        pred_mag: Predicted STFT magnitude (batch, freq, time).
        energy_cap: Maximum allowed energy (sum of mag^2).

    Returns:
        Scalar loss value.

    Physical Basis:
        Enforcing a fixed cap on total high-band energy reduces IMD risk
        and keeps ultrasonic content in a safe range.
    """
    _validate_mag_tensor(pred_mag, "pred_mag")
    _validate_positive_float(energy_cap, "energy_cap")

    energy = torch.sum(pred_mag**2, dim=(-2, -1))
    excess = torch.clamp(energy - energy_cap, min=0.0)
    return torch.mean(excess)


def _stft_magnitude(signal: torch.Tensor, config: STFTLossConfig) -> torch.Tensor:
    """Compute STFT magnitude for a batched signal.

    Args:
        signal: Input signal (batch, time).
        config: STFT configuration.

    Returns:
        STFT magnitude (batch, freq, time).

    Physical Basis:
        Magnitude STFT exposes mirror artifacts as time-frequency energy
        patterns used for mask and preservation losses.
    """
    _validate_signal_2d(signal, "signal")
    window = torch.hann_window(
        config.win_length,
        periodic=True,
        device=signal.device,
        dtype=signal.dtype,
    )
    stft = torch.stft(
        signal,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=config.center,
        return_complex=True,
    )
    return torch.abs(stft)


def _broadcast_mask(mask: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Broadcast a mirror mask to a target shape.

    Args:
        mask: Mirror mask tensor (freq, time) or (batch, freq, time).
        shape: Target shape (batch, freq, time).

    Returns:
        Broadcasted mirror mask.

    Physical Basis:
        Mirror detection is defined per time-frequency bin and must align
        with batch-wise STFT magnitudes.
    """
    if len(shape) != 3:
        raise ValueError("shape must be (batch, freq, time).")
    if not torch.is_tensor(mask):
        raise ValueError("mirror_mask must be a torch.Tensor.")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError("mirror_mask must be 2D or 3D.")
    if mask.shape[1:] != shape[1:]:
        raise ValueError("mirror_mask frequency/time dims must match.")
    if mask.shape[0] == 1 and shape[0] > 1:
        mask = mask.expand(shape[0], -1, -1)
    if mask.shape[0] != shape[0]:
        raise ValueError("mirror_mask batch dimension mismatch.")
    return mask


def _validate_signal_2d(signal: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(signal):
        raise ValueError(f"{name} must be a torch.Tensor.")
    if signal.ndim != 2:
        raise ValueError(f"{name} must be 2D (batch, time).")
    if signal.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")


def _validate_mag_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(tensor):
        raise ValueError(f"{name} must be a torch.Tensor.")
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be 3D (batch, freq, time).")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_positive_float(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _validate_non_negative(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
