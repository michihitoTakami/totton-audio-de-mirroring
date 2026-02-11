"""Loss functions for mirror suppression training."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

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
        subtract: Weight for subtractive-suppression penalty.
        cap_strict: Weight for strict cap-violation penalty.
        edge: Weight for edge-aligned ringing loss.
        step: Weight for step-response ringing loss.

    Physical Basis:
        Weighted combination balances suppression accuracy, time-frequency
        fidelity, and safety constraints in the high band.
    """

    mask: float = 1.0
    stft: float = 1.0
    preserve: float = 1.0
    energy: float = 1.0
    subtract: float = 0.0
    cap_strict: float = 0.0
    edge: float = 0.0
    step: float = 0.0

    def __post_init__(self) -> None:
        _validate_non_negative(self.mask, "mask")
        _validate_non_negative(self.stft, "stft")
        _validate_non_negative(self.preserve, "preserve")
        _validate_non_negative(self.energy, "energy")
        _validate_non_negative(self.subtract, "subtract")
        _validate_non_negative(self.cap_strict, "cap_strict")
        _validate_non_negative(self.edge, "edge")
        _validate_non_negative(self.step, "step")


@dataclass(frozen=True)
class RingingLossConfig:
    """Configuration for edge/step ringing auxiliary losses.

    Args:
        edge_weight_cap: Upper bound for edge weighting.
        step_window_size: Dilation window for edge-focused step loss.
        eps: Numerical stability constant.

    Physical Basis:
        Ringing artifacts emerge around sharp edges. Emphasizing edge
        neighborhoods in the objective helps training avoid ripple regression.
    """

    edge_weight_cap: float = 4.0
    step_window_size: int = 33
    eps: float = 1.0e-5

    def __post_init__(self) -> None:
        _validate_positive_float(self.edge_weight_cap, "edge_weight_cap")
        _validate_positive_int(self.step_window_size, "step_window_size")
        if self.step_window_size % 2 == 0:
            raise ValueError("step_window_size must be odd.")
        _validate_positive_float(self.eps, "eps")


@dataclass(frozen=True)
class LossTerms:
    """Loss terms for mirror suppression training.

    Args:
        total: Weighted total loss.
        mask: Mask prediction loss.
        stft: Multi-resolution STFT loss.
        preserve: Preservation loss outside mirror bins.
        energy: Energy cap penalty.
        subtract: Subtractive-suppression penalty.
        cap_strict: Strict cap-violation penalty.
        edge: Edge-aligned ringing loss.
        step: Step-response ringing loss.

    Physical Basis:
        Tracking each term ensures mirror suppression while maintaining
        time-domain integrity and high-band safety.
    """

    total: torch.Tensor
    mask: torch.Tensor
    stft: torch.Tensor
    preserve: torch.Tensor
    energy: torch.Tensor
    subtract: torch.Tensor
    cap_strict: torch.Tensor
    edge: torch.Tensor
    step: torch.Tensor


@dataclass(frozen=True)
class LossContributionRatios:
    """Weighted loss contribution ratios for one optimization step.

    Args:
        mask: Weighted contribution ratio of mask loss.
        stft: Weighted contribution ratio of STFT loss.
        preserve: Weighted contribution ratio of preservation loss.
        energy: Weighted contribution ratio of energy cap loss.
        subtract: Weighted contribution ratio of subtractive suppression loss.
        cap_strict: Weighted contribution ratio of strict cap loss.
        edge: Weighted contribution ratio of edge ringing loss.
        step: Weighted contribution ratio of step ringing loss.

    Physical Basis:
        Tracking weighted ratios quantifies which objectives dominate
        optimization and prevents accidental drift from mirror-first goals.
    """

    mask: float
    stft: float
    preserve: float
    energy: float
    subtract: float
    cap_strict: float
    edge: float
    step: float


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
    highband_mask: torch.Tensor | None = None,
    ringing_config: RingingLossConfig | None = None,
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
        energy_cap: Maximum allowed mean high-band STFT energy (mean of mag^2).
        highband_mask: Optional (freq,) mask indicating high-band bins.
        ringing_config: Configuration for ringing auxiliary losses.
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
    active_ringing_config = ringing_config or RingingLossConfig()

    hb_in_mag = _stft_magnitude(hb_in, mask_config)
    hb_target_mag = _stft_magnitude(hb_target, mask_config)
    hb_pred_mag = _stft_magnitude(hb_pred, mask_config)

    pred_mask = compute_target_mask(hb_in_mag, hb_pred_mag, eps=eps)
    target_mask = compute_target_mask(hb_in_mag, hb_target_mag, eps=eps)
    loss_mask = mask_loss(pred_mask, target_mask, mode=mask_mode)
    loss_preserve = preserve_loss(hb_pred_mag, hb_in_mag, mirror_mask)
    loss_stft = multi_resolution_stft_loss(hb_pred, hb_target, stft_configs)
    loss_energy = energy_cap_loss(hb_pred_mag, energy_cap, highband_mask=highband_mask)
    loss_subtract = subtractive_suppression_loss(hb_pred_mag, hb_in_mag)
    loss_cap_strict = strict_energy_cap_loss(
        hb_pred_mag, energy_cap, highband_mask=highband_mask
    )
    loss_edge = ringing_edge_loss(hb_pred, hb_target, config=active_ringing_config)
    loss_step = ringing_step_loss(hb_pred, hb_target, config=active_ringing_config)

    total = (
        weights.mask * loss_mask
        + weights.stft * loss_stft
        + weights.preserve * loss_preserve
        + weights.energy * loss_energy
        + weights.subtract * loss_subtract
        + weights.cap_strict * loss_cap_strict
        + weights.edge * loss_edge
        + weights.step * loss_step
    )

    return LossTerms(
        total=total,
        mask=loss_mask,
        stft=loss_stft,
        preserve=loss_preserve,
        energy=loss_energy,
        subtract=loss_subtract,
        cap_strict=loss_cap_strict,
        edge=loss_edge,
        step=loss_step,
    )


def compute_loss_contribution_ratios(
    terms: LossTerms,
    weights: LossWeights,
    *,
    eps: float = 1.0e-12,
) -> LossContributionRatios:
    """Compute weighted contribution ratios of each loss term.

    Args:
        terms: Per-step loss terms.
        weights: Loss weights used for total loss.
        eps: Numerical stability constant.

    Returns:
        Weighted contribution ratios that sum to 1.0.

    Physical Basis:
        Weighted ratios provide interpretable diagnostics for balancing
        mirror suppression, safety, and ringing constraints.
    """
    _validate_positive_float(eps, "eps")
    weighted = {
        "mask": max(weights.mask * float(terms.mask.detach().item()), 0.0),
        "stft": max(weights.stft * float(terms.stft.detach().item()), 0.0),
        "preserve": max(weights.preserve * float(terms.preserve.detach().item()), 0.0),
        "energy": max(weights.energy * float(terms.energy.detach().item()), 0.0),
        "subtract": max(
            weights.subtract * float(terms.subtract.detach().item()),
            0.0,
        ),
        "cap_strict": max(
            weights.cap_strict * float(terms.cap_strict.detach().item()),
            0.0,
        ),
        "edge": max(weights.edge * float(terms.edge.detach().item()), 0.0),
        "step": max(weights.step * float(terms.step.detach().item()), 0.0),
    }
    total = sum(weighted.values())
    if total <= eps:
        uniform = 1.0 / 8.0
        return LossContributionRatios(
            mask=uniform,
            stft=uniform,
            preserve=uniform,
            energy=uniform,
            subtract=uniform,
            cap_strict=uniform,
            edge=uniform,
            step=uniform,
        )
    return LossContributionRatios(
        mask=weighted["mask"] / total,
        stft=weighted["stft"] / total,
        preserve=weighted["preserve"] / total,
        energy=weighted["energy"] / total,
        subtract=weighted["subtract"] / total,
        cap_strict=weighted["cap_strict"] / total,
        edge=weighted["edge"] / total,
        step=weighted["step"] / total,
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


def energy_cap_loss(
    pred_mag: torch.Tensor,
    energy_cap: float,
    *,
    highband_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize violations of the high-band energy cap.

    Args:
        pred_mag: Predicted STFT magnitude (batch, freq, time).
        energy_cap: Maximum allowed mean energy (mean of mag^2).
        highband_mask: Optional (freq,) mask indicating high-band bins.

    Returns:
        Scalar loss value.

    Physical Basis:
        Enforcing a fixed cap on total high-band energy reduces IMD risk
        and keeps ultrasonic content in a safe range.
    """
    _validate_mag_tensor(pred_mag, "pred_mag")
    _validate_positive_float(energy_cap, "energy_cap")

    mag_sq = pred_mag**2
    if highband_mask is not None:
        if highband_mask.ndim != 1:
            raise ValueError("highband_mask must be 1D (freq,).")
        if highband_mask.shape[0] != pred_mag.shape[1]:
            raise ValueError("highband_mask length mismatch.")
        mask = highband_mask.to(device=pred_mag.device, dtype=torch.bool)
        if torch.count_nonzero(mask) > 0:
            mag_sq = mag_sq[:, mask, :]

    energy = torch.mean(mag_sq, dim=(-2, -1))
    excess = torch.clamp(energy - energy_cap, min=0.0)
    return torch.mean(excess)


def subtractive_suppression_loss(
    pred_mag: torch.Tensor,
    input_mag: torch.Tensor,
) -> torch.Tensor:
    """Penalize high-band magnitude increase to prioritize subtraction.

    Args:
        pred_mag: Predicted STFT magnitude (batch, freq, time).
        input_mag: Input STFT magnitude (batch, freq, time).

    Returns:
        Scalar additive-energy penalty.

    Physical Basis:
        Raw teacher migration should prioritize removing mirrored energy.
        Penalizing only positive gain keeps learning subtractive.
    """
    _validate_mag_tensor(pred_mag, "pred_mag")
    _validate_mag_tensor(input_mag, "input_mag")
    if pred_mag.shape != input_mag.shape:
        raise ValueError("pred_mag and input_mag must share shape.")
    additive = torch.clamp(pred_mag - input_mag, min=0.0)
    return torch.mean(additive)


def strict_energy_cap_loss(
    pred_mag: torch.Tensor,
    energy_cap: float,
    *,
    highband_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply nonlinear penalty to any high-band cap violation.

    Args:
        pred_mag: Predicted STFT magnitude (batch, freq, time).
        energy_cap: Maximum allowed mean energy (mean of mag^2).
        highband_mask: Optional (freq,) mask indicating high-band bins.

    Returns:
        Scalar strict cap penalty.

    Physical Basis:
        Squared normalized excess heavily penalizes cap violations and drives
        violation rate toward zero for IMD safety.
    """
    _validate_mag_tensor(pred_mag, "pred_mag")
    _validate_positive_float(energy_cap, "energy_cap")
    mag_sq = pred_mag**2
    if highband_mask is not None:
        if highband_mask.ndim != 1:
            raise ValueError("highband_mask must be 1D (freq,).")
        if highband_mask.shape[0] != pred_mag.shape[1]:
            raise ValueError("highband_mask length mismatch.")
        mask = highband_mask.to(device=pred_mag.device, dtype=torch.bool)
        if torch.count_nonzero(mask) > 0:
            mag_sq = mag_sq[:, mask, :]

    energy = torch.mean(mag_sq, dim=(-2, -1))
    normalized_excess = torch.clamp((energy - energy_cap) / energy_cap, min=0.0)
    return torch.mean(normalized_excess**2)


def ringing_edge_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    config: RingingLossConfig | None = None,
) -> torch.Tensor:
    """Penalize edge-derivative mismatch to suppress ringing regression.

    Args:
        pred: Predicted high-band signal (batch, time).
        target: Target high-band signal (batch, time).
        config: Ringing loss configuration.

    Returns:
        Scalar edge-focused loss value.

    Physical Basis:
        Ringing growth appears as derivative mismatch near transitions.
        Edge-weighted derivative alignment constrains transient behavior.
    """
    _validate_signal_2d(pred, "pred")
    _validate_signal_2d(target, "target")
    if pred.shape != target.shape:
        raise ValueError("pred and target must share shape.")
    active_config = config or RingingLossConfig()
    promoted_dtype = torch.promote_types(pred.dtype, target.dtype)
    working_dtype = _ringing_compute_dtype(promoted_dtype)

    with _autocast_disabled(pred.device.type):
        pred_stable = pred.to(dtype=working_dtype)
        target_stable = target.to(dtype=working_dtype)
        pred_diff = torch.diff(pred_stable, dim=-1)
        target_diff = torch.diff(target_stable, dim=-1)
        weights = _edge_weights(
            target_diff,
            edge_weight_cap=active_config.edge_weight_cap,
            eps=active_config.eps,
        )
        return torch.mean(torch.abs(pred_diff - target_diff) * weights)


def ringing_step_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    config: RingingLossConfig | None = None,
) -> torch.Tensor:
    """Penalize edge-neighborhood step-response mismatch.

    Args:
        pred: Predicted high-band signal (batch, time).
        target: Target high-band signal (batch, time).
        config: Ringing loss configuration.

    Returns:
        Scalar step-response loss value.

    Physical Basis:
        Step-response mismatch around edges captures overshoot and ripple
        tendencies that are not fully reflected by global spectral losses.
    """
    _validate_signal_2d(pred, "pred")
    _validate_signal_2d(target, "target")
    if pred.shape != target.shape:
        raise ValueError("pred and target must share shape.")
    active_config = config or RingingLossConfig()
    promoted_dtype = torch.promote_types(pred.dtype, target.dtype)
    working_dtype = _ringing_compute_dtype(promoted_dtype)

    with _autocast_disabled(pred.device.type):
        pred_stable = pred.to(dtype=working_dtype)
        target_stable = target.to(dtype=working_dtype)
        pred_diff = torch.diff(pred_stable, dim=-1)
        target_diff = torch.diff(target_stable, dim=-1)
        weights = _edge_weights(
            target_diff,
            edge_weight_cap=active_config.edge_weight_cap,
            eps=active_config.eps,
        )
        dilated_weights = F.max_pool1d(
            weights.unsqueeze(1),
            kernel_size=active_config.step_window_size,
            stride=1,
            padding=active_config.step_window_size // 2,
        ).squeeze(1)

        pred_step = torch.cumsum(pred_diff, dim=-1)
        target_step = torch.cumsum(target_diff, dim=-1)
        return torch.mean(torch.abs(pred_step - target_step) * dilated_weights)


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
        normalized=True,
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
    if mask.shape[0] == 1 and shape[0] > 1:
        mask = mask.expand(shape[0], -1, -1)
    if mask.shape[0] != shape[0]:
        raise ValueError("mirror_mask batch dimension mismatch.")
    if mask.shape[1:] != shape[1:]:
        mask = _resize_mask(mask, target_freq=shape[1], target_time=shape[2])
    return mask


def _edge_weights(
    target_diff: torch.Tensor,
    *,
    edge_weight_cap: float,
    eps: float,
) -> torch.Tensor:
    """Build normalized edge weights from first-order target derivative."""
    if target_diff.ndim != 2:
        raise ValueError("target_diff must be 2D (batch, time-1).")
    magnitude = torch.abs(target_diff)
    scale = torch.mean(magnitude, dim=-1, keepdim=True)
    safe_min = max(eps, torch.finfo(scale.dtype).tiny)
    safe_scale = torch.clamp(scale, min=safe_min)
    normalized = magnitude / safe_scale
    return torch.clamp(normalized, min=0.0, max=edge_weight_cap)


def _ringing_compute_dtype(input_dtype: torch.dtype) -> torch.dtype:
    """Select stable compute dtype for ringing auxiliary losses.

    Args:
        input_dtype: Input tensor dtype.

    Returns:
        Float32 for low-precision inputs, otherwise the original dtype.

    Physical Basis:
        Derivative and cumulative operations need higher dynamic range
        than float16/bfloat16 to avoid NaN/Inf under AMP.
    """
    if input_dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return input_dtype


@contextmanager
def _autocast_disabled(device_type: str) -> Iterator[None]:
    """Temporarily disable autocast for numerically sensitive losses.

    Args:
        device_type: Device type string (for example, "cuda" or "cpu").

    Yields:
        Context where autocast is disabled when supported.

    Physical Basis:
        Ringing losses use derivative and cumulative operations that are
        sensitive to float16 underflow and accumulation error.
    """
    if device_type in {"cuda", "cpu"}:
        with torch.amp.autocast(device_type=device_type, enabled=False):
            yield
        return
    yield


def _resize_mask(
    mask: torch.Tensor, target_freq: int, target_time: int
) -> torch.Tensor:
    """Resize mask to match STFT bin/time dimensions.

    Args:
        mask: Mask tensor with shape (batch, freq, time).
        target_freq: Target frequency-bin count.
        target_time: Target time-frame count.

    Returns:
        Resized mask tensor with shape (batch, target_freq, target_time).

    Physical Basis:
        Mirror masks can be generated on grids that differ slightly from
        training STFT grids. Nearest-neighbor resizing preserves bin-wise
        detection semantics while aligning dimensions for loss weighting.
    """
    if target_freq <= 0 or target_time <= 0:
        raise ValueError("target_freq and target_time must be positive.")

    original_dtype = mask.dtype
    mask_4d = mask.unsqueeze(1).to(dtype=torch.float32)
    resized = F.interpolate(
        mask_4d,
        size=(target_freq, target_time),
        mode="nearest",
    )
    resized_3d = resized.squeeze(1)
    if original_dtype == torch.bool:
        return resized_3d > 0.5
    return resized_3d.to(dtype=original_dtype)


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
