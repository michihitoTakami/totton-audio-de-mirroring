"""Loss terms for CAPB training (gate-aligned, mask-driven).

The fidelity terms pull toward the band-limited teacher; the ringing terms
use the dataset's clean-signal masks to penalize exactly what the gates
measure: high-frequency ripple on plateaus and any energy in silences.
The TV term keeps the blend trajectory slow and decisive.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from totton_audio_de_mirroring.training.losses import (
    STFTLossConfig,
    multi_resolution_stft_loss,
)

_EPS = 1e-8


@dataclass(frozen=True)
class CAPBLossWeights:
    """Weights for the CAPB composite loss.

    Args:
        wave: Time-domain L1 against the band-limited teacher.
        stft: Multi-resolution STFT magnitude loss.
        plateau: Ripple penalty on clean-signal plateaus.
        quiet: Energy penalty in clean-signal silences (pre/post echo).
        tv: Total variation of the blend-weight trajectories.

    Physical Basis:
        wave/stft define fidelity to the alias-free teacher; plateau/quiet
        are differentiable versions of the ringing gates and deliberately
        oppose fidelity at edges - the controller resolves the conflict by
        selecting the gentle prototype only where ringing terms dominate.
    """

    wave: float = 1.0
    stft: float = 1.0
    plateau: float = 20.0
    quiet: float = 20.0
    tv: float = 0.1


def compute_capb_losses(
    output: torch.Tensor,
    target: torch.Tensor,
    weights_frames: torch.Tensor,
    flat_mask: torch.Tensor,
    quiet_mask: torch.Tensor,
    stft_configs: list[STFTLossConfig],
    loss_weights: CAPBLossWeights,
    trim: int = 512,
) -> dict[str, torch.Tensor]:
    """Compute all CAPB loss terms.

    Args:
        output: Model output (batch, time) at the target rate.
        target: Band-limited teacher (batch, time).
        weights_frames: Blend weights (batch, K, frames).
        flat_mask: Plateau mask from the clean signal (batch, time).
        quiet_mask: Silence mask from the clean signal (batch, time).
        stft_configs: STFT resolutions for the spectral loss.
        loss_weights: Term weights.
        trim: Samples trimmed at each chunk border (filter edge effects).

    Returns:
        Mapping with per-term losses and the weighted "total".

    Raises:
        ValueError: If shapes are inconsistent.

    Physical Basis:
        Chunk borders carry incomplete convolution context for the long
        sharp prototype, so a border margin is excluded from every term.
    """
    if output.shape != target.shape:
        raise ValueError("output and target must share shape.")
    if flat_mask.shape != output.shape or quiet_mask.shape != output.shape:
        raise ValueError("masks must share the output shape.")
    if trim < 0 or 2 * trim >= output.shape[-1]:
        raise ValueError("trim must be non-negative and smaller than half length.")

    sl = slice(trim, output.shape[-1] - trim) if trim else slice(None)
    out = output[:, sl]
    tgt = target[:, sl]
    flat = flat_mask[:, sl]
    quiet = quiet_mask[:, sl]

    losses = {
        "wave": torch.mean(torch.abs(out - tgt)),
        "stft": multi_resolution_stft_loss(out, tgt, stft_configs),
        "plateau": plateau_ripple_loss(out, flat),
        "quiet": quiet_energy_loss(out, quiet),
        "tv": weight_tv_loss(weights_frames),
    }
    losses["total"] = (
        loss_weights.wave * losses["wave"]
        + loss_weights.stft * losses["stft"]
        + loss_weights.plateau * losses["plateau"]
        + loss_weights.quiet * losses["quiet"]
        + loss_weights.tv * losses["tv"]
    )
    return losses


def plateau_ripple_loss(output: torch.Tensor, flat_mask: torch.Tensor) -> torch.Tensor:
    """Penalize sample-to-sample ripple on clean-signal plateaus.

    Args:
        output: Model output (batch, time).
        flat_mask: Plateau mask (batch, time), 1.0 on plateaus.

    Returns:
        Scalar loss.

    Physical Basis:
        On a plateau of the clean signal the ideal time response is flat;
        the first difference isolates the 20 kHz-scale Gibbs ripple that
        the plateau gates measure, independent of the plateau level.
    """
    _validate_pair(output, flat_mask)
    ripple = output[:, 1:] - output[:, :-1]
    mask = flat_mask[:, 1:] * flat_mask[:, :-1]
    return torch.sum(torch.square(ripple) * mask) / (torch.sum(mask) + _EPS)


def quiet_energy_loss(output: torch.Tensor, quiet_mask: torch.Tensor) -> torch.Tensor:
    """Penalize output energy where the clean signal is silent.

    Args:
        output: Model output (batch, time).
        quiet_mask: Silence mask (batch, time), 1.0 in silence.

    Returns:
        Scalar loss.

    Physical Basis:
        Energy emitted in clean-signal silence is pre/post-echo of nearby
        transients (kernel side lobes); this is the differentiable form of
        the pre-echo gate and needs no onset detection.
    """
    _validate_pair(output, quiet_mask)
    return torch.sum(torch.square(output) * quiet_mask) / (torch.sum(quiet_mask) + _EPS)


def weight_tv_loss(weights_frames: torch.Tensor) -> torch.Tensor:
    """Total variation of blend-weight trajectories.

    Args:
        weights_frames: Blend weights (batch, K, frames).

    Returns:
        Scalar loss.

    Physical Basis:
        Slow weight trajectories bound the bandwidth of blend modulation,
        preventing the time-varying mix itself from creating sidebands.
    """
    if weights_frames.dim() != 3:
        raise ValueError("weights_frames must be (batch, K, frames).")
    if weights_frames.shape[-1] < 2:
        return weights_frames.new_zeros(())
    return torch.mean(torch.abs(weights_frames[:, :, 1:] - weights_frames[:, :, :-1]))


def _validate_pair(output: torch.Tensor, mask: torch.Tensor) -> None:
    if output.dim() != 2 or output.shape != mask.shape:
        raise ValueError("output and mask must be matching (batch, time) tensors.")
