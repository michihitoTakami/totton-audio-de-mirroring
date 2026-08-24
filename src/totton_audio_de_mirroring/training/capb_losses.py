"""Loss terms for CAPB training (gate-aligned, mask-driven).

The fidelity terms pull toward the band-limited teacher; the ringing terms
use the dataset's clean-signal masks to penalize exactly what the gates
measure: high-frequency ripple on plateaus and any energy in silences.
The TV term keeps the blend trajectory slow and decisive.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from totton_audio_de_mirroring.training.stft_loss import (
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
        entropy_floor: Barrier keeping softmax away from one-hot dead zones.
        edge_fidelity_relax: Fraction by which waveform fidelity is relaxed
            inside edge zones (0.9 keeps 10% of the wave loss there).
        edge_ring: Penalty for edge-zone ripple exceeding the gentle
            prototype's ripple (covers dense-edge content with no plateaus).
        selection: Generator-labelled prototype-selection supervision.
        min_entropy: Per-frame entropy floor in nats (~0.05 allows a max
            blend weight of roughly 0.99).

    Physical Basis:
        wave/stft define fidelity to the alias-free teacher; plateau/quiet
        are differentiable versions of the ringing gates and deliberately
        oppose fidelity at edges - the controller resolves the conflict by
        selecting the gentle prototype only where ringing terms dominate.
    """

    wave: float = 1.0
    stft: float = 1.0
    plateau: float = 100.0
    quiet: float = 100.0
    tv: float = 0.1
    entropy_floor: float = 10.0
    edge_fidelity_relax: float = 0.9
    edge_ring: float = 300.0
    selection: float = 0.5
    min_entropy: float = 0.05


def compute_capb_losses(
    output: torch.Tensor,
    target: torch.Tensor,
    weights_frames: torch.Tensor,
    flat_mask: torch.Tensor,
    quiet_mask: torch.Tensor,
    stft_configs: list[STFTLossConfig],
    loss_weights: CAPBLossWeights,
    trim: int = 512,
    edge_mask: torch.Tensor | None = None,
    selection_mask: torch.Tensor | None = None,
    gentle_output: torch.Tensor | None = None,
    gentle_index: int = -1,
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
        edge_mask: Broadband-transient neighborhoods where waveform
            fidelity is relaxed (see edge_fidelity_relax).
        selection_mask: Frame target for gentle prototype selection.
        gentle_output: The gentle prototype's output for the same input;
            enables the edge_ring loss when given with edge_mask.
        gentle_index: Index of the gentle weight in weights_frames.

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

    # At broadband transients the brickwall teacher itself rings; demanding
    # full fidelity there would force the controller to reproduce ringing.
    # Fidelity is therefore relaxed inside the edge zones (and only there).
    wave_error = torch.abs(out - tgt)
    if edge_mask is not None:
        if edge_mask.shape != output.shape:
            raise ValueError("edge_mask must share the output shape.")
        relax = loss_weights.edge_fidelity_relax
        wave_error = wave_error * (1.0 - relax * edge_mask[:, sl])
    if selection_mask is not None and selection_mask.shape != output.shape:
        raise ValueError("selection_mask must share the output shape.")

    losses = {
        "wave": torch.mean(wave_error),
        "stft": multi_resolution_stft_loss(out, tgt, stft_configs),
        "plateau": plateau_ripple_loss(out, flat),
        "quiet": quiet_energy_loss(out, quiet),
        "tv": weight_tv_loss(weights_frames),
        "entropy_floor": entropy_floor_loss(
            weights_frames, min_entropy=loss_weights.min_entropy
        ),
    }
    if edge_mask is not None and gentle_output is not None:
        losses["edge_ring"] = edge_ring_loss(
            out, gentle_output[:, sl], edge_mask[:, sl]
        )
        target_mask = selection_mask if selection_mask is not None else edge_mask
        losses["selection"] = prototype_selection_loss(
            weights_frames, target_mask[:, sl], gentle_index
        )
    else:
        losses["edge_ring"] = output.new_zeros(())
        losses["selection"] = output.new_zeros(())
    losses["total"] = (
        loss_weights.wave * losses["wave"]
        + loss_weights.stft * losses["stft"]
        + loss_weights.plateau * losses["plateau"]
        + loss_weights.quiet * losses["quiet"]
        + loss_weights.tv * losses["tv"]
        + loss_weights.entropy_floor * losses["entropy_floor"]
        + loss_weights.edge_ring * losses["edge_ring"]
        + loss_weights.selection * losses["selection"]
    )
    return losses


def plateau_ripple_loss(
    output: torch.Tensor,
    flat_mask: torch.Tensor,
    top_fraction: float = 0.02,
) -> torch.Tensor:
    """Penalize the worst sample-to-sample ripple on clean-signal plateaus.

    Args:
        output: Model output (batch, time).
        flat_mask: Plateau mask (batch, time), 1.0 on plateaus.
        top_fraction: Fraction of worst masked samples averaged.

    Returns:
        Scalar loss.

    Physical Basis:
        Interpolation ringing concentrates in the few milliseconds next to
        each edge; a plain mean over whole plateaus dilutes it below the
        fidelity gradients (observed as always-sharp collapse in run1). A
        top-k mean matches the gates' worst-case semantics instead.
    """
    _validate_pair(output, flat_mask)
    ripple = output[:, 1:] - output[:, :-1]
    mask = flat_mask[:, 1:] * flat_mask[:, :-1]
    return _top_k_masked_mean(torch.square(ripple), mask, top_fraction)


def quiet_energy_loss(
    output: torch.Tensor,
    quiet_mask: torch.Tensor,
    top_fraction: float = 0.05,
) -> torch.Tensor:
    """Penalize the worst output energy where the clean signal is silent.

    Args:
        output: Model output (batch, time).
        quiet_mask: Silence mask (batch, time), 1.0 in silence.
        top_fraction: Fraction of worst masked samples averaged.

    Returns:
        Scalar loss.

    Physical Basis:
        Pre/post-echo is localized around transients; averaging over all
        silence dilutes it, so the top-k mean targets the echo skirt that
        the pre-echo gate actually measures.
    """
    _validate_pair(output, quiet_mask)
    return _top_k_masked_mean(torch.square(output), quiet_mask, top_fraction)


def _top_k_masked_mean(
    values: torch.Tensor, mask: torch.Tensor, top_fraction: float
) -> torch.Tensor:
    """Mean of the largest masked values (differentiable worst-case proxy)."""
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}.")
    masked = (values * mask).flatten()
    count = int(torch.count_nonzero(mask).item())
    if count == 0:
        return values.new_zeros(())
    k = max(1, int(count * top_fraction))
    top_values, _ = torch.topk(masked, k)
    return top_values.mean()


def edge_ring_loss(
    output: torch.Tensor,
    gentle_output: torch.Tensor,
    edge_mask: torch.Tensor,
    top_fraction: float = 0.05,
) -> torch.Tensor:
    """Penalize edge-zone ripple exceeding the gentle prototype's ripple.

    Args:
        output: Model output (batch, time).
        gentle_output: Gentle prototype output for the same input.
        edge_mask: Broadband-transient mask (batch, time).
        top_fraction: Fraction of worst masked samples averaged.

    Returns:
        Scalar loss.

    Physical Basis:
        Dense-edge content (e.g. 5 kHz squares) has no plateaus for the
        plateau loss to act on, yet the ringing gates still demand
        reference-like smoothness there. The gentle prototype defines the
        achievable ring-free ripple level, so only the EXCESS over it is
        penalized - the loss is exactly zero when the blend goes gentle.
    """
    _validate_pair(output, edge_mask)
    if gentle_output.shape != output.shape:
        raise ValueError("gentle_output must share the output shape.")
    ripple = torch.square(output[:, 1:] - output[:, :-1])
    gentle_ripple = torch.square(gentle_output[:, 1:] - gentle_output[:, :-1])
    excess = torch.relu(ripple - gentle_ripple)
    mask = edge_mask[:, 1:] * edge_mask[:, :-1]
    return _top_k_masked_mean(excess, mask, top_fraction)


def prototype_selection_loss(
    weights_frames: torch.Tensor,
    edge_mask: torch.Tensor,
    gentle_index: int = -1,
) -> torch.Tensor:
    """Teach edge frames to select gentle and steady frames to reject it.

    Args:
        weights_frames: Blend weights (batch, prototypes, frames).
        edge_mask: Generator-labelled edge mask (batch, output time).
        gentle_index: Index of the gentle prototype.

    Returns:
        Frame-weighted edge/steady binary cross-entropy.

    Raises:
        ValueError: If shapes or the gentle index are invalid.

    Physical Basis:
        Ringing losses alone can make an always-gentle solution locally
        attractive. Generator labels provide an unambiguous control target:
        use gentle only around known discontinuities and retain the
        high-rejection Kaiser endpoints elsewhere. Frame weighting preserves
        the corpus' actual transient duty cycle, preventing sparse labelled
        events from imposing an artificial 50/50 gentle prior.
    """
    if weights_frames.dim() != 3 or edge_mask.dim() != 2:
        raise ValueError("weights_frames and edge_mask must be 3D/2D tensors.")
    if weights_frames.shape[0] != edge_mask.shape[0]:
        raise ValueError("weights_frames and edge_mask batch sizes must match.")
    num_prototypes = weights_frames.shape[1]
    if not -num_prototypes <= gentle_index < num_prototypes:
        raise ValueError(f"gentle_index out of range: {gentle_index}.")

    frame_edges = F.adaptive_max_pool1d(
        edge_mask.unsqueeze(1), weights_frames.shape[-1]
    ).squeeze(1)
    frame_edges = (frame_edges > 0.0).to(weights_frames.dtype)
    gentle = weights_frames[:, gentle_index].clamp(1.0e-6, 1.0 - 1.0e-6)
    return torch.mean(
        -frame_edges * torch.log(gentle) - (1.0 - frame_edges) * torch.log1p(-gentle)
    )


def entropy_floor_loss(
    weights_frames: torch.Tensor, min_entropy: float = 0.15
) -> torch.Tensor:
    """Penalize blend distributions that fall below an entropy floor.

    Args:
        weights_frames: Blend weights (batch, K, frames).
        min_entropy: Minimum per-frame entropy in nats (~0.15 allows a max
            weight of roughly 0.97).

    Returns:
        Scalar loss.

    Physical Basis:
        A fully one-hot softmax is a gradient dead zone: runs 1-3 collapsed
        to a single static prototype within one epoch and never recovered.
        The floor only activates beyond ~97% purity, so decisive per-frame
        choices remain free while the optimization stays alive everywhere.
    """
    if weights_frames.dim() != 3:
        raise ValueError("weights_frames must be (batch, K, frames).")
    entropy = -torch.sum(
        weights_frames * torch.log(weights_frames.clamp_min(1e-12)), dim=1
    )
    return torch.mean(torch.relu(min_entropy - entropy))


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
