"""Loss terms for CAPB training (gate-aligned, mask-driven).

The fidelity terms pull toward the band-limited teacher; the ringing terms
use the dataset's clean-signal masks to penalize exactly what the gates
measure: high-frequency ripple on plateaus and any energy in silences.
The TV term keeps the blend trajectory slow and decisive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from totton_audio_de_mirroring.training.stft_loss import (
    STFTLossConfig,
    multi_resolution_stft_loss,
)

_EPS = 1e-8
_ROUTING_TARGET_PROBABILITY = 0.999
_ROUTING_TOP_FRACTION = 0.25


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
        stationary_modulation: Penalty for time-varying blends on stationary
            signals relative to the same signal's time-mean fixed blend.
        edge_fidelity_relax: Fraction by which waveform fidelity is relaxed
            inside edge zones (0.9 keeps 10% of the wave loss there).
        edge_ring: Penalty for edge-zone ripple exceeding the gentle
            prototype's ripple (covers dense-edge content with no plateaus).
        pre_echo_excess: Penalty for gate-window energy exceeding the gentle
            prototype on focused transient samples.
        post_echo_excess: Equivalent penalty after an impulse or burst end.
        prototype_routing: Label-supervised prototype-role routing penalty.
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
    stationary_modulation: float = 0.0
    edge_fidelity_relax: float = 0.9
    edge_ring: float = 300.0
    pre_echo_excess: float = 0.0
    post_echo_excess: float = 0.0
    prototype_routing: float = 0.0
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
    gentle_output: torch.Tensor | None = None,
    prototype_outputs: torch.Tensor | None = None,
    stationary: torch.Tensor | None = None,
    pre_echo_mask: torch.Tensor | None = None,
    post_echo_mask: torch.Tensor | None = None,
    far_pre_echo_mask: torch.Tensor | None = None,
    safe_active_mask: torch.Tensor | None = None,
    focused_transient: torch.Tensor | None = None,
    sharp_index: int = 0,
    gentle_index: int = -1,
    focused_gentle_fraction: float | torch.Tensor = 0.0,
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
        gentle_output: The gentle prototype's output for the same input;
            enables the edge_ring loss when given with edge_mask.
        prototype_outputs: Fixed prototype outputs, shaped (batch, K, time).
        stationary: Boolean batch flags selecting stationary signals.
        pre_echo_mask: Gate-aligned mask immediately before focused events.
        post_echo_mask: Gate-aligned mask after focused event ends.
        far_pre_echo_mask: Supplemental 4--12 ms pre-event mask for long FIR
            tails. It uses the same excess-energy coefficient as G2b.
        safe_active_mask: Active samples outside every labelled risk window.
        focused_transient: Boolean batch flags selecting focused click and
            tone-burst examples governed by the continuous pre-echo loss.
        sharp_index: Prototype index used for stationary non-edge frames.
        gentle_index: Prototype index used for non-focused edge frames.
        focused_gentle_fraction: Gentle share of the focused-risk routing
            target, scalar or per-frame (batch, frames); the remainder goes
            to the middle prototype when one exists.

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
    if prototype_outputs is not None and stationary is not None:
        losses["stationary_modulation"] = stationary_modulation_loss(
            output,
            prototype_outputs,
            weights_frames,
            stationary,
            trim=trim,
        )
    else:
        losses["stationary_modulation"] = output.new_zeros(())
    if edge_mask is not None and gentle_output is not None:
        losses["edge_ring"] = edge_ring_loss(
            out, gentle_output[:, sl], edge_mask[:, sl]
        )
    else:
        losses["edge_ring"] = output.new_zeros(())
    if pre_echo_mask is not None and gentle_output is not None:
        losses["pre_echo_excess"] = pre_echo_excess_loss(
            out, gentle_output[:, sl], pre_echo_mask[:, sl]
        )
    else:
        losses["pre_echo_excess"] = output.new_zeros(())
    if post_echo_mask is not None and gentle_output is not None:
        losses["post_echo_excess"] = pre_echo_excess_loss(
            out, gentle_output[:, sl], post_echo_mask[:, sl]
        )
    else:
        losses["post_echo_excess"] = output.new_zeros(())
    if far_pre_echo_mask is not None:
        if far_pre_echo_mask.shape != output.shape:
            raise ValueError("far_pre_echo_mask must share the output shape.")
        if gentle_output is not None:
            losses["pre_echo_excess"] = losses[
                "pre_echo_excess"
            ] + pre_echo_excess_loss(
                out,
                gentle_output[:, sl],
                far_pre_echo_mask[:, sl],
            )
    if edge_mask is not None and stationary is not None:
        routing_mask = edge_mask
        focused_risk_mask = edge_mask.clone()
        if pre_echo_mask is not None:
            routing_mask = torch.maximum(routing_mask, pre_echo_mask)
            focused_risk_mask = torch.maximum(focused_risk_mask, pre_echo_mask)
        if post_echo_mask is not None:
            routing_mask = torch.maximum(routing_mask, post_echo_mask)
            focused_risk_mask = torch.maximum(focused_risk_mask, post_echo_mask)
        if far_pre_echo_mask is not None:
            routing_mask = torch.maximum(routing_mask, far_pre_echo_mask)
            focused_risk_mask = torch.maximum(focused_risk_mask, far_pre_echo_mask)
        losses["prototype_routing"] = prototype_routing_loss(
            weights_frames,
            routing_mask,
            stationary,
            focused_transient=focused_transient,
            focused_risk_mask=focused_risk_mask,
            safe_active_mask=safe_active_mask,
            sharp_index=sharp_index,
            gentle_index=gentle_index,
            focused_gentle_fraction=focused_gentle_fraction,
        )
    else:
        losses["prototype_routing"] = output.new_zeros(())
    losses["total"] = (
        loss_weights.wave * losses["wave"]
        + loss_weights.stft * losses["stft"]
        + loss_weights.plateau * losses["plateau"]
        + loss_weights.quiet * losses["quiet"]
        + loss_weights.tv * losses["tv"]
        + loss_weights.entropy_floor * losses["entropy_floor"]
        + loss_weights.stationary_modulation * losses["stationary_modulation"]
        + loss_weights.edge_ring * losses["edge_ring"]
        + loss_weights.pre_echo_excess * losses["pre_echo_excess"]
        + loss_weights.post_echo_excess * losses["post_echo_excess"]
        + loss_weights.prototype_routing * losses["prototype_routing"]
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


def pre_echo_excess_loss(
    output: torch.Tensor,
    gentle_output: torch.Tensor,
    pre_echo_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize pre-echo energy above the gentle prototype per event.

    Args:
        output: Model output (batch, time).
        gentle_output: Gentle prototype output for the same source.
        pre_echo_mask: G2b-aligned pre-event mask (batch, time).

    Returns:
        Peak-normalized mean excess energy over applicable batch items.

    Physical Basis:
        Legitimate augmented noise can produce non-zero output in a nominally
        quiet interval. Comparing the same source through the low-ringing
        gentle endpoint avoids treating that noise as hallucination, while
        still driving the adaptive blend no worse than the validated endpoint.
    """
    _validate_pair(output, pre_echo_mask)
    if gentle_output.shape != output.shape:
        raise ValueError("gentle_output must share the output shape.")
    counts = torch.sum(pre_echo_mask, dim=1)
    applicable = counts > 0.0
    if not bool(torch.any(applicable)):
        return output.new_zeros(())
    denominator = counts.clamp_min(1.0)
    output_energy = torch.sum(torch.square(output) * pre_echo_mask, dim=1) / denominator
    gentle_energy = (
        torch.sum(torch.square(gentle_output) * pre_echo_mask, dim=1) / denominator
    )
    peak_energy = torch.square(torch.amax(torch.abs(gentle_output), dim=1)).clamp_min(
        _EPS
    )
    excess = torch.relu(output_energy - gentle_energy) / peak_energy
    return torch.mean(excess[applicable])


def prototype_routing_loss(
    weights_frames: torch.Tensor,
    edge_mask: torch.Tensor,
    stationary: torch.Tensor,
    *,
    focused_transient: torch.Tensor | None = None,
    focused_risk_mask: torch.Tensor | None = None,
    safe_active_mask: torch.Tensor | None = None,
    sharp_index: int,
    gentle_index: int,
    focused_gentle_fraction: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Teach physically labelled sharp-versus-gentle routing.

    Args:
        weights_frames: Convex prototype weights (batch, K, frames).
        edge_mask: Target-rate edge and pre-echo mask (batch, time).
        stationary: Boolean batch flags for stationary signal families.
        focused_transient: Optional boolean flags for focused transient
            examples. Only their explicit pre/post risk windows are routed.
        focused_risk_mask: Target-rate pre/post echo mask for focused events.
        safe_active_mask: Active non-risk samples eligible for sharp routing.
        sharp_index: Prototype index for stationary non-edge frames.
        gentle_index: Prototype index for edge frames.
        focused_gentle_fraction: Gentle share of the focused-risk target as a
            scalar or a (batch, frames) tensor; the remainder is assigned to
            the middle prototype when three prototypes are present.

    Returns:
        Worst-quartile mean negative log weight over labelled frames.

    Raises:
        ValueError: If shapes or the gentle fraction are invalid.

    Physical Basis:
        Procedural data provides reliable labels unavailable in arbitrary
        audio: stationary content uses sharp, sustained edges use gentle,
        and focused sparse transients split between gentle (least ringing)
        and the middle compensation prototype (preserves 48 kHz impulse gain
        within G5). The split follows the model's routing prior so the loss
        never demands a blend the prior forbids. Worst-quartile mining
        mirrors worst-probe selection.
    """
    if weights_frames.dim() != 3:
        raise ValueError("weights_frames must be (batch, K, frames).")
    if edge_mask.dim() != 2 or edge_mask.shape[0] != weights_frames.shape[0]:
        raise ValueError("edge_mask must be a matching (batch, time) tensor.")
    if stationary.dim() != 1 or stationary.shape[0] != weights_frames.shape[0]:
        raise ValueError("stationary must be a batch-length vector.")
    if focused_transient is not None and (
        focused_transient.dim() != 1
        or focused_transient.shape[0] != weights_frames.shape[0]
    ):
        raise ValueError("focused_transient must be a batch-length vector.")
    if focused_risk_mask is not None and focused_risk_mask.shape != edge_mask.shape:
        raise ValueError("focused_risk_mask must share edge_mask shape.")
    if safe_active_mask is not None and safe_active_mask.shape != edge_mask.shape:
        raise ValueError("safe_active_mask must share edge_mask shape.")
    prototype_count = weights_frames.shape[1]
    if prototype_count < 2:
        raise ValueError("prototype routing requires at least two prototypes.")
    if not -prototype_count <= gentle_index < prototype_count:
        raise ValueError("gentle_index is out of range.")
    if not 0 <= sharp_index < prototype_count:
        raise ValueError("sharp_index is out of range.")
    frame_edges = F.adaptive_max_pool1d(
        edge_mask.unsqueeze(1), weights_frames.shape[-1]
    ).squeeze(1)
    frame_edges = torch.clamp(frame_edges, 0.0, 1.0)
    stationary_frames = (
        stationary.to(weights_frames.dtype).unsqueeze(1).expand_as(frame_edges)
    )
    focused_frames = torch.zeros_like(frame_edges)
    frame_focused_risk = torch.zeros_like(frame_edges)
    if focused_transient is not None:
        focused_frames = focused_transient.to(weights_frames.dtype).unsqueeze(1)
    if focused_risk_mask is not None:
        frame_focused_risk = F.adaptive_max_pool1d(
            focused_risk_mask.unsqueeze(1), weights_frames.shape[-1]
        ).squeeze(1)
    gentle_mask = frame_edges * (1.0 - focused_frames)
    focused_risk = torch.clamp(frame_focused_risk * focused_frames, 0.0, 1.0)
    if prototype_count == 2:
        gentle_mask = torch.clamp(gentle_mask + focused_risk, 0.0, 1.0)
    sharp_mask = stationary_frames * (1.0 - frame_edges)
    if safe_active_mask is not None:
        frame_safe = F.adaptive_max_pool1d(
            safe_active_mask.unsqueeze(1), weights_frames.shape[-1]
        ).squeeze(1)
        sharp_mask = frame_safe * (1.0 - frame_edges)
    log_floor = torch.finfo(weights_frames.dtype).tiny
    target_probability = _ROUTING_TARGET_PROBABILITY
    other_probability = (1.0 - target_probability) / (prototype_count - 1)
    sharp_targets = torch.full_like(weights_frames, other_probability)
    gentle_targets = torch.full_like(weights_frames, other_probability)
    sharp_targets[:, sharp_index, :] = target_probability
    gentle_targets[:, gentle_index, :] = target_probability
    log_weights = torch.log(weights_frames.clamp_min(log_floor))
    sharp_loss = -torch.sum(sharp_targets * log_weights, dim=1)
    gentle_loss = -torch.sum(gentle_targets * log_weights, dim=1)
    sharp_counts = torch.sum(sharp_mask, dim=1)
    gentle_counts = torch.sum(gentle_mask, dim=1)
    sharp_mean = torch.sum(sharp_loss * sharp_mask, dim=1) / sharp_counts.clamp_min(1.0)
    gentle_mean = torch.sum(gentle_loss * gentle_mask, dim=1) / gentle_counts.clamp_min(
        1.0
    )
    class_count = (sharp_counts > 0.0).to(weights_frames.dtype)
    class_count += (gentle_counts > 0.0).to(weights_frames.dtype)
    middle_mean = torch.zeros_like(sharp_mean)
    if prototype_count > 2:
        normalized_gentle_index = gentle_index % prototype_count
        middle_index = next(
            index
            for index in range(prototype_count)
            if index not in {sharp_index, normalized_gentle_index}
        )
        middle_targets = torch.full_like(weights_frames, other_probability)
        middle_targets[:, middle_index, :] = target_probability
        gentle_share = _focused_gentle_share(focused_gentle_fraction, weights_frames)
        focused_targets = (
            gentle_share * gentle_targets + (1.0 - gentle_share) * middle_targets
        )
        middle_loss = -torch.sum(focused_targets * log_weights, dim=1)
        middle_counts = torch.sum(focused_risk, dim=1)
        middle_mean = torch.sum(
            middle_loss * focused_risk, dim=1
        ) / middle_counts.clamp_min(1.0)
        class_count += (middle_counts > 0.0).to(weights_frames.dtype)
    applicable = class_count > 0.0
    if not bool(torch.any(applicable)):
        return weights_frames.new_zeros(())
    per_sample = (sharp_mean + gentle_mean + middle_mean) / class_count.clamp_min(1.0)
    applicable_losses = per_sample[applicable]
    top_count = max(1, math.ceil(_ROUTING_TOP_FRACTION * applicable_losses.numel()))
    return torch.mean(torch.topk(applicable_losses, top_count).values)


def _focused_gentle_share(
    value: float | torch.Tensor, weights_frames: torch.Tensor
) -> torch.Tensor:
    """Broadcast the focused gentle fraction to (batch, 1, frames)."""
    if isinstance(value, torch.Tensor):
        expected = (weights_frames.shape[0], weights_frames.shape[-1])
        if tuple(value.shape) != expected:
            raise ValueError("focused_gentle_fraction tensor must be (batch, frames).")
        share = value.to(dtype=weights_frames.dtype, device=weights_frames.device)
        return torch.clamp(share, 0.0, 1.0).unsqueeze(1)
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("focused_gentle_fraction must lie in [0, 1].")
    return weights_frames.new_full((1, 1, 1), float(value))


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


def stationary_modulation_loss(
    output: torch.Tensor,
    prototype_outputs: torch.Tensor,
    weights_frames: torch.Tensor,
    stationary: torch.Tensor,
    trim: int = 512,
) -> torch.Tensor:
    """Penalize signal-dependent blend modulation on stationary inputs.

    Args:
        output: Time-varying CAPB output, shaped (batch, time).
        prototype_outputs: Fixed FIR outputs, shaped (batch, K, time).
        weights_frames: Convex blend weights, shaped (batch, K, frames).
        stationary: Boolean flags, shaped (batch,), selecting applicable input.
        trim: Samples excluded at each output border.

    Returns:
        Sum of the relative output deviation and mean blend-weight deviation
        from the time-mean fixed blend.

    Raises:
        ValueError: If shapes or trim are invalid.

    Physical Basis:
        Multiplying prototype outputs by signal-synchronous weights turns the
        overall path into a modulator and creates sidebands. A stationary
        input should use one fixed convex blend; comparing with the same
        signal's time-mean blend removes modulation without preferring a
        particular prototype or restricting transient behavior. The direct
        weight term remains observable where adjacent prototype outputs are
        locally similar, preventing small controller oscillations from hiding
        behind a weak output-domain gradient and later producing IMD
        sidebands on a different stationary signal.
    """
    if prototype_outputs.dim() != 3 or prototype_outputs.shape[0] != output.shape[0]:
        raise ValueError("prototype_outputs must be (batch, K, time).")
    if prototype_outputs.shape[1] != weights_frames.shape[1]:
        raise ValueError("prototype and weight counts must match.")
    if prototype_outputs.shape[-1] != output.shape[-1]:
        raise ValueError("prototype_outputs and output time lengths must match.")
    if stationary.dim() != 1 or stationary.shape[0] != output.shape[0]:
        raise ValueError("stationary must be a batch-length vector.")
    if trim < 0 or 2 * trim >= output.shape[-1]:
        raise ValueError("trim must be non-negative and smaller than half length.")
    selected = stationary.to(dtype=torch.bool, device=output.device)
    if not bool(torch.any(selected)):
        return output.new_zeros(())
    mean_weights = torch.mean(weights_frames, dim=-1, keepdim=True)
    fixed_output = torch.sum(mean_weights * prototype_outputs, dim=1)
    sl = slice(trim, output.shape[-1] - trim) if trim else slice(None)
    output_difference = torch.mean(
        torch.abs(output[selected, sl] - fixed_output[selected, sl]), dim=1
    )
    reference = torch.mean(torch.abs(fixed_output[selected, sl]), dim=1).clamp_min(_EPS)
    weight_difference = torch.mean(
        torch.abs(weights_frames[selected] - mean_weights[selected])
    )
    return torch.mean(output_difference / reference) + weight_difference


def _validate_pair(output: torch.Tensor, mask: torch.Tensor) -> None:
    if output.dim() != 2 or output.shape != mask.shape:
        raise ValueError("output and mask must be matching (batch, time) tensors.")
