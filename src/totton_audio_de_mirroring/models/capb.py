"""CAPB: Constrained Adaptive Prototype-Blend 2x upsampler.

The model upsamples 44.1 kHz audio to 88.2 kHz as a convex, slowly
time-varying blend of fixed linear-phase interpolation prototypes. A small
waveform controller predicts per-frame blend weights; everything else is
frozen DSP. The network cannot synthesize arbitrary FIR coefficients or an
independent waveform: it only chooses a point on the sharp-vs-gentle trade-off
curve over time. Because those weights depend on the input, however, the full
system is time-varying and can create modulation sidebands unless constrained.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from totton_audio_de_mirroring.models.proto_bank import (
    DEFAULT_PROTOTYPE_SPECS,
    RELEASE_PROTOTYPE_PROFILE,
    PrototypeBank,
    build_prototype_bank,
    build_prototype_bank_for_profile,
)

DEFAULT_TARGET_SAMPLE_RATE = 88_200

DEFAULT_CONTROL_STRIDE = 64
DEFAULT_INIT_WEIGHTS = (0.85, 0.10, 0.05)
TWO_PROTOTYPE_INIT_WEIGHTS = (0.95, 0.05)
_CONTROLLER_CHANNELS = (24, 32, 40, 48, 48)
_CONTROLLER_STRIDES = (2, 2, 2, 2, 4)
_CONTROLLER_KERNEL = 9
FIRComputeDType = Literal["float32", "float64"]
SUPPORTED_FIR_COMPUTE_DTYPES: tuple[FIRComputeDType, ...] = ("float32", "float64")
SUPPORTED_CONTROLLER_FEATURE_MODES = (
    "waveform",
    "waveform_envelope",
    "envelope_flux",
    "physics_routing",
)
_LEVEL_RISK_SLOPE = 60.0
_CREST_RISK_THRESHOLD = 5.5
_CREST_RISK_SLOPE = 8.0
_SUSTAINED_DENSITY_THRESHOLD = 0.35
_SUSTAINED_DENSITY_SLOPE = 30.0
_PRIOR_RESIDUAL = 5.0e-4
_PRIOR_MIDDLE_FLOOR = 1.0e-12


@dataclass(frozen=True)
class RoutingPriorConfig:
    """Constants of the physics routing prior that travel with a checkpoint.

    Args:
        focused_gentle_fraction: Share of the sparse-impulse risk mass routed
            to the gentle prototype instead of the middle one. Zero reproduces
            the legacy middle-only policy; one removes middle from impulses.
        level_change_threshold: Smoothed relative RMS change above which a
            frame is treated as an envelope onset or offset.

    Physical Basis:
        Gentle rings least but loses impulse gain at 48 kHz (gate G5), so
        the impulse split trades near-lobe ringing against gain error per
        rate family. Stationary noise carries slow RMS wander that must stay
        below the level threshold, otherwise gentle leaks into steady frames
        and image rejection degrades.
    """

    focused_gentle_fraction: float = 0.0
    level_change_threshold: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.focused_gentle_fraction <= 1.0:
            raise ValueError("focused_gentle_fraction must lie in [0, 1].")
        if not 0.0 < self.level_change_threshold < 1.0:
            raise ValueError("level_change_threshold must lie in (0, 1).")

    def to_dict(self) -> dict[str, float]:
        """Return a JSON/torch.save friendly mapping."""
        return {
            "focused_gentle_fraction": float(self.focused_gentle_fraction),
            "level_change_threshold": float(self.level_change_threshold),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> RoutingPriorConfig:
        """Build a config from checkpoint or YAML data (legacy when absent)."""
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("routing_prior must be a mapping.")
        return cls(
            focused_gentle_fraction=float(raw.get("focused_gentle_fraction", 0.0)),
            level_change_threshold=float(raw.get("level_change_threshold", 0.15)),
        )


def _backward_difference(signal: torch.Tensor) -> torch.Tensor:
    """Return x[n] - x[n-1] with a zero first sample (ONNX-exportable form).

    Physical Basis:
        Equivalent to ``torch.diff`` with the first sample prepended; written
        with slicing so the traced graph avoids the unsupported Diff operator.
    """
    padded = torch.cat([signal[..., :1], signal], dim=-1)
    return padded[..., 1:] - padded[..., :-1]


class CAPBController(nn.Module):
    """Waveform-domain controller predicting prototype blend logits.

    Args:
        num_prototypes: Number of blend weights per frame.
        init_weights: Initial blend distribution (softmax bias init).
        dilation: Positive temporal dilation for every encoder convolution.
        feature_mode: Versioned controller input feature representation.

    Physical Basis:
        The controller only needs enough temporal context (~20-40 ms) to
        classify local signal character (steady vs transient); its output
        rate is far below the audio rate, so it cannot imprint audio-rate
        modulation on the blend.
    """

    def __init__(
        self,
        num_prototypes: int,
        init_weights: tuple[float, ...] = DEFAULT_INIT_WEIGHTS,
        dilation: int = 1,
        feature_mode: str = "waveform",
    ) -> None:
        super().__init__()
        if num_prototypes != len(init_weights):
            raise ValueError(
                "init_weights length must match num_prototypes, got "
                f"{len(init_weights)} vs {num_prototypes}."
            )
        if any(weight <= 0.0 for weight in init_weights):
            raise ValueError("init_weights must be strictly positive.")
        if dilation <= 0:
            raise ValueError("dilation must be positive.")
        if feature_mode not in SUPPORTED_CONTROLLER_FEATURE_MODES:
            raise ValueError(f"Unsupported controller feature_mode: {feature_mode!r}.")
        self.feature_mode = feature_mode

        layers: list[nn.Module] = []
        feature_channels = {
            "waveform": 1,
            "waveform_envelope": 3,
            "envelope_flux": 4,
            "physics_routing": 4,
        }
        in_channels = feature_channels[feature_mode]
        for channels, stride in zip(
            _CONTROLLER_CHANNELS, _CONTROLLER_STRIDES, strict=True
        ):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    channels,
                    kernel_size=_CONTROLLER_KERNEL,
                    stride=stride,
                    padding=dilation * (_CONTROLLER_KERNEL // 2),
                    dilation=dilation,
                )
            )
            layers.append(nn.LeakyReLU(0.1))
            in_channels = channels
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Conv1d(in_channels, num_prototypes, kernel_size=1)

        with torch.no_grad():
            self.head.weight.zero_()
            bias = self.head.bias
            if bias is None:
                raise ValueError("Controller head must have a bias term.")
            bias.copy_(torch.log(torch.tensor(init_weights, dtype=torch.float32)))
        # Freeze the static blend component: with zero-initialized head
        # weights the controller output is bias-only early in training, and
        # a trainable bias races to a one-hot static optimum where softmax
        # gradients vanish (observed as always-sharp/always-mid collapse).
        # Freezing the bias makes content-dependent modulation the only way
        # to reduce the loss.
        bias.requires_grad_(False)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Compute blend logits.

        Args:
            source: Input waveform (batch, time) at the source rate.

        Returns:
            Logits of shape (batch, num_prototypes, time // 64).
        """
        features = self.encoder(self._input_features(source))
        return torch.as_tensor(self.head(features))

    def _input_features(self, source: torch.Tensor) -> torch.Tensor:
        """Build versioned waveform and envelope controller features.

        Physical Basis:
            Raw waveform preserves polarity-changing discontinuities. A
            short absolute envelope and its lagged change expose onsets and
            decays without asking the CNN to relearn phase invariance.
        """
        waveform = source.unsqueeze(1)
        if self.feature_mode == "waveform":
            return waveform
        envelope = F.avg_pool1d(
            torch.abs(waveform), kernel_size=65, stride=1, padding=32
        )
        lag = 32
        lagged = F.pad(envelope[..., :-lag], (lag, 0), mode="replicate")
        envelope_change = torch.abs(envelope - lagged)
        if self.feature_mode in {"envelope_flux", "physics_routing"}:
            slope = torch.abs(_backward_difference(waveform))
            slope_envelope = F.avg_pool1d(slope, kernel_size=65, stride=1, padding=32)
            lagged_slope = F.pad(slope_envelope[..., :-lag], (lag, 0), mode="replicate")
            envelope_flux = envelope_change / (envelope + lagged + 1.0e-3)
            slope_flux = torch.abs(slope_envelope - lagged_slope) / (
                slope_envelope + lagged_slope + 1.0e-3
            )
            return torch.cat(
                (envelope, envelope_flux, slope_envelope, slope_flux), dim=1
            )
        return torch.cat((waveform, envelope, envelope_change), dim=1)


class CAPB(nn.Module):
    """Constrained Adaptive Prototype-Blend upsampler (44.1k -> 88.2k).

    Args:
        bank: Prototype bank (defaults to the validated Phase 0 bank).
        init_weights: Initial blend distribution.
        fir_compute_dtype: Arithmetic dtype for the fixed FIR path.
        controller_dilation: Positive temporal dilation shared by encoder
            convolutions.
        controller_feature_mode: Versioned controller input features.

    Physical Basis:
        With convex weights over gain-matched linear-phase kernels, the
        instantaneous blend response lies between the prototype responses:
        low-band content is invariant up to the prototypes' passband spread,
        group delay is exactly constant, and image suppression interpolates
        between the gentle (reference-like) and sharp (-90 dB) endpoints.
    """

    def __init__(
        self,
        bank: PrototypeBank | None = None,
        init_weights: tuple[float, ...] | None = None,
        fir_compute_dtype: FIRComputeDType = "float32",
        controller_dilation: int = 1,
        controller_feature_mode: str = "waveform",
        routing_prior: RoutingPriorConfig | None = None,
    ) -> None:
        super().__init__()
        if bank is None:
            bank = build_prototype_bank(DEFAULT_PROTOTYPE_SPECS)
        if fir_compute_dtype not in SUPPORTED_FIR_COMPUTE_DTYPES:
            raise ValueError(
                "fir_compute_dtype must be 'float32' or 'float64', got "
                f"{fir_compute_dtype!r}."
            )
        self.upsample_ratio = bank.upsample_ratio
        self.num_prototypes = len(bank.names)
        self.prototype_names = bank.names
        self.kernel_size = int(bank.kernels.shape[1])
        self.control_stride = DEFAULT_CONTROL_STRIDE
        self.prototype_profile = bank.profile_name
        self.prototype_hash = bank.coefficient_hash
        self.fir_compute_dtype = fir_compute_dtype
        if controller_dilation <= 0:
            raise ValueError("controller_dilation must be positive.")
        self.controller_dilation = controller_dilation
        if controller_feature_mode not in SUPPORTED_CONTROLLER_FEATURE_MODES:
            raise ValueError(
                f"Unsupported controller_feature_mode: {controller_feature_mode!r}."
            )
        self.controller_feature_mode = controller_feature_mode
        self.routing_prior = routing_prior or RoutingPriorConfig()

        if init_weights is None:
            init_weights = initial_weights_for_prototypes(bank.names)

        kernel_dtype = (
            torch.float32 if fir_compute_dtype == "float32" else torch.float64
        )
        kernels = torch.from_numpy(np.ascontiguousarray(bank.kernels)).to(kernel_dtype)
        self.register_buffer("kernels", kernels.unsqueeze(1))

        self.controller = CAPBController(
            self.num_prototypes,
            init_weights,
            dilation=controller_dilation,
            feature_mode=controller_feature_mode,
        )

    def forward_with_details(
        self, source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upsample and also return blend weights and prototype outputs.

        Args:
            source: Input waveform (batch, time) at the source rate.

        Returns:
            Tuple of (output, weights (B, K, frames), prototype outputs
            (B, K, time * ratio)).

        Physical Basis:
            Training losses that reference a specific prototype's behavior
            (e.g. "ripple no worse than gentle") reuse the same prototype
            convolutions instead of recomputing them.
        """
        if source.dim() != 2 or source.shape[-1] == 0:
            raise ValueError("source must be a non-empty (batch, time) tensor.")

        prototype_outputs = self._prototype_outputs(source)
        weights = self.controller_weights(source)
        weights_up = F.interpolate(
            weights,
            size=prototype_outputs.shape[-1],
            mode="linear",
            align_corners=False,
        )
        output = (weights_up.to(dtype=prototype_outputs.dtype) * prototype_outputs).sum(
            dim=1
        )
        return output, weights, prototype_outputs

    def controller_weights(self, source: torch.Tensor) -> torch.Tensor:
        """Return normalized controller weights with all routing priors applied.

        Args:
            source: Input waveform (batch, time) at the source rate.

        Returns:
            Convex prototype weights with shape (batch, prototypes, frames).

        Raises:
            ValueError: If the input is not a non-empty batched waveform.

        Physical Basis:
            Monitoring and evaluation must use the same level-independent
            routing path as waveform inference. Otherwise a physics-informed
            transient prior can be silently omitted from reported behavior.
        """
        if source.dim() != 2 or source.shape[-1] == 0:
            raise ValueError("source must be a non-empty (batch, time) tensor.")
        controller_dtype = self.controller.head.weight.dtype
        controller_source = source.to(dtype=controller_dtype)
        peak = controller_source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized_source = controller_source / peak
        logits = self.controller(normalized_source)
        if self.controller_feature_mode == "physics_routing":
            logits = self._apply_physics_routing_prior(logits, normalized_source)
        return torch.softmax(logits, dim=1)

    def focused_gentle_fraction_frames(
        self, source: torch.Tensor, frames: int
    ) -> torch.Tensor:
        """Return the per-frame gentle share of the protective routing target.

        Args:
            source: Input waveform (batch, time) at the source rate.
            frames: Controller frame count the result must match.

        Returns:
            Tensor (batch, frames) in [0, 1]; one means "gentle only".

        Raises:
            ValueError: If the input is not a non-empty batched waveform.

        Physical Basis:
            The routing loss must ask for exactly the split the prior can
            deliver: only sparse high-crest impulses are eligible for the
            middle prototype, so envelope onsets and sustained edges keep a
            gentle-only target regardless of the impulse fraction.
        """
        if source.dim() != 2 or source.shape[-1] == 0:
            raise ValueError("source must be a non-empty (batch, time) tensor.")
        if frames <= 0:
            raise ValueError("frames must be positive.")
        ones = source.new_ones((source.shape[0], frames), dtype=torch.float32)
        if self.controller_feature_mode != "physics_routing":
            return ones
        if "mid" not in self.prototype_names:
            return ones
        with torch.no_grad():
            controller_dtype = self.controller.head.weight.dtype
            controller_source = source.to(dtype=controller_dtype)
            peak = controller_source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            _, crest_risk, sustained = self._routing_prior_terms(
                controller_source / peak, frames
            )
        if sustained is None:
            return ones
        impulsive = crest_risk * (1.0 - sustained)
        gentle_share = self.routing_prior.focused_gentle_fraction
        return (1.0 - (1.0 - gentle_share) * impulsive).squeeze(1)

    def _routing_prior_terms(
        self, normalized_source: torch.Tensor, frames: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return (level_risk, crest_risk, sustained) at the controller rate."""
        waveform = normalized_source.unsqueeze(1)
        derivative = _backward_difference(waveform)
        waveform_risk = self._local_level_change(waveform)
        derivative_risk = self._local_level_change(derivative)
        level_change = torch.maximum(waveform_risk, derivative_risk)
        contextual = F.max_pool1d(level_change, kernel_size=9, stride=1, padding=4)
        crest = self._local_derivative_crest(derivative)
        crest_context = F.max_pool1d(crest, kernel_size=9, stride=1, padding=4)
        if contextual.shape[-1] != frames:
            contextual = F.adaptive_max_pool1d(contextual, frames)
            crest_context = F.adaptive_max_pool1d(crest_context, frames)
        level_risk = torch.sigmoid(
            (contextual - self.routing_prior.level_change_threshold) * _LEVEL_RISK_SLOPE
        )
        crest_risk = torch.sigmoid(
            (crest_context - _CREST_RISK_THRESHOLD) * _CREST_RISK_SLOPE
        )
        sustained: torch.Tensor | None = None
        if "mid" in self.prototype_names:
            density = self._local_activity_density(waveform)
            if density.shape[-1] != frames:
                density = F.adaptive_avg_pool1d(density, frames)
            sustained = torch.sigmoid(
                (density - _SUSTAINED_DENSITY_THRESHOLD) * _SUSTAINED_DENSITY_SLOPE
            )
        return level_risk, crest_risk, sustained

    def _apply_physics_routing_prior(
        self, logits: torch.Tensor, normalized_source: torch.Tensor
    ) -> torch.Tensor:
        """Add a noncausal transient-risk endpoint prior to learned logits.

        Physical Basis:
            Smoothed short-time RMS changes are measured for both waveform
            and derivative. This detects onset/offset energy and sparse
            discontinuities without treating a finite-duration sustain,
            every carrier cycle, or stationary noise as a transient. Centered
            smoothing and symmetric pooling supply pre-ringing look-ahead.
            Sparse impulses split their protective mass between middle and
            gentle by ``focused_gentle_fraction``; the split leaves the sharp
            share untouched so the prior remains a valid simplex.
        """
        level_risk, crest_risk, sustained = self._routing_prior_terms(
            normalized_source, logits.shape[-1]
        )
        residual = _PRIOR_RESIDUAL
        sharp_index = self.prototype_names.index("sharp")
        gentle_index = self.prototype_names.index("gentle")
        channels: dict[int, torch.Tensor] = {}
        if sustained is not None:
            middle_index = self.prototype_names.index("mid")
            middle_floor = _PRIOR_MIDDLE_FLOOR
            routable_mass = 1.0 - 2.0 * residual - middle_floor
            impulsive = crest_risk * (1.0 - sustained)
            gentle_share = self.routing_prior.focused_gentle_fraction
            middle_risk = (1.0 - gentle_share) * impulsive
            gentle_risk = gentle_share * impulsive + crest_risk * sustained
            gentle_risk = gentle_risk + (1.0 - crest_risk) * level_risk
            middle = middle_floor + routable_mass * middle_risk
            gentle = residual + routable_mass * gentle_risk
            sharp = 1.0 - middle - gentle
            channels[middle_index] = middle
        else:
            risk = torch.maximum(level_risk, crest_risk)
            gentle = residual + (1.0 - 2.0 * residual) * risk
            sharp = 1.0 - gentle
        channels[sharp_index] = sharp
        channels[gentle_index] = gentle
        # Concatenate per-prototype channels instead of in-place index writes so
        # the ONNX trace stays a plain Concat graph.
        probabilities = torch.cat(
            [
                channels.get(index, torch.full_like(sharp, residual))
                for index in range(self.num_prototypes)
            ],
            dim=1,
        )
        bias = self.controller.head.bias
        if bias is None:
            raise RuntimeError("Controller head must retain its fixed bias.")
        return logits - bias.view(1, -1, 1) + torch.log(probabilities)

    @staticmethod
    def _local_level_change(source: torch.Tensor) -> torch.Tensor:
        """Measure coherent changes in a smoothed short-time RMS envelope.

        Physical Basis:
            Ringing is exposed around coherent changes in local energy. RMS
            aggregation and roughly 25 ms smoothing reject stochastic carrier
            variation. A symmetric past/future difference aligns the risk
            peak to the event for equal pre- and post-ringing protection.
        """
        level = torch.sqrt(
            F.avg_pool1d(
                source.square(),
                kernel_size=257,
                stride=DEFAULT_CONTROL_STRIDE,
                padding=128,
                count_include_pad=False,
            ).clamp_min(1.0e-12)
        )
        smoothed = F.avg_pool1d(
            level,
            kernel_size=17,
            stride=1,
            padding=8,
            count_include_pad=False,
        )
        lag = 4
        previous = F.pad(smoothed[..., :-lag], (lag, 0), mode="replicate")
        following = F.pad(smoothed[..., lag:], (0, lag), mode="replicate")
        return torch.abs(following - previous) / (following + previous + 1.0e-2)

    @staticmethod
    def _local_derivative_crest(derivative: torch.Tensor) -> torch.Tensor:
        """Return frame-rate derivative crest factor.

        Physical Basis:
            Isolated steps and impulse trains have a derivative peak far
            above their local derivative RMS. Smooth tones and statistically
            stationary noise retain a low crest factor, so this closes the
            periodic-edge blind spot without a frequency or 20 kHz split.
        """
        peak = F.max_pool1d(
            torch.abs(derivative),
            kernel_size=257,
            stride=DEFAULT_CONTROL_STRIDE,
            padding=128,
        )
        rms = torch.sqrt(
            F.avg_pool1d(
                derivative.square(),
                kernel_size=257,
                stride=DEFAULT_CONTROL_STRIDE,
                padding=128,
                count_include_pad=False,
            ).clamp_min(1.0e-12)
        )
        return peak / (rms + 1.0e-2)

    @staticmethod
    def _local_activity_density(source: torch.Tensor) -> torch.Tensor:
        """Measure local waveform RMS relative to its peak.

        Physical Basis:
            A sustained plateau edge occupies most samples in a short window,
            whereas an impulse or sparse click train occupies very few. This
            separates square-like ringing risk from gain-sensitive impulses
            even when both share the same edge repetition interval.
        """
        peak = F.max_pool1d(
            torch.abs(source),
            kernel_size=257,
            stride=DEFAULT_CONTROL_STRIDE,
            padding=128,
        )
        rms = torch.sqrt(
            F.avg_pool1d(
                source.square(),
                kernel_size=257,
                stride=DEFAULT_CONTROL_STRIDE,
                padding=128,
                count_include_pad=False,
            ).clamp_min(1.0e-12)
        )
        return rms / (peak + 1.0e-2)

    def forward(
        self, source: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Upsample the source waveform.

        Args:
            source: Input waveform (batch, time) at the source rate.
            return_weights: Also return per-frame blend weights.

        Returns:
            Output waveform (batch, time * ratio), optionally with weights
            of shape (batch, num_prototypes, frames).

        Raises:
            ValueError: If the input shape is invalid.

        Physical Basis:
            Prototype outputs are computed once per kernel; the blend is a
            pointwise convex combination, so the output at every instant is
            the response of SOME valid linear-phase interpolation filter.
        """
        # Peak-normalize the controller input (inside forward_with_details):
        # the correct interpolation choice is a property of signal SHAPE,
        # not level, and run6 showed the controller otherwise learns an
        # amplitude shortcut that fails on out-of-distribution probe levels.
        # The blend still applies to the unnormalized prototype outputs, so
        # this normalization does not change output level. Input-dependent
        # weights still make the overall system time-varying/nonlinear.
        output, weights, _ = self.forward_with_details(source)
        if return_weights:
            return output, weights
        return output

    def _prototype_outputs(self, source: torch.Tensor) -> torch.Tensor:
        """Run the fixed prototype interpolators; shape (B, K, T*ratio).

        Physical Basis:
            Zero-stuffing followed by centered same-padding convolution with
            the linear-phase kernels keeps every prototype output aligned to
            the zero-stuffed timeline (constant group delay compensated).
        """
        batch, time = source.shape
        kernels = torch.as_tensor(self.kernels)
        fir_source = source.to(dtype=kernels.dtype)
        stuffed = fir_source.new_zeros(batch, 1, time * self.upsample_ratio)
        stuffed[:, 0, :: self.upsample_ratio] = fir_source
        return F.conv1d(stuffed, kernels, padding=self.kernel_size // 2)

    def mean_weights(self, source: torch.Tensor) -> torch.Tensor:
        """Return the batch-mean blend weights (monitoring helper).

        Physical Basis:
            A collapse to always-sharp (or always-gentle) is visible
            immediately in the mean weight statistics during training.
        """
        with torch.no_grad():
            weights = self.controller_weights(source)
        return weights.mean(dim=(0, 2))


def initial_weights_for_prototypes(names: tuple[str, ...]) -> tuple[float, ...]:
    """Return the fixed controller prior for a supported prototype topology.

    Args:
        names: Ordered prototype names from the fixed FIR bank.

    Returns:
        Positive convex initial weights in the same order.

    Raises:
        ValueError: If the topology is unsupported.

    Physical Basis:
        Sharp is the stationary-signal default. Removing an unused middle
        prototype transfers its prior mass to sharp while retaining the same
        gentle safety prior at discontinuities.
    """
    if names == ("sharp", "mid", "gentle"):
        return DEFAULT_INIT_WEIGHTS
    if names == ("sharp", "gentle"):
        return TWO_PROTOTYPE_INIT_WEIGHTS
    raise ValueError(f"Unsupported CAPB prototype topology: {names!r}.")


def capb_from_checkpoint(checkpoint: dict[str, Any]) -> CAPB:
    """Build a CAPB with the rate-correct bank and load its weights.

    Args:
        checkpoint: Loaded checkpoint dictionary (torch.load result).

    Returns:
        CAPB model in eval mode with the checkpoint weights applied.

    Raises:
        ValueError: If the checkpoint rate metadata is inconsistent.
        RuntimeError: If the checkpoint is missing model_state.

    Physical Basis:
        The prototype bank is deterministic DSP rebuilt from its rate
        family's design specs; only the controller weights come from the
        checkpoint. Loading a 48k checkpoint into the default 44.1k bank
        would silently pair the controller with wrong kernels, so the bank
        must follow the checkpoint's target_sample_rate.
    """
    target_rate = int(checkpoint.get("target_sample_rate", DEFAULT_TARGET_SAMPLE_RATE))
    expected_input_rate = checkpoint.get("expected_input_rate")
    profile_name = str(checkpoint.get("prototype_profile", RELEASE_PROTOTYPE_PROFILE))
    fir_compute_dtype = _checkpoint_fir_dtype(checkpoint)
    bank = build_prototype_bank_for_profile(target_rate, profile_name)
    if (
        expected_input_rate is not None
        and int(expected_input_rate) * bank.upsample_ratio != target_rate
    ):
        raise ValueError(
            f"Checkpoint expected_input_rate {expected_input_rate} Hz is "
            f"inconsistent with target_sample_rate {target_rate} Hz."
        )
    expected_hash = checkpoint.get("prototype_hash")
    if expected_hash is not None and str(expected_hash) != bank.coefficient_hash:
        raise ValueError(
            "Checkpoint prototype_hash does not match the reconstructed "
            f"'{profile_name}' bank."
        )
    controller_dilation = int(checkpoint.get("controller_dilation", 1))
    controller_feature_mode = str(checkpoint.get("controller_feature_mode", "waveform"))
    model = CAPB(
        bank=bank,
        fir_compute_dtype=fir_compute_dtype,
        controller_dilation=controller_dilation,
        controller_feature_mode=controller_feature_mode,
        routing_prior=RoutingPriorConfig.from_mapping(checkpoint.get("routing_prior")),
    )
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise RuntimeError("Invalid checkpoint: model_state is missing.")
    if "prototype_profile" in checkpoint:
        _load_profiled_model_state(model, model_state)
    else:
        model.load_state_dict(model_state)
    model.eval()
    return model


def capb_candidate_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    prototype_profile: str,
    fir_compute_dtype: FIRComputeDType = "float32",
    controller_dilation: int | None = None,
    controller_feature_mode: str | None = None,
    routing_prior: RoutingPriorConfig | None = None,
) -> CAPB:
    """Pair an existing controller with an experimental prototype profile.

    Args:
        checkpoint: Source checkpoint containing validated controller weights.
        prototype_profile: Explicit experimental prototype profile.
        fir_compute_dtype: Arithmetic dtype for fixed FIR convolution.
        controller_dilation: Optional controller dilation override. When
            omitted, preserve checkpoint metadata (legacy checkpoints use 1).
        controller_feature_mode: Optional feature-mode override. When omitted,
            preserve checkpoint metadata (legacy checkpoints use waveform).
        routing_prior: Optional routing-prior override. When omitted, preserve
            checkpoint metadata (legacy checkpoints use the middle-only policy).

    Returns:
        Evaluation-only CAPB candidate in eval mode.

    Physical Basis:
        Controller-only transfer isolates the acoustic effect of changing the
        fixed prototype bank. Saved kernel buffers must not overwrite the
        named experimental coefficients.
    """
    target_rate = int(checkpoint.get("target_sample_rate", DEFAULT_TARGET_SAMPLE_RATE))
    bank = build_prototype_bank_for_profile(target_rate, prototype_profile)
    expected_input_rate = checkpoint.get("expected_input_rate")
    if (
        expected_input_rate is not None
        and int(expected_input_rate) * bank.upsample_ratio != target_rate
    ):
        raise ValueError(
            f"Checkpoint expected_input_rate {expected_input_rate} Hz is "
            f"inconsistent with target_sample_rate {target_rate} Hz."
        )
    dilation = (
        int(checkpoint.get("controller_dilation", 1))
        if controller_dilation is None
        else controller_dilation
    )
    feature_mode = (
        str(checkpoint.get("controller_feature_mode", "waveform"))
        if controller_feature_mode is None
        else controller_feature_mode
    )
    model = CAPB(
        bank=bank,
        fir_compute_dtype=fir_compute_dtype,
        controller_dilation=dilation,
        controller_feature_mode=feature_mode,
        routing_prior=(
            RoutingPriorConfig.from_mapping(checkpoint.get("routing_prior"))
            if routing_prior is None
            else routing_prior
        ),
    )
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise RuntimeError("Invalid checkpoint: model_state is missing.")
    _load_controller_state(model, model_state)
    model.eval()
    return model


def _checkpoint_fir_dtype(checkpoint: dict[str, Any]) -> FIRComputeDType:
    value = str(checkpoint.get("fir_compute_dtype", "float32"))
    if value not in SUPPORTED_FIR_COMPUTE_DTYPES:
        raise ValueError(f"Unsupported checkpoint fir_compute_dtype: {value!r}.")
    return cast(FIRComputeDType, value)


def _load_profiled_model_state(model: CAPB, model_state: dict[str, Any]) -> None:
    state_without_kernels = {
        name: value for name, value in model_state.items() if name != "kernels"
    }
    incompatible = model.load_state_dict(state_without_kernels, strict=False)
    if incompatible.missing_keys != ["kernels"] or incompatible.unexpected_keys:
        raise RuntimeError(
            "Invalid profiled checkpoint model_state: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )


def _load_controller_state(model: CAPB, model_state: dict[str, Any]) -> None:
    prefix = "controller."
    controller_state = {
        name.removeprefix(prefix): value
        for name, value in model_state.items()
        if name.startswith(prefix)
    }
    if not controller_state:
        raise RuntimeError("Checkpoint contains no controller state.")
    model.controller.load_state_dict(controller_state)
