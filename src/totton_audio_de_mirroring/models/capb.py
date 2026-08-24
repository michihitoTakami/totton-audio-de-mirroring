"""CAPB: Constrained Adaptive Prototype-Blend 2x upsampler.

The model upsamples 44.1 kHz audio to 88.2 kHz as a convex, slowly
time-varying blend of fixed linear-phase interpolation prototypes. A small
waveform controller predicts per-frame blend weights; everything else is
frozen DSP. The network therefore cannot create frequencies, cannot change
gain, and cannot ring beyond the worst fixed prototype - it only chooses a
point on the sharp-vs-gentle trade-off curve over time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from totton_audio_de_mirroring.models.proto_bank import (
    DEFAULT_PROTOTYPE_SPECS,
    PrototypeBank,
    build_prototype_bank,
    prototype_specs_for_target_rate,
)

DEFAULT_TARGET_SAMPLE_RATE = 88_200

DEFAULT_CONTROL_STRIDE = 64
DEFAULT_INIT_WEIGHTS = (0.85, 0.10, 0.05)
_CONTROLLER_CHANNELS = (24, 32, 40, 48, 48)
_CONTROLLER_STRIDES = (2, 2, 2, 2, 4)
_CONTROLLER_KERNEL = 9
_TRANSIENT_RMS_WINDOW = 257
_TRANSIENT_CREST_START = 6.0
_TRANSIENT_CREST_FULL = 12.0
_TRANSIENT_FRAME_DILATION = 5
_CONTROLLER_CONTEXT_PAD = 128
_DISCONTINUITY_WINDOW = 257
_DISCONTINUITY_FLAT_EPSILON = 1.0e-4
_DISCONTINUITY_FLAT_START = 0.70
_DISCONTINUITY_FLAT_FULL = 0.90
_DISCONTINUITY_SLOPE_START = 0.25
_DISCONTINUITY_SLOPE_FULL = 0.50


class CAPBController(nn.Module):
    """Waveform-domain controller predicting prototype blend logits.

    Args:
        num_prototypes: Number of blend weights per frame.
        init_weights: Initial blend distribution (softmax bias init).

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
    ) -> None:
        super().__init__()
        if num_prototypes != len(init_weights):
            raise ValueError(
                "init_weights length must match num_prototypes, got "
                f"{len(init_weights)} vs {num_prototypes}."
            )
        if any(weight <= 0.0 for weight in init_weights):
            raise ValueError("init_weights must be strictly positive.")

        layers: list[nn.Module] = []
        in_channels = 1
        for channels, stride in zip(
            _CONTROLLER_CHANNELS, _CONTROLLER_STRIDES, strict=True
        ):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    channels,
                    kernel_size=_CONTROLLER_KERNEL,
                    stride=stride,
                    padding=_CONTROLLER_KERNEL // 2,
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
        features = self.encoder(source.unsqueeze(1))
        return torch.as_tensor(self.head(features))


class CAPB(nn.Module):
    """Constrained Adaptive Prototype-Blend upsampler (44.1k -> 88.2k).

    Args:
        bank: Prototype bank (defaults to the validated Phase 0 bank).
        init_weights: Initial blend distribution.

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
        init_weights: tuple[float, ...] = DEFAULT_INIT_WEIGHTS,
    ) -> None:
        super().__init__()
        if bank is None:
            bank = build_prototype_bank(DEFAULT_PROTOTYPE_SPECS)
        self.upsample_ratio = bank.upsample_ratio
        self.target_sample_rate = bank.sample_rate
        self.num_prototypes = len(bank.names)
        self.prototype_names = bank.names
        self.kernel_size = int(bank.kernels.shape[1])
        self.control_stride = DEFAULT_CONTROL_STRIDE

        kernels = torch.from_numpy(
            np.ascontiguousarray(bank.kernels, dtype=np.float64)
        ).to(torch.float32)
        self.register_buffer("kernels", kernels.unsqueeze(1))

        self.controller = CAPBController(self.num_prototypes, init_weights)

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
        weights = self.blend_weights(source)
        weights_up = F.interpolate(
            weights,
            size=prototype_outputs.shape[-1],
            mode="linear",
            align_corners=False,
        )
        output = (weights_up * prototype_outputs).sum(dim=1)
        return output, weights, prototype_outputs

    def blend_weights(self, source: torch.Tensor) -> torch.Tensor:
        """Return convex weights with deterministic sparse-event safety.

        Args:
            source: Input waveform (batch, time) at the source rate.

        Returns:
            Guarded blend weights (batch, prototypes, control frames).

        Physical Basis:
            A single-sample impulse is too sparse for corpus-averaged losses
            to classify reliably, yet the sharp FIR's symmetric support can
            create measurable pre-echo. A local crest-factor guard detects
            only sparse high-peak events and biases their surrounding frames
            toward the rate-family prototype that satisfies both pre-echo and
            gain gates. Stationary tones, squares, and Gaussian noise remain
            below the crest threshold. A separate discontinuity-density guard
            sends square/step edges to gentle without selecting Gaussian noise.
        """
        peak = source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = source / peak
        logits = self._context_padded_controller(normalized)
        weights = torch.softmax(logits, dim=1)
        transient_score = self._sparse_transient_score(normalized, logits.shape[-1])
        discontinuity_score = self._discontinuity_score(normalized, logits.shape[-1])
        gentle_score = discontinuity_score
        gentle_index = self.prototype_names.index("gentle")
        gentle = torch.zeros_like(weights)
        gentle[:, gentle_index] = 1.0
        weights = weights * (
            1.0 - gentle_score.unsqueeze(1)
        ) + gentle * gentle_score.unsqueeze(1)
        return self._apply_sparse_transient_guard(weights, transient_score)

    def _apply_sparse_transient_guard(
        self, weights: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        """Blend sparse events toward the rate-family-safe prototype."""
        safe_name = "mid" if self.target_sample_rate == 96_000 else "gentle"
        safe = torch.zeros_like(weights)
        safe[:, self.prototype_names.index(safe_name)] = 1.0
        return weights * (1.0 - score.unsqueeze(1)) + safe * score.unsqueeze(1)

    def _context_padded_controller(self, normalized: torch.Tensor) -> torch.Tensor:
        """Run the controller with signal-derived boundary context.

        Physical Basis:
            Forcing incomplete boundary frames to the gentle prototype avoids
            ringing but weakens image rejection at chunk ends. Reflecting real
            waveform context across each boundary fills the controller's
            receptive field without changing the FIR blend policy, allowing a
            steady high-frequency sweep to retain the sharp stopband response.
        """
        mode = (
            "reflect" if normalized.shape[-1] > _CONTROLLER_CONTEXT_PAD else "replicate"
        )
        padded = F.pad(
            normalized.unsqueeze(1),
            (_CONTROLLER_CONTEXT_PAD, _CONTROLLER_CONTEXT_PAD),
            mode=mode,
        ).squeeze(1)
        padded_logits = self.controller(padded)
        crop_frames = _CONTROLLER_CONTEXT_PAD // self.control_stride
        return torch.as_tensor(padded_logits[..., crop_frames:-crop_frames])

    @staticmethod
    def _sparse_transient_score(
        normalized_source: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """Return a frame-rate sparse-event score in [0, 1]."""
        squared = normalized_source.unsqueeze(1).square()
        local_rms = torch.sqrt(
            F.avg_pool1d(
                squared,
                kernel_size=_TRANSIENT_RMS_WINDOW,
                stride=1,
                padding=_TRANSIENT_RMS_WINDOW // 2,
            ).clamp_min(1.0e-12)
        )
        crest = normalized_source.abs().unsqueeze(1) / local_rms
        score = (
            (crest - _TRANSIENT_CREST_START)
            / (_TRANSIENT_CREST_FULL - _TRANSIENT_CREST_START)
        ).clamp(0.0, 1.0)
        frames = F.adaptive_max_pool1d(score, num_frames)
        dilated = F.max_pool1d(
            frames,
            kernel_size=_TRANSIENT_FRAME_DILATION,
            stride=1,
            padding=_TRANSIENT_FRAME_DILATION // 2,
        )
        return dilated.squeeze(1)

    @staticmethod
    def _discontinuity_score(
        normalized_source: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """Return a guard score for sparse, large waveform discontinuities.

        Physical Basis:
            Square/step signals have large sample differences separated by
            exactly flat plateaus. Gaussian noise and multitone signals do not
            have a high density of equal adjacent samples. Combining a large
            slope with plateau density therefore protects discontinuities
            without repeating the former noise false positive.
        """
        difference = F.pad(normalized_source.diff(dim=-1), (1, 0)).abs()
        flat = (difference <= _DISCONTINUITY_FLAT_EPSILON).to(difference.dtype)
        flat_density = F.avg_pool1d(
            flat.unsqueeze(1),
            kernel_size=_DISCONTINUITY_WINDOW,
            stride=1,
            padding=_DISCONTINUITY_WINDOW // 2,
        )
        plateau_score = (
            (flat_density - _DISCONTINUITY_FLAT_START)
            / (_DISCONTINUITY_FLAT_FULL - _DISCONTINUITY_FLAT_START)
        ).clamp(0.0, 1.0)
        slope_score = (
            (difference.unsqueeze(1) - _DISCONTINUITY_SLOPE_START)
            / (_DISCONTINUITY_SLOPE_FULL - _DISCONTINUITY_SLOPE_START)
        ).clamp(0.0, 1.0)
        score = plateau_score * slope_score
        frames = F.adaptive_max_pool1d(score, num_frames)
        return F.max_pool1d(
            frames,
            kernel_size=_TRANSIENT_FRAME_DILATION,
            stride=1,
            padding=_TRANSIENT_FRAME_DILATION // 2,
        ).squeeze(1)

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
        # the system stays linear.
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
        stuffed = source.new_zeros(batch, 1, time * self.upsample_ratio)
        stuffed[:, 0, :: self.upsample_ratio] = source
        kernels = torch.as_tensor(self.kernels)
        return F.conv1d(stuffed, kernels, padding=self.kernel_size // 2)

    def mean_weights(self, source: torch.Tensor) -> torch.Tensor:
        """Return the batch-mean blend weights (monitoring helper).

        Physical Basis:
            A collapse to always-sharp (or always-gentle) is visible
            immediately in the mean weight statistics during training.
        """
        with torch.no_grad():
            weights = self.blend_weights(source)
        return weights.mean(dim=(0, 2))


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
    bank = build_prototype_bank(
        prototype_specs_for_target_rate(target_rate), sample_rate=target_rate
    )
    if (
        expected_input_rate is not None
        and int(expected_input_rate) * bank.upsample_ratio != target_rate
    ):
        raise ValueError(
            f"Checkpoint expected_input_rate {expected_input_rate} Hz is "
            f"inconsistent with target_sample_rate {target_rate} Hz."
        )
    model = CAPB(bank=bank)
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise RuntimeError("Invalid checkpoint: model_state is missing.")
    model.load_state_dict(model_state)
    model.eval()
    return model
