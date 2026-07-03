"""CAPB: Constrained Adaptive Prototype-Blend 2x upsampler.

The model upsamples 44.1 kHz audio to 88.2 kHz as a convex, slowly
time-varying blend of fixed linear-phase interpolation prototypes. A small
waveform controller predicts per-frame blend weights; everything else is
frozen DSP. The network therefore cannot create frequencies, cannot change
gain, and cannot ring beyond the worst fixed prototype - it only chooses a
point on the sharp-vs-gentle trade-off curve over time.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from totton_audio_de_mirroring.models.proto_bank import (
    DEFAULT_PROTOTYPE_SPECS,
    PrototypeBank,
    build_prototype_bank,
)

DEFAULT_CONTROL_STRIDE = 64
DEFAULT_INIT_WEIGHTS = (0.85, 0.10, 0.05)
_CONTROLLER_CHANNELS = (24, 32, 40, 48, 48)
_CONTROLLER_STRIDES = (2, 2, 2, 2, 4)
_CONTROLLER_KERNEL = 9


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
        peak = source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = self.controller(source / peak)
        weights = torch.softmax(logits, dim=1)
        weights_up = F.interpolate(
            weights,
            size=prototype_outputs.shape[-1],
            mode="linear",
            align_corners=False,
        )
        output = (weights_up * prototype_outputs).sum(dim=1)
        return output, weights, prototype_outputs

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
            weights = torch.softmax(self.controller(source), dim=1)
        return weights.mean(dim=(0, 2))
