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
_CONTROLLER_CHANNELS = (24, 32, 40, 48, 48)
_CONTROLLER_STRIDES = (2, 2, 2, 2, 4)
_CONTROLLER_KERNEL = 9
FIRComputeDType = Literal["float32", "float64"]
SUPPORTED_FIR_COMPUTE_DTYPES: tuple[FIRComputeDType, ...] = ("float32", "float64")


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
        fir_compute_dtype: Arithmetic dtype for the fixed FIR path.

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
        fir_compute_dtype: FIRComputeDType = "float32",
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

        kernel_dtype = (
            torch.float32 if fir_compute_dtype == "float32" else torch.float64
        )
        kernels = torch.from_numpy(np.ascontiguousarray(bank.kernels)).to(kernel_dtype)
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
        controller_dtype = self.controller.head.weight.dtype
        controller_source = source.to(dtype=controller_dtype)
        peak = controller_source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = self.controller(controller_source / peak)
        weights = torch.softmax(logits, dim=1)
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
            weights = torch.softmax(self.controller(source), dim=1)
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
    model = CAPB(bank=bank, fir_compute_dtype=fir_compute_dtype)
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
) -> CAPB:
    """Pair an existing controller with an experimental prototype profile.

    Args:
        checkpoint: Source checkpoint containing validated controller weights.
        prototype_profile: Explicit experimental prototype profile.
        fir_compute_dtype: Arithmetic dtype for fixed FIR convolution.

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
    model = CAPB(bank=bank, fir_compute_dtype=fir_compute_dtype)
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
