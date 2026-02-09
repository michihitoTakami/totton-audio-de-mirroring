"""Lightweight NMSE model definition for deployment-oriented distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from totton_audio_de_mirroring.models.nmse import NMSE
from totton_audio_de_mirroring.models.unet import UNet2D

MODEL_TYPE_NMSE_LIGHT = "nmse_light"


@dataclass(frozen=True)
class NMSELightConfig:
    """Configuration for lightweight Stage 1 NMSE architecture.

    Args:
        base_channels: Base channels for the student U-Net.
        num_downsamples: Number of downsampling blocks.
        channel_multiplier: Channel multiplier per scale.
        activation: Convolution activation name.
        use_batch_norm: Whether to use batch normalization.

    Physical Basis:
        Reducing U-Net depth and channel growth lowers compute and memory
        cost while keeping the same band-split and safety constraints.
    """

    base_channels: int = 40
    num_downsamples: int = 3
    channel_multiplier: int = 2
    activation: Literal["relu", "leaky_relu"] = "leaky_relu"
    use_batch_norm: bool = True

    def __post_init__(self) -> None:
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if self.num_downsamples <= 0:
            raise ValueError("num_downsamples must be positive.")
        if self.channel_multiplier <= 0:
            raise ValueError("channel_multiplier must be positive.")
        if self.activation not in {"relu", "leaky_relu"}:
            raise ValueError(f"Unsupported activation: {self.activation}.")

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """Serialize config for checkpoint metadata."""
        return {
            "model_type": MODEL_TYPE_NMSE_LIGHT,
            "base_channels": self.base_channels,
            "num_downsamples": self.num_downsamples,
            "channel_multiplier": self.channel_multiplier,
            "activation": self.activation,
            "use_batch_norm": self.use_batch_norm,
        }

    @staticmethod
    def from_mapping(raw: Mapping[str, Any]) -> NMSELightConfig:
        """Create config from checkpoint metadata mapping."""
        if not isinstance(raw, Mapping):
            raise ValueError("raw must be a mapping.")
        activation = str(raw.get("activation", "leaky_relu")).strip().lower()
        if activation not in {"relu", "leaky_relu"}:
            raise ValueError(f"Unsupported activation: {activation}.")
        use_batch_norm = _parse_bool(raw.get("use_batch_norm", True))
        return NMSELightConfig(
            base_channels=int(raw.get("base_channels", 40)),
            num_downsamples=int(raw.get("num_downsamples", 3)),
            channel_multiplier=int(raw.get("channel_multiplier", 2)),
            activation=cast(Literal["relu", "leaky_relu"], activation),
            use_batch_norm=use_batch_norm,
        )


class NMSELight(NMSE):
    """Lightweight NMSE preserving Stage 1 structural guarantees.

    Args:
        sample_rate: Sample rate of input signal in Hz.
        cutoff_hz: Crossover frequency in Hz.
        energy_cap: High-band energy cap.
        envelope_floor: Envelope floor at Nyquist.
        lowpass_taps: FIR taps for low-band split.
        highpass_taps: FIR taps for high-band split.
        model_config: Optional lightweight architecture config.

    Physical Basis:
        The low-band bypass (`LB_out = LB_in`) is unchanged. Only high-band
        mask estimation capacity is reduced for better deployment efficiency.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        cutoff_hz: float,
        energy_cap: float,
        envelope_floor: float,
        lowpass_taps: np.ndarray,
        highpass_taps: np.ndarray,
        model_config: NMSELightConfig | None = None,
    ) -> None:
        config = model_config or NMSELightConfig()
        unet = UNet2D(
            base_channels=config.base_channels,
            num_downsamples=config.num_downsamples,
            channel_multiplier=config.channel_multiplier,
            activation=config.activation,
            use_batch_norm=config.use_batch_norm,
        )
        super().__init__(
            sample_rate=sample_rate,
            cutoff_hz=cutoff_hz,
            stft_config=None,
            unet=unet,
            envelope_floor=envelope_floor,
            energy_cap=energy_cap,
            lowpass_taps=lowpass_taps,
            highpass_taps=highpass_taps,
        )
        self.model_config = config


def _parse_bool(value: Any) -> bool:
    """Parse bool-like metadata values from checkpoint mappings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected boolean-like value, got {value!r}.")
