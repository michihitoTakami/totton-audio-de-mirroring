"""Neural Bandwidth Extension Engine (NBEE) — opt-in Stage 1b HB generation.

Physical Basis:
    The suppression-only NMSE applies a [0, 1] mask and therefore cannot
    recreate >22.05kHz content that was destroyed by downsampling to 44.1kHz
    (it can only attenuate what the degraded input already contains). NBEE
    instead *generates* an absolute high-band magnitude from the full-band
    context, trained on genuine native high-band targets, while keeping the
    0–20kHz band a structural bypass and enforcing the same energy-cap,
    envelope, band-limit and high-pass safety constraints as NMSE.

    This crosses the project's default anti-hallucination line on purpose and
    is a separate, opt-in engine: the suppression NMSE is unchanged and remains
    the default. NBEE checkpoints/scripts are labelled ``stage1b``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import torch

from totton_audio_de_mirroring.models.nmse import (
    NMSE,
    STFTConfig,
    _apply_fir_filter,
    _crop_to_shape,
    _flatten_signal,
    _pad_to_multiple,
)
from totton_audio_de_mirroring.models.safety_constraints import (
    apply_safety_constraints,
    enforce_highpass_dc_block,
)
from totton_audio_de_mirroring.models.unet import UNet2D

MODEL_TYPE_NBEE = "nbee"


@dataclass(frozen=True)
class NBEEConfig:
    """Configuration for the bandwidth-extension U-Net.

    Args:
        base_channels: Base channels of the generative U-Net.
        num_downsamples: Number of downsampling blocks.
        channel_multiplier: Channel multiplier per scale.
        activation: Convolution activation name.
        use_batch_norm: Whether to use batch normalization.

    Physical Basis:
        The generator predicts an absolute high-band magnitude from the
        full-band spectrogram, so it needs enough capacity to infer plausible
        high-band structure from the audible-band context.
    """

    base_channels: int = 32
    num_downsamples: int = 4
    channel_multiplier: int = 2
    activation: Literal["relu", "leaky_relu"] = "leaky_relu"
    use_batch_norm: bool = True
    generation_mode: Literal["absolute", "residual"] = "residual"
    gen_start_hz: float = 22_050.0

    def __post_init__(self) -> None:
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if self.num_downsamples <= 0:
            raise ValueError("num_downsamples must be positive.")
        if self.channel_multiplier <= 0:
            raise ValueError("channel_multiplier must be positive.")
        if self.activation not in {"relu", "leaky_relu"}:
            raise ValueError(f"Unsupported activation: {self.activation}.")
        if self.generation_mode not in {"absolute", "residual"}:
            raise ValueError(f"Unsupported generation_mode: {self.generation_mode}.")
        if self.gen_start_hz <= 0.0:
            raise ValueError("gen_start_hz must be positive.")

    @property
    def out_channels(self) -> int:
        """Number of U-Net output channels for the chosen generation mode."""
        return 2 if self.generation_mode == "residual" else 1

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """Serialize config for checkpoint metadata."""
        return {
            "model_type": MODEL_TYPE_NBEE,
            "base_channels": self.base_channels,
            "num_downsamples": self.num_downsamples,
            "channel_multiplier": self.channel_multiplier,
            "activation": self.activation,
            "use_batch_norm": self.use_batch_norm,
            "generation_mode": self.generation_mode,
            "gen_start_hz": self.gen_start_hz,
        }

    @staticmethod
    def from_mapping(raw: Mapping[str, Any]) -> NBEEConfig:
        """Create config from checkpoint metadata mapping."""
        if not isinstance(raw, Mapping):
            raise ValueError("raw must be a mapping.")
        activation = str(raw.get("activation", "leaky_relu")).strip().lower()
        if activation not in {"relu", "leaky_relu"}:
            raise ValueError(f"Unsupported activation: {activation}.")
        mode = str(raw.get("generation_mode", "residual")).strip().lower()
        if mode not in {"absolute", "residual"}:
            raise ValueError(f"Unsupported generation_mode: {mode}.")
        return NBEEConfig(
            base_channels=int(raw.get("base_channels", 32)),
            num_downsamples=int(raw.get("num_downsamples", 4)),
            channel_multiplier=int(raw.get("channel_multiplier", 2)),
            activation=cast(Literal["relu", "leaky_relu"], activation),
            use_batch_norm=bool(raw.get("use_batch_norm", True)),
            generation_mode=cast(Literal["absolute", "residual"], mode),
            gen_start_hz=float(raw.get("gen_start_hz", 22_050.0)),
        )


class NeuralBandwidthExtension(NMSE):
    """Stage 1b engine that generates the high band instead of suppressing it.

    Args:
        sample_rate: Sample rate of the input signal in Hz.
        cutoff_hz: Crossover frequency between LB and HB in Hz.
        energy_cap: Maximum mean energy allowed in the 20–44kHz band.
        envelope_floor: Minimum envelope value at Nyquist.
        lowpass_taps: FIR taps for low-band extraction.
        highpass_taps: FIR taps for high-band extraction.
        stft_config: Optional STFT configuration.
        model_config: Optional generative U-Net configuration.
        cap_start_hz: Start frequency for the frequency-dependent energy cap.
        cap_floor_ratio: Nyquist floor ratio for the frequency-dependent cap.

    Physical Basis:
        Low-band is bypassed (`LB_out = LB_in`) so 0–20kHz is preserved by
        structure. The generator sees the full-band magnitude (low-band context
        included) and predicts an absolute high-band magnitude, which is shaped
        by the shared envelope/energy-cap safety constraints and band-limited by
        a high-pass filter before being added back to the low band.
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
        stft_config: STFTConfig | None = None,
        model_config: NBEEConfig | None = None,
        cap_start_hz: float = 20_000.0,
        cap_floor_ratio: float = 0.1,
    ) -> None:
        config = model_config or NBEEConfig()
        unet = UNet2D(
            in_channels=1,
            out_channels=config.out_channels,
            base_channels=config.base_channels,
            num_downsamples=config.num_downsamples,
            channel_multiplier=config.channel_multiplier,
            activation=config.activation,
            use_batch_norm=config.use_batch_norm,
            output_activation="none",
        )
        super().__init__(
            sample_rate=sample_rate,
            cutoff_hz=cutoff_hz,
            stft_config=stft_config,
            unet=unet,
            envelope_floor=envelope_floor,
            cap_start_hz=cap_start_hz,
            cap_floor_ratio=cap_floor_ratio,
            energy_cap=energy_cap,
            lowpass_taps=lowpass_taps,
            highpass_taps=highpass_taps,
        )
        self.model_config = config
        self.gen_band_mask: torch.Tensor
        freq_bins = self.stft_config.n_fft // 2 + 1
        freqs = torch.fft.rfftfreq(self.stft_config.n_fft, d=1.0 / sample_rate)
        gen_band = (freqs >= config.gen_start_hz).to(torch.float32)
        if gen_band.shape[0] != freq_bins:
            raise ValueError("gen_band_mask length mismatch with STFT bins.")
        self.register_buffer("gen_band_mask", gen_band)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Generate a bandwidth-extended signal.

        Args:
            signal: Input signal (batch, time) or (batch, channels, time).

        Returns:
            Output signal with a generated high band added to the bypassed
            low band.

        Physical Basis:
            The low band is preserved by structure while the high band is
            synthesized from full-band context under safety constraints.
        """
        self._validate_signal(signal)
        lb = _apply_fir_filter(signal, self.lowpass_taps)
        hb_out = self.generate_highband(signal)
        return lb + hb_out

    def generate_highband(self, signal: torch.Tensor) -> torch.Tensor:
        """Generate the high band from the full-band input signal.

        Args:
            signal: Full-band input (batch, time) or (batch, channels, time).

        Returns:
            Generated, band-limited, safety-constrained high-band time signal.

        Physical Basis:
            In "residual" mode the model suppresses the input high band (like
            NMSE) and ADDS generated energy only above ``gen_start_hz``, so it
            defaults to the safe suppressor and stays quiet where the input is
            quiet; in "absolute" mode it predicts the high-band magnitude
            outright. The shared safety constraints cap and envelope the result.
        """
        self._validate_signal(signal)
        flattened, restore_shape = _flatten_signal(signal)
        stft = self._stft(flattened)
        phase = torch.angle(stft)

        input_mag = torch.abs(stft)
        magnitude_4d = input_mag.unsqueeze(1)
        padded, pad_f, pad_t = _pad_to_multiple(
            magnitude_4d,
            multiple=2**self._num_downsamples,
        )
        generated = self.unet(padded)
        generated = _crop_to_shape(generated, pad_f, pad_t)
        generated_mag = self._combine_output(generated, input_mag)

        envelope = cast(torch.Tensor, self.envelope_target)
        highband = cast(torch.Tensor, self.highband_mask)
        hb_mag = apply_safety_constraints(
            generated_mag,
            envelope_target=envelope,
            highband_mask=highband,
            energy_cap=self.energy_cap,
            energy_cap_profile=cast(torch.Tensor, self.energy_cap_profile),
        )

        complex_spec = hb_mag * torch.exp(1j * phase)
        time_signal = self._istft(complex_spec, length=signal.shape[-1])
        time_signal = enforce_highpass_dc_block(time_signal, self.highpass_taps)
        if restore_shape is None:
            return time_signal
        batch, channels, time = restore_shape
        return time_signal.reshape(batch, channels, time)

    def _combine_output(
        self, generated: torch.Tensor, input_mag: torch.Tensor
    ) -> torch.Tensor:
        """Combine the U-Net output into a high-band magnitude.

        Args:
            generated: U-Net output (batch, out_channels, freq, time).
            input_mag: Input STFT magnitude (batch, freq, time).

        Returns:
            High-band magnitude (batch, freq, time) prior to safety shaping.

        Physical Basis:
            "residual" anchors on suppression (`input_mag * sigmoid(mask)`) and
            adds a softplus generation term, gated above ``gen_start_hz`` and
            biased toward zero, so the model only injects high-band energy where
            it actively learns to; "absolute" predicts the magnitude directly.
        """
        if self.model_config.generation_mode == "absolute":
            if generated.shape[1] != 1:
                raise ValueError("absolute mode requires a single output channel.")
            return torch.clamp(generated.squeeze(1), min=0.0)
        if generated.shape[1] != 2:
            raise ValueError("residual mode requires two output channels.")
        mask = torch.sigmoid(generated[:, 0])
        add = torch.nn.functional.softplus(generated[:, 1] - 4.0)
        gen_band = cast(torch.Tensor, self.gen_band_mask).unsqueeze(0).unsqueeze(-1)
        return input_mag * mask + add * gen_band
