"""ONNX Runtime Stage 1 processor for NMSE inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from totton_audio_de_mirroring.data.degradation import upsample_bessel_reference
from totton_audio_de_mirroring.data.filters import design_band_split_filters
from totton_audio_de_mirroring.data.pipeline_config import load_data_config
from totton_audio_de_mirroring.models.nmse import (
    STFTConfig,
    _apply_fir_filter,
    _crop_to_shape,
    _pad_to_multiple,
)
from totton_audio_de_mirroring.models.safety_constraints import (
    apply_safety_constraints,
    build_envelope_target,
    build_highband_mask,
    enforce_highpass_dc_block,
)


class OnnxSessionProtocol(Protocol):
    """Protocol for ONNX Runtime session methods used by this module."""

    def get_inputs(self) -> list[Any]:
        """Return input metadata."""

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        """Run inference."""


@dataclass(frozen=True)
class OnnxStage1Processor:
    """Stage 1 processor backed by ONNX Runtime.

    Args:
        session: ONNX Runtime inference session for U-Net mask prediction.
        sample_rate: Stage 1 sample rate in Hz.
        cutoff_hz: Crossover frequency in Hz.
        energy_cap: High-band energy cap.
        envelope_floor: Minimum envelope gain at Nyquist.
        lowpass_taps: Low-band FIR taps.
        highpass_taps: High-band FIR taps.
        stft_config: STFT configuration.
        stft_downsample_power: U-Net downsample power for STFT padding.
        bessel_cutoff_hz: Cutoff frequency for Stage 1 reference SRC preprocessing.
        iir_order: Bessel IIR order for Stage 1 reference SRC preprocessing.

    Physical Basis:
        Stage 1 input in this project is defined as Bessel-reference 2x SRC.
        ONNX inference is applied on that same 88.2kHz domain so low-band
        preservation and mirror-suppression semantics remain aligned with
        the PyTorch path.
    """

    session: OnnxSessionProtocol
    sample_rate: int
    cutoff_hz: float
    energy_cap: float
    envelope_floor: float
    lowpass_taps: np.ndarray
    highpass_taps: np.ndarray
    stft_config: STFTConfig
    stft_downsample_power: int = 4
    bessel_cutoff_hz: float = 20_000.0
    iir_order: int = 6

    def process(
        self,
        signal: np.ndarray,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray:
        """Run Stage 1 ONNX inference.

        Args:
            signal: Mono input signal at source sample rate.
            source_sample_rate: Source sample rate in Hz.
            target_sample_rate: Stage 1 target sample rate in Hz.

        Returns:
            Stage 1 output signal at target sample rate.

        Raises:
            ValueError: If input shape or sample-rate ratio is invalid.
            RuntimeError: If ONNX runtime returns an unexpected payload.

        Physical Basis:
            Enforcing exact 2x mapping for Stage 1 preserves the project's
            fixed signal path (44.1kHz -> 88.2kHz before suppression).
        """
        _validate_input_signal(signal)
        if target_sample_rate != source_sample_rate * 2:
            raise ValueError("OnnxStage1Processor requires exact 2x upsampling ratio.")

        stage1_input = upsample_bessel_reference(
            signal=np.asarray(signal, dtype=np.float64),
            source_sr=source_sample_rate,
            target_sr=target_sample_rate,
            cutoff_hz=self.bessel_cutoff_hz,
            order=self.iir_order,
        )
        stage1_tensor = torch.from_numpy(
            np.asarray(stage1_input, dtype=np.float32)
        ).view(1, -1)
        lowpass_taps = torch.from_numpy(np.asarray(self.lowpass_taps, dtype=np.float32))
        highpass_taps = torch.from_numpy(
            np.asarray(self.highpass_taps, dtype=np.float32)
        )
        low_band = _apply_fir_filter(stage1_tensor, lowpass_taps)
        high_band = _apply_fir_filter(stage1_tensor, highpass_taps)
        high_band_out = self._process_highband(high_band, highpass_taps)
        output = low_band + high_band_out
        return np.asarray(output.squeeze(0).cpu().numpy(), dtype=np.float64)

    def _process_highband(
        self, high_band: torch.Tensor, highpass_taps: torch.Tensor
    ) -> torch.Tensor:
        """Run STFT masking with ONNX Runtime U-Net output."""
        stft = torch.stft(
            high_band,
            n_fft=self.stft_config.n_fft,
            hop_length=self.stft_config.hop_length,
            win_length=self.stft_config.win_length,
            window=torch.hann_window(self.stft_config.win_length, periodic=True),
            center=self.stft_config.center,
            return_complex=True,
        )
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        magnitude_4d = magnitude.unsqueeze(1)

        padded, pad_f, pad_t = _pad_to_multiple(
            magnitude_4d, multiple=2**self.stft_downsample_power
        )
        input_name = _resolve_input_name(self.session)
        onnx_input = np.asarray(padded.detach().cpu().numpy(), dtype=np.float32)
        outputs = self.session.run(None, {input_name: onnx_input})
        if len(outputs) != 1:
            raise RuntimeError(
                "ONNX Runtime must return exactly one output tensor. "
                f"Got {len(outputs)} outputs."
            )

        mask = torch.from_numpy(np.asarray(outputs[0], dtype=np.float32))
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise RuntimeError(
                "ONNX mask must have shape (batch, 1, freq, time). "
                f"Got {tuple(mask.shape)}."
            )
        mask = torch.clamp(mask, 0.0, 1.0)
        mask = _crop_to_shape(mask, pad_f, pad_t)

        masked_mag = magnitude * mask.squeeze(1)
        freq_bins = self.stft_config.n_fft // 2 + 1
        envelope = build_envelope_target(
            num_freqs=freq_bins,
            sample_rate=self.sample_rate,
            cutoff_hz=self.cutoff_hz,
            floor=self.envelope_floor,
        )
        highband_mask = build_highband_mask(
            num_freqs=freq_bins,
            sample_rate=self.sample_rate,
            cutoff_hz=self.cutoff_hz,
        )
        masked_mag = apply_safety_constraints(
            masked_mag,
            envelope_target=envelope,
            highband_mask=highband_mask,
            energy_cap=self.energy_cap,
        )

        complex_spec = torch.polar(masked_mag, phase)
        time_signal = torch.istft(
            complex_spec,
            n_fft=self.stft_config.n_fft,
            hop_length=self.stft_config.hop_length,
            win_length=self.stft_config.win_length,
            window=torch.hann_window(self.stft_config.win_length, periodic=True),
            center=self.stft_config.center,
            length=high_band.shape[-1],
        )
        return enforce_highpass_dc_block(time_signal, highpass_taps)


def load_onnx_stage1_processor(
    *,
    model_path: Path,
    data_config_path: Path,
    device: str = "cpu",
    energy_cap: float | None = None,
    iir_order: int = 6,
) -> OnnxStage1Processor:
    """Build ONNX Runtime Stage 1 processor from exported NMSE model.

    Args:
        model_path: Path to exported ONNX U-Net model file.
        data_config_path: Data generation config path for Stage 1 params.
        device: Runtime device string (`cpu` or `cuda`).
        energy_cap: Optional override for high-band energy cap.
        iir_order: Bessel filter order for preprocessing.

    Returns:
        Initialized ONNX-based Stage 1 processor.

    Raises:
        FileNotFoundError: If ONNX model path does not exist.
        RuntimeError: If onnxruntime import/session initialization fails.
        ValueError: If `device` is unsupported.

    Physical Basis:
        Runtime backend changes must not change Stage 1 signal semantics;
        preprocessing remains the same Bessel-reference route regardless
        of inference engine.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not data_config_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_config_path}")

    runtime_device = device.strip().lower()
    if runtime_device not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'.")

    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Failed to import onnxruntime. Install 'onnxruntime' first."
        ) from exc

    available_providers = set(ort.get_available_providers())
    if runtime_device == "cuda" and "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(
            model_path.as_posix(),
            providers=providers,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to create ONNX Runtime session: {exc}") from exc
    data_config = load_data_config(data_config_path)
    lowpass_taps, highpass_taps = design_band_split_filters(
        cutoff_hz=data_config.band_split.cutoff_hz,
        sample_rate=data_config.band_split.sample_rate,
        num_taps=data_config.band_split.num_taps,
        window=data_config.band_split.window,
    )
    _ = _resolve_input_name(session)
    return OnnxStage1Processor(
        session=session,
        sample_rate=data_config.target_sample_rate,
        cutoff_hz=data_config.band_split.cutoff_hz,
        energy_cap=(
            float(data_config.hb_target.energy_cap)
            if energy_cap is None
            else float(energy_cap)
        ),
        envelope_floor=float(data_config.hb_target.envelope_min),
        lowpass_taps=lowpass_taps,
        highpass_taps=highpass_taps,
        stft_config=STFTConfig(),
        stft_downsample_power=4,
        bessel_cutoff_hz=data_config.band_split.cutoff_hz,
        iir_order=iir_order,
    )


def _resolve_input_name(session: OnnxSessionProtocol) -> str:
    """Resolve single input tensor name from ONNX Runtime session."""
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(
            f"ONNX model must define exactly one input tensor. Got {len(inputs)}."
        )
    name = getattr(inputs[0], "name", None)
    if not isinstance(name, str) or name.strip() == "":
        raise RuntimeError("ONNX model input name is missing.")
    return name


def _validate_input_signal(signal: np.ndarray) -> None:
    """Validate 1D finite input signal."""
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got {signal.ndim}D.")
    if signal.size == 0:
        raise ValueError("signal cannot be empty.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must contain only finite values.")
