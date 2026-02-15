"""TensorRT Stage 1 processor for NMSE inference."""

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


class TensorRtSessionProtocol(Protocol):
    """Protocol for TensorRT inference session used by this module."""

    def run(self, input_magnitude: np.ndarray) -> np.ndarray:
        """Run inference and return mask tensor."""


class _TensorRtSession:
    """Minimal TensorRT session wrapper using torch CUDA buffers.

    Args:
        engine: Deserialized TensorRT engine.
        context: TensorRT execution context.
        input_name: Input tensor name.
        output_name: Output tensor name.

    Physical Basis:
        TensorRT acceleration changes only the HB mask estimator backend.
        Band split, safety constraints, and LB bypass remain unchanged.
    """

    def __init__(
        self,
        *,
        engine: Any,
        context: Any,
        input_name: str,
        output_name: str,
    ) -> None:
        self._engine = engine
        self._context = context
        self._input_name = input_name
        self._output_name = output_name

    def run(self, input_magnitude: np.ndarray) -> np.ndarray:
        """Run TensorRT inference for one 4D magnitude tensor.

        Args:
            input_magnitude: Input magnitude tensor `(N, C, F, T)`.

        Returns:
            Output mask tensor with the same shape convention.

        Raises:
            ValueError: If input shape/dtype is invalid.
            RuntimeError: If TensorRT execution fails.

        Physical Basis:
            Input is STFT magnitude in HB path. The engine predicts mask
            coefficients and does not alter phase handling or LB bypass.
        """
        if input_magnitude.ndim != 4:
            raise ValueError(
                "TensorRT input must be 4D (batch, channels, freq, time). "
                f"Got ndim={input_magnitude.ndim}."
            )
        host_input = np.asarray(input_magnitude, dtype=np.float32)

        input_tensor = torch.from_numpy(host_input).to(
            device="cuda", dtype=torch.float32
        )
        input_tensor = input_tensor.contiguous()
        self._set_input_shape(tuple(int(v) for v in input_tensor.shape))

        output_shape = self._resolve_output_shape(input_tensor)
        output_tensor = torch.empty(output_shape, device="cuda", dtype=torch.float32)

        self._set_tensor_address(self._input_name, int(input_tensor.data_ptr()))
        self._set_tensor_address(self._output_name, int(output_tensor.data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        if hasattr(self._context, "execute_async_v3"):
            ok = bool(self._context.execute_async_v3(stream))
        elif hasattr(self._context, "execute_async_v2"):
            bindings = self._build_bindings(
                input_ptr=int(input_tensor.data_ptr()),
                output_ptr=int(output_tensor.data_ptr()),
            )
            ok = bool(
                self._context.execute_async_v2(bindings=bindings, stream_handle=stream)
            )
        else:
            raise RuntimeError("TensorRT execution context lacks async execute API.")
        if not ok:
            raise RuntimeError("TensorRT execution failed.")

        torch.cuda.current_stream().synchronize()
        return np.asarray(output_tensor.detach().cpu().numpy(), dtype=np.float32)

    def _set_input_shape(self, shape: tuple[int, ...]) -> None:
        if hasattr(self._context, "set_input_shape"):
            if not bool(self._context.set_input_shape(self._input_name, shape)):
                raise RuntimeError(
                    f"Failed to set TensorRT input shape {shape} for {self._input_name}."
                )
            return
        if hasattr(self._context, "set_binding_shape") and hasattr(
            self._engine, "get_binding_index"
        ):
            binding_index = int(self._engine.get_binding_index(self._input_name))
            if binding_index < 0:
                raise RuntimeError(f"Input binding not found: {self._input_name}")
            self._context.set_binding_shape(binding_index, shape)
            return
        raise RuntimeError("TensorRT context does not support dynamic shape setup.")

    def _resolve_output_shape(self, input_tensor: torch.Tensor) -> tuple[int, ...]:
        if hasattr(self._context, "get_tensor_shape"):
            raw = tuple(
                int(v) for v in self._context.get_tensor_shape(self._output_name)
            )
            if all(v > 0 for v in raw):
                return raw
        # U-Net mask output should match input magnitude shape.
        return tuple(int(v) for v in input_tensor.shape)

    def _set_tensor_address(self, name: str, ptr: int) -> None:
        if hasattr(self._context, "set_tensor_address"):
            if not bool(self._context.set_tensor_address(name, ptr)):
                raise RuntimeError(f"Failed to set tensor address for {name}.")
            return
        # execute_async_v2 fallback path uses explicit binding list.

    def _build_bindings(self, *, input_ptr: int, output_ptr: int) -> list[int]:
        if not hasattr(self._engine, "num_bindings"):
            raise RuntimeError("TensorRT engine does not expose num_bindings.")
        num_bindings = int(self._engine.num_bindings)
        bindings = [0] * num_bindings
        input_index = int(self._engine.get_binding_index(self._input_name))
        output_index = int(self._engine.get_binding_index(self._output_name))
        if input_index < 0 or output_index < 0:
            raise RuntimeError("Failed to resolve TensorRT binding indices.")
        bindings[input_index] = input_ptr
        bindings[output_index] = output_ptr
        return bindings


@dataclass(frozen=True)
class TensorRtStage1Processor:
    """Stage 1 processor backed by TensorRT mask inference.

    Args:
        session: TensorRT inference session for U-Net mask prediction.
        sample_rate: Stage 1 sample rate in Hz.
        cutoff_hz: Crossover frequency in Hz.
        energy_cap: High-band energy cap.
        envelope_floor: Minimum envelope gain at Nyquist.
        lowpass_taps: Low-band FIR taps.
        highpass_taps: High-band FIR taps.
        stft_config: STFT configuration.
        stft_downsample_power: U-Net downsample power for STFT padding.
        bessel_cutoff_hz: Cutoff for Stage 1 reference SRC preprocessing.
        iir_order: Bessel IIR order for Stage 1 reference SRC preprocessing.

    Physical Basis:
        Mixed precision optimization targets HB neural suppression only.
        LB bypass and filter-domain safety behavior remain FP32/float64 path.
    """

    session: TensorRtSessionProtocol
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
        """Run Stage 1 TensorRT inference.

        Args:
            signal: Mono input signal at source sample rate.
            source_sample_rate: Source sample rate in Hz.
            target_sample_rate: Stage 1 target sample rate in Hz.

        Returns:
            Stage 1 output signal at target sample rate.

        Raises:
            ValueError: If input shape or sample-rate ratio is invalid.
            RuntimeError: If runtime returns malformed output.

        Physical Basis:
            Stage 1 is constrained to fixed 2x mapping (44.1kHz -> 88.2kHz)
            to preserve mirror-suppression semantics.
        """
        _validate_input_signal(signal)
        if target_sample_rate != source_sample_rate * 2:
            raise ValueError(
                "TensorRtStage1Processor requires exact 2x upsampling ratio."
            )

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
        self,
        high_band: torch.Tensor,
        highpass_taps: torch.Tensor,
    ) -> torch.Tensor:
        """Run STFT masking with TensorRT U-Net output."""
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
        mask_np = self.session.run(
            np.asarray(padded.detach().cpu().numpy(), dtype=np.float32)
        )
        mask = torch.from_numpy(np.asarray(mask_np, dtype=np.float32))
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise RuntimeError(
                "TensorRT mask must have shape (batch, 1, freq, time). "
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


def load_tensorrt_stage1_processor(
    *,
    engine_path: Path,
    data_config_path: Path,
    device: str = "cuda",
    energy_cap: float | None = None,
    iir_order: int = 6,
) -> TensorRtStage1Processor:
    """Build TensorRT Stage 1 processor from serialized engine.

    Args:
        engine_path: Path to TensorRT engine file.
        data_config_path: Data generation config path for Stage 1 params.
        device: Runtime device string. TensorRT requires `cuda`.
        energy_cap: Optional override for high-band energy cap.
        iir_order: Bessel filter order for preprocessing.

    Returns:
        Initialized TensorRT-based Stage 1 processor.

    Raises:
        FileNotFoundError: If engine/config path does not exist.
        ValueError: If device is unsupported.
        RuntimeError: If TensorRT session creation fails.

    Physical Basis:
        Backend swap must preserve Stage 1 constraints. This loader keeps
        identical preprocessing and band-structure parameters as PyTorch.
    """
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    if not data_config_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_config_path}")

    runtime_device = device.strip().lower()
    if runtime_device != "cuda":
        raise ValueError("device must be 'cuda' for TensorRT inference.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. TensorRT runtime requires CUDA.")

    session = _load_tensorrt_session(engine_path)

    data_config = load_data_config(data_config_path)
    stage1_sample_rate = int(data_config.target_sample_rate)
    cutoff_hz = float(data_config.band_split.cutoff_hz)
    lowpass_taps, highpass_taps = design_band_split_filters(
        cutoff_hz=cutoff_hz,
        sample_rate=data_config.band_split.sample_rate,
        num_taps=data_config.band_split.num_taps,
        window=data_config.band_split.window,
    )
    resolved_cap = float(
        energy_cap if energy_cap is not None else data_config.hb_target.energy_cap
    )
    return TensorRtStage1Processor(
        session=session,
        sample_rate=stage1_sample_rate,
        cutoff_hz=cutoff_hz,
        energy_cap=resolved_cap,
        envelope_floor=0.0,
        lowpass_taps=lowpass_taps,
        highpass_taps=highpass_taps,
        stft_config=STFTConfig(),
        stft_downsample_power=4,
        bessel_cutoff_hz=float(data_config.band_split.cutoff_hz),
        iir_order=iir_order,
    )


def _load_tensorrt_session(engine_path: Path) -> TensorRtSessionProtocol:
    """Load TensorRT engine and build runtime session.

    Args:
        engine_path: Path to serialized TensorRT engine.

    Returns:
        TensorRT inference session.

    Raises:
        RuntimeError: If TensorRT import/deserialization/setup fails.
    """
    try:
        import tensorrt as trt  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Failed to import tensorrt. Install TensorRT Python bindings first."
        ) from exc

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine_bytes = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context.")

    input_name, output_name = _resolve_io_names(engine)
    return _TensorRtSession(
        engine=engine,
        context=context,
        input_name=input_name,
        output_name=output_name,
    )


def _resolve_io_names(engine: Any) -> tuple[str, str]:
    """Resolve single input/output tensor names from TensorRT engine."""
    # TensorRT 10 style API.
    if hasattr(engine, "num_io_tensors") and hasattr(engine, "get_tensor_mode"):
        try:
            import tensorrt as trt  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                "Failed to import tensorrt while resolving tensor IO."
            ) from exc

        input_names: list[str] = []
        output_names: list[str] = []
        for index in range(int(engine.num_io_tensors)):
            name = str(engine.get_tensor_name(index))
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                output_names.append(name)
        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                "TensorRT engine must have exactly one input and one output tensor."
            )
        return input_names[0], output_names[0]

    # TensorRT <= 8 fallback API.
    if hasattr(engine, "num_bindings") and hasattr(engine, "binding_is_input"):
        input_names_fallback: list[str] = []
        output_names_fallback: list[str] = []
        for index in range(int(engine.num_bindings)):
            name = str(engine.get_binding_name(index))
            if bool(engine.binding_is_input(index)):
                input_names_fallback.append(name)
            else:
                output_names_fallback.append(name)
        if len(input_names_fallback) != 1 or len(output_names_fallback) != 1:
            raise RuntimeError(
                "TensorRT engine must have exactly one input and one output binding."
            )
        return input_names_fallback[0], output_names_fallback[0]

    raise RuntimeError("Unsupported TensorRT engine IO API.")


def _validate_input_signal(signal: np.ndarray) -> None:
    """Validate mono input waveform for Stage 1 processing."""
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D mono waveform, got ndim={signal.ndim}.")
    if signal.size == 0:
        raise ValueError("signal must not be empty.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains non-finite values.")
