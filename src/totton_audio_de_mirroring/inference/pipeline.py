"""Stage 1 -> Stage 2 integration pipeline for offline inference."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

from totton_audio_de_mirroring.data.degradation import upsample_bessel_reference
from totton_audio_de_mirroring.data.filters import design_band_split_filters
from totton_audio_de_mirroring.data.pipeline_config import load_data_config
from totton_audio_de_mirroring.evaluation.metrics import (
    Stage1HardMetrics,
    evaluate_stage1_hard_metrics,
)
from totton_audio_de_mirroring.inference.chunk_processor import (
    ChunkProcessingConfig,
    HannOverlapAddStreamer,
    iterate_chunk_frames,
)
from totton_audio_de_mirroring.models.nmse import NMSE
from totton_audio_de_mirroring.stage2.cpp_backend import (
    CppStage2RuntimeConfig,
    CppStage2Upsampler,
)
from totton_audio_de_mirroring.stage2.overshoot import cascade_upsample, load_stage_taps


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for Stage 1 -> Stage 2 integrated inference.

    Args:
        source_sample_rate: Stage 1 input sample rate (44.1kHz expected).
        stage1_sample_rate: Stage 1 output sample rate (88.2kHz expected).
        output_sample_rate: Final sample rate after Stage 2 (705.6kHz expected).
        stage2_config_dir: Directory containing stage{i}_taps.txt files.
        stage2_num_stages: Number of 2x stages in Stage 2.
        stage2_backend: Stage 2 backend implementation ("cpp" or "python").
        stage2_cpp_project_dir: C++ project directory used for C API build.
        stage2_cpp_build_dir: C++ build output directory.
        chunk_duration_sec: Input chunk duration for long audio processing.
        overlap_ratio: Chunk overlap ratio (fixed to 0.5).
        chunk_window: Stitching window type (fixed to "hann").
        crossfade_duration_sec: Legacy option converted to overlap_ratio.
        stage1_energy_cap: Energy cap used for Stage 1 hard-metric checks.
        evaluate_stage1_metrics: Whether to compute Stage 1 hard metrics.

    Physical Basis:
        Chunked processing with Hann-window overlap-add prevents boundary artifacts
        when Stage 1 and Stage 2 are evaluated as a single long-form pipeline.
    """

    source_sample_rate: int = 44_100
    stage1_sample_rate: int = 88_200
    output_sample_rate: int = 705_600
    stage2_config_dir: Path = Path("cpp/configs")
    stage2_num_stages: int = 3
    stage2_backend: str = "cpp"
    stage2_cpp_project_dir: Path = Path("cpp")
    stage2_cpp_build_dir: Path = Path("cpp/build")
    chunk_duration_sec: float = 0.25
    overlap_ratio: float = 0.5
    chunk_window: str = "hann"
    crossfade_duration_sec: float | None = None
    stage1_energy_cap: float = 1.0e-3
    evaluate_stage1_metrics: bool = True

    def __post_init__(self) -> None:
        if self.source_sample_rate <= 0:
            raise ValueError("source_sample_rate must be positive.")
        if self.stage1_sample_rate <= 0:
            raise ValueError("stage1_sample_rate must be positive.")
        if self.output_sample_rate <= 0:
            raise ValueError("output_sample_rate must be positive.")
        if self.stage2_num_stages <= 0:
            raise ValueError("stage2_num_stages must be positive.")
        backend = self.stage2_backend.strip().lower()
        if backend not in {"cpp", "python"}:
            raise ValueError("stage2_backend must be either 'cpp' or 'python'.")
        if self.chunk_duration_sec <= 0.0:
            raise ValueError("chunk_duration_sec must be positive.")
        if self.crossfade_duration_sec is not None:
            if self.crossfade_duration_sec <= 0.0:
                raise ValueError("crossfade_duration_sec must be positive when set.")
            if self.crossfade_duration_sec >= self.chunk_duration_sec:
                raise ValueError(
                    "crossfade_duration_sec must be smaller than chunk size."
                )
            overlap_ratio = self.crossfade_duration_sec / self.chunk_duration_sec
            object.__setattr__(self, "overlap_ratio", float(overlap_ratio))
        if not np.isclose(self.overlap_ratio, 0.5, atol=1.0e-9):
            raise ValueError("overlap_ratio must be exactly 0.5 for Issue #33.")
        window = self.chunk_window.strip().lower()
        if window != "hann":
            raise ValueError("chunk_window must be 'hann'.")
        if self.stage1_energy_cap <= 0.0:
            raise ValueError("stage1_energy_cap must be positive.")

        if self.stage1_sample_rate != self.source_sample_rate * 2:
            raise ValueError("stage1_sample_rate must be source_sample_rate * 2.")
        expected_output = self.stage1_sample_rate * (2**self.stage2_num_stages)
        if self.output_sample_rate != expected_output:
            raise ValueError(
                "output_sample_rate must match "
                "stage1_sample_rate * (2**stage2_num_stages)."
            )
        object.__setattr__(self, "stage2_backend", backend)
        object.__setattr__(self, "chunk_window", window)


@dataclass(frozen=True)
class PipelinePerformance:
    """Performance and memory summary for one pipeline run.

    Attributes:
        latency_sec: Total processing latency.
        input_duration_sec: Input audio duration in seconds.
        throughput_x_realtime: Input-duration normalized throughput.
        num_chunks: Number of processed input chunks.
        chunk_latency_ms: Mean latency per input chunk in milliseconds.
        peak_memory_mb: Peak resident memory (best-effort process-level).

    Physical Basis:
        Stage 2 integration targets Jetson-class constraints, so throughput
        and memory are required acceptance metrics in addition to audio quality.
    """

    latency_sec: float
    input_duration_sec: float
    throughput_x_realtime: float
    num_chunks: int
    chunk_latency_ms: float
    peak_memory_mb: float


@dataclass(frozen=True)
class PipelineResult:
    """Outputs and measurements for integrated inference run.

    Attributes:
        output_signal: Final 705.6kHz output.
        stage1_signal: Optional assembled Stage 1 output at 88.2kHz.
        stage1_reference: Optional baseline 2x SRC signal at 88.2kHz.
        stage1_metrics: Optional hard-metric evaluation payload.
        performance: Performance summary.

    Physical Basis:
        Keeping Stage 1 intermediate signals makes low-band preservation and
        high-band safety checks verifiable in integrated runs.
    """

    output_signal: np.ndarray
    stage1_signal: np.ndarray | None
    stage1_reference: np.ndarray | None
    stage1_metrics: Stage1HardMetrics | None
    performance: PipelinePerformance


class Stage1Processor(Protocol):
    """Interface for Stage 1 inference implementations."""

    def process(
        self,
        signal: np.ndarray,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray:
        """Process one input chunk and return Stage 1 output."""


class Stage2Processor(Protocol):
    """Interface for Stage 2 inference implementations."""

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Process one Stage 1 chunk and return Stage 2 output chunk."""

    def close(self) -> None:
        """Release backend resources if needed."""


@dataclass
class PythonStage2Processor:
    """Pure Python Stage 2 processor using local FIR taps.

    Physical Basis:
        This path mirrors zero-stuff + FIR cascade logic and is kept for
        regression/testing parity with C++ integration.
    """

    stage_taps: Sequence[np.ndarray]

    def process(self, signal: np.ndarray) -> np.ndarray:
        return cascade_upsample(signal, self.stage_taps)

    def close(self) -> None:
        return None


@dataclass
class CppStage2Processor:
    """C++ core API-backed Stage 2 processor."""

    upsampler: CppStage2Upsampler

    def process(self, signal: np.ndarray) -> np.ndarray:
        return self.upsampler.process(signal)

    def close(self) -> None:
        self.upsampler.close()


@dataclass(frozen=True)
class ReferenceStage1Processor:
    """Reference Stage 1 implementation using deterministic 2x SRC.

    Physical Basis:
        This baseline performs no neural suppression; it exists to validate
        Stage 1 -> Stage 2 wiring and provide a reproducible integration floor.
    """

    cutoff_hz: float = 20_000.0
    iir_order: int = 6

    def process(
        self,
        signal: np.ndarray,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray:
        _validate_input_signal(signal)
        if target_sample_rate != source_sample_rate * 2:
            raise ValueError(
                "ReferenceStage1Processor requires exact 2x upsampling ratio."
            )
        return upsample_bessel_reference(
            signal=np.asarray(signal, dtype=np.float64),
            source_sr=source_sample_rate,
            target_sr=target_sample_rate,
            cutoff_hz=self.cutoff_hz,
            order=self.iir_order,
        )


@dataclass(frozen=True)
class NMSEStage1Processor:
    """Stage 1 processor backed by a loaded NMSE Torch module.

    Args:
        model: NMSE model instance in eval mode.
        device: Torch device to run inference on.
        cutoff_hz: Cutoff used by Bessel reference SRC.
        iir_order: Bessel IIR order for reference SRC.

    Physical Basis:
        Stage 1 keeps low-band identity by structure and only suppresses
        high-band mirror patterns before handing the signal to Stage 2.
    """

    model: nn.Module
    device: torch.device
    cutoff_hz: float = 20_000.0
    iir_order: int = 6

    def process(
        self,
        signal: np.ndarray,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray:
        _validate_input_signal(signal)
        if target_sample_rate != source_sample_rate * 2:
            raise ValueError("NMSEStage1Processor requires exact 2x upsampling ratio.")

        stage1_input = upsample_bessel_reference(
            signal=np.asarray(signal, dtype=np.float64),
            source_sr=source_sample_rate,
            target_sr=target_sample_rate,
            cutoff_hz=self.cutoff_hz,
            order=self.iir_order,
        )
        tensor = (
            torch.from_numpy(np.asarray(stage1_input, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            output = self.model(tensor)
        return np.asarray(output.squeeze(0).detach().cpu().numpy(), dtype=np.float64)


def load_nmse_stage1_processor(
    *,
    checkpoint_path: Path,
    data_config_path: Path,
    device: str = "cpu",
) -> NMSEStage1Processor:
    """Build an NMSE Stage 1 processor from checkpoint and data config.

    Args:
        checkpoint_path: Path to Stage 1 checkpoint.
        data_config_path: Data config used to define NMSE filter settings.
        device: Torch device string.

    Returns:
        Initialized NMSEStage1Processor.

    Raises:
        FileNotFoundError: If required files are missing.
        RuntimeError: If checkpoint loading fails.

    Physical Basis:
        Checkpoint restoration must preserve original band-split and safety
        parameters to keep Stage 1 guarantees valid during inference.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_config_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_config_path}")

    data_config = load_data_config(data_config_path)
    lowpass_taps, highpass_taps = design_band_split_filters(
        cutoff_hz=data_config.band_split.cutoff_hz,
        sample_rate=data_config.band_split.sample_rate,
        num_taps=data_config.band_split.num_taps,
        window=data_config.band_split.window,
    )

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {exc}") from exc

    training_config_raw = checkpoint.get("training_config", {})
    energy_cap = float(
        training_config_raw.get("energy_cap", data_config.hb_target.energy_cap)
    )
    model = NMSE(
        sample_rate=data_config.target_sample_rate,
        cutoff_hz=data_config.band_split.cutoff_hz,
        energy_cap=energy_cap,
        envelope_floor=data_config.hb_target.envelope_min,
        lowpass_taps=lowpass_taps,
        highpass_taps=highpass_taps,
    )
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise RuntimeError("Invalid checkpoint: model_state is missing.")
    model.load_state_dict(model_state)
    model.eval()

    torch_device = torch.device(device)
    model = model.to(torch_device)
    return NMSEStage1Processor(model=model, device=torch_device)


def run_stage1_stage2_pipeline(
    signal: np.ndarray,
    *,
    stage1_processor: Stage1Processor,
    config: PipelineConfig,
) -> PipelineResult:
    """Run Stage 1 -> Stage 2 integrated processing on full-length audio.

    Args:
        signal: Input mono signal at 44.1kHz.
        stage1_processor: Stage 1 processing implementation.
        config: Pipeline runtime configuration.

    Returns:
        PipelineResult with output and measurements.

    Physical Basis:
        Processing follows 44.1kHz -> 88.2kHz(Stage 1) -> 705.6kHz(Stage 2)
        while using Hann-window 50% overlap-add to control boundary artifacts.
    """
    _validate_input_signal(signal)
    stage2_processor = _build_stage2_processor(config)

    input_signal = np.asarray(signal, dtype=np.float64)
    chunking = ChunkProcessingConfig(
        sample_rate=config.source_sample_rate,
        chunk_duration_sec=config.chunk_duration_sec,
        overlap_ratio=config.overlap_ratio,
        window=config.chunk_window,
    )
    output_ratio = config.output_sample_rate // config.source_sample_rate
    stage1_ratio = config.stage1_sample_rate // config.source_sample_rate

    stage1_chunk_samples = chunking.chunk_samples * stage1_ratio
    stage1_overlap_samples = chunking.overlap_samples * stage1_ratio
    output_chunk_samples = chunking.chunk_samples * output_ratio
    output_overlap_samples = chunking.overlap_samples * output_ratio

    start_time = time.perf_counter()
    processed_chunks = 0
    output_segments: list[np.ndarray] = []
    output_streamer = HannOverlapAddStreamer(
        chunk_samples=output_chunk_samples,
        overlap_samples=output_overlap_samples,
        window=config.chunk_window,
    )

    stage1_segments: list[np.ndarray] = []
    stage1_ref_segments: list[np.ndarray] = []
    stage1_streamer: HannOverlapAddStreamer | None = None
    stage1_ref_streamer: HannOverlapAddStreamer | None = None
    if config.evaluate_stage1_metrics:
        stage1_streamer = HannOverlapAddStreamer(
            chunk_samples=stage1_chunk_samples,
            overlap_samples=stage1_overlap_samples,
            window=config.chunk_window,
        )
        stage1_ref_streamer = HannOverlapAddStreamer(
            chunk_samples=stage1_chunk_samples,
            overlap_samples=stage1_overlap_samples,
            window=config.chunk_window,
        )

    try:
        for frame in iterate_chunk_frames(
            input_signal,
            chunk_samples=chunking.chunk_samples,
            overlap_samples=chunking.overlap_samples,
        ):
            processed_chunks += 1
            chunk = frame.samples
            stage1_chunk = stage1_processor.process(
                chunk,
                source_sample_rate=config.source_sample_rate,
                target_sample_rate=config.stage1_sample_rate,
            )
            stage2_chunk = stage2_processor.process(stage1_chunk)
            output_piece = output_streamer.process_chunk(
                np.asarray(stage2_chunk, dtype=np.float64)
            )
            if output_piece.size > 0:
                output_segments.append(np.asarray(output_piece, dtype=np.float64))
            if config.evaluate_stage1_metrics:
                reference_chunk = upsample_bessel_reference(
                    signal=np.asarray(chunk, dtype=np.float64),
                    source_sr=config.source_sample_rate,
                    target_sr=config.stage1_sample_rate,
                    cutoff_hz=20_000.0,
                    order=6,
                )
                if stage1_streamer is None or stage1_ref_streamer is None:
                    raise RuntimeError("Stage1 streamers must be initialized.")

                stage1_piece = stage1_streamer.process_chunk(
                    np.asarray(stage1_chunk, dtype=np.float64)
                )
                if stage1_piece.size > 0:
                    stage1_segments.append(np.asarray(stage1_piece, dtype=np.float64))
                stage1_ref_piece = stage1_ref_streamer.process_chunk(
                    np.asarray(reference_chunk, dtype=np.float64)
                )
                if stage1_ref_piece.size > 0:
                    stage1_ref_segments.append(
                        np.asarray(stage1_ref_piece, dtype=np.float64)
                    )
    finally:
        stage2_processor.close()

    output_tail = output_streamer.finalize()
    if output_tail.size > 0:
        output_segments.append(np.asarray(output_tail, dtype=np.float64))
    output_assembled = _concat_segments(output_segments)

    stage1_assembled = np.zeros(0, dtype=np.float64)
    stage1_ref_assembled = np.zeros(0, dtype=np.float64)
    if config.evaluate_stage1_metrics:
        if stage1_streamer is None or stage1_ref_streamer is None:
            raise RuntimeError("Stage1 streamers must be initialized.")
        stage1_tail = stage1_streamer.finalize()
        stage1_ref_tail = stage1_ref_streamer.finalize()
        if stage1_tail.size > 0:
            stage1_segments.append(np.asarray(stage1_tail, dtype=np.float64))
        if stage1_ref_tail.size > 0:
            stage1_ref_segments.append(np.asarray(stage1_ref_tail, dtype=np.float64))
        stage1_assembled = _concat_segments(stage1_segments)
        stage1_ref_assembled = _concat_segments(stage1_ref_segments)

    latency = time.perf_counter() - start_time
    duration_sec = input_signal.shape[0] / float(config.source_sample_rate)
    throughput = duration_sec / max(latency, 1.0e-12)
    performance = PipelinePerformance(
        latency_sec=float(latency),
        input_duration_sec=float(duration_sec),
        throughput_x_realtime=float(throughput),
        num_chunks=int(processed_chunks),
        chunk_latency_ms=float((latency / max(processed_chunks, 1)) * 1_000.0),
        peak_memory_mb=float(_get_peak_memory_mb()),
    )

    metrics: Stage1HardMetrics | None = None
    if config.evaluate_stage1_metrics:
        metrics = evaluate_stage1_hard_metrics(
            input_signal=np.asarray(stage1_ref_assembled, dtype=np.float64),
            output_signal=np.asarray(stage1_assembled, dtype=np.float64),
            sample_rate=config.stage1_sample_rate,
            energy_cap=config.stage1_energy_cap,
        )
        return PipelineResult(
            output_signal=np.asarray(output_assembled, dtype=np.float64),
            stage1_signal=np.asarray(stage1_assembled, dtype=np.float64),
            stage1_reference=np.asarray(stage1_ref_assembled, dtype=np.float64),
            stage1_metrics=metrics,
            performance=performance,
        )

    return PipelineResult(
        output_signal=np.asarray(output_assembled, dtype=np.float64),
        stage1_signal=None,
        stage1_reference=None,
        stage1_metrics=None,
        performance=performance,
    )


def _build_stage2_processor(config: PipelineConfig) -> Stage2Processor:
    """Build Stage 2 processor from runtime configuration."""
    if config.stage2_backend == "python":
        stage_taps = load_stage_taps(config.stage2_config_dir, config.stage2_num_stages)
        return PythonStage2Processor(stage_taps=stage_taps)

    cpp_config = CppStage2RuntimeConfig(
        config_dir=config.stage2_config_dir,
        num_stages=config.stage2_num_stages,
        cpp_project_dir=config.stage2_cpp_project_dir,
        cpp_build_dir=config.stage2_cpp_build_dir,
    )
    return CppStage2Processor(upsampler=CppStage2Upsampler(cpp_config))


def _concat_segments(segments: Sequence[np.ndarray]) -> np.ndarray:
    if not segments:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(
        [np.asarray(segment, dtype=np.float64) for segment in segments]
    )


def _validate_input_signal(signal: np.ndarray) -> None:
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got {signal.ndim}D.")
    if signal.size == 0:
        raise ValueError("signal cannot be empty.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must contain only finite values.")


def _get_peak_memory_mb() -> float:
    """Get process peak RSS in MB on Unix-like systems."""
    try:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(rss_kb / 1024.0)
    except Exception:
        return 0.0
