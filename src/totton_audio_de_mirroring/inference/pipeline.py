"""Stage 1 -> Stage 2 integration pipeline for offline inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import scipy.signal as sp_signal
import torch
from torch import nn

from totton_audio_de_mirroring.data.filters import design_band_split_filters
from totton_audio_de_mirroring.data.pipeline_config import load_data_config
from totton_audio_de_mirroring.evaluation.metrics import (
    Stage1HardMetrics,
    evaluate_stage1_hard_metrics,
)
from totton_audio_de_mirroring.models.nmse import NMSE
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
        chunk_duration_sec: Input chunk duration for long audio processing.
        crossfade_duration_sec: Input-domain crossfade duration between chunks.
        stage1_energy_cap: Energy cap used for Stage 1 hard-metric checks.
        evaluate_stage1_metrics: Whether to compute Stage 1 hard metrics.

    Physical Basis:
        Chunked processing with overlap/crossfade prevents boundary artifacts
        when Stage 1 and Stage 2 are evaluated as a single long-form pipeline.
    """

    source_sample_rate: int = 44_100
    stage1_sample_rate: int = 88_200
    output_sample_rate: int = 705_600
    stage2_config_dir: Path = Path("cpp/configs")
    stage2_num_stages: int = 3
    chunk_duration_sec: float = 0.25
    crossfade_duration_sec: float = 0.05
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
        if self.chunk_duration_sec <= 0.0:
            raise ValueError("chunk_duration_sec must be positive.")
        if self.crossfade_duration_sec < 0.0:
            raise ValueError("crossfade_duration_sec must be non-negative.")
        if self.crossfade_duration_sec >= self.chunk_duration_sec:
            raise ValueError("crossfade_duration_sec must be smaller than chunk size.")
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


@dataclass(frozen=True)
class PipelinePerformance:
    """Performance and memory summary for one pipeline run.

    Attributes:
        latency_sec: Total processing latency.
        input_duration_sec: Input audio duration in seconds.
        throughput_x_realtime: Input-duration normalized throughput.
        peak_memory_mb: Peak resident memory (best-effort process-level).

    Physical Basis:
        Stage 2 integration targets Jetson-class constraints, so throughput
        and memory are required acceptance metrics in addition to audio quality.
    """

    latency_sec: float
    input_duration_sec: float
    throughput_x_realtime: float
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


@dataclass(frozen=True)
class ReferenceStage1Processor:
    """Reference Stage 1 implementation using deterministic 2x SRC.

    Physical Basis:
        This baseline performs no neural suppression; it exists to validate
        Stage 1 -> Stage 2 wiring and provide a reproducible integration floor.
    """

    window: str | tuple[str, float] = ("kaiser", 8.0)

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
        upsampled = sp_signal.resample_poly(
            np.asarray(signal, dtype=np.float64),
            up=2,
            down=1,
            window=self.window,
        )
        return np.asarray(upsampled, dtype=np.float64)


@dataclass(frozen=True)
class NMSEStage1Processor:
    """Stage 1 processor backed by a loaded NMSE Torch module.

    Args:
        model: NMSE model instance in eval mode.
        device: Torch device to run inference on.
        reference_window: Window used for 44.1->88.2 reference SRC.

    Physical Basis:
        Stage 1 keeps low-band identity by structure and only suppresses
        high-band mirror patterns before handing the signal to Stage 2.
    """

    model: nn.Module
    device: torch.device
    reference_window: str | tuple[str, float] = ("kaiser", 8.0)

    def process(
        self,
        signal: np.ndarray,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray:
        _validate_input_signal(signal)
        if target_sample_rate != source_sample_rate * 2:
            raise ValueError("NMSEStage1Processor requires exact 2x upsampling ratio.")

        stage1_input = sp_signal.resample_poly(
            np.asarray(signal, dtype=np.float64),
            up=2,
            down=1,
            window=self.reference_window,
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
        while using crossfaded chunk stitching to control boundary artifacts.
    """
    _validate_input_signal(signal)
    stage_taps = load_stage_taps(config.stage2_config_dir, config.stage2_num_stages)

    input_signal = np.asarray(signal, dtype=np.float64)
    chunk_samples = int(round(config.chunk_duration_sec * config.source_sample_rate))
    crossfade_in = int(round(config.crossfade_duration_sec * config.source_sample_rate))
    output_ratio = config.output_sample_rate // config.source_sample_rate
    stage1_ratio = config.stage1_sample_rate // config.source_sample_rate

    if chunk_samples <= 0:
        raise ValueError("chunk_duration_sec produced zero chunk length.")
    crossfade_stage1 = crossfade_in * stage1_ratio
    crossfade_output = crossfade_in * output_ratio

    start_time = time.perf_counter()
    output_assembled = np.zeros(0, dtype=np.float64)
    stage1_assembled = np.zeros(0, dtype=np.float64)
    stage1_ref_assembled = np.zeros(0, dtype=np.float64)

    for chunk in _iterate_chunks(input_signal, chunk_samples, crossfade_in):
        stage1_chunk = stage1_processor.process(
            chunk,
            source_sample_rate=config.source_sample_rate,
            target_sample_rate=config.stage1_sample_rate,
        )
        stage2_chunk = cascade_upsample(stage1_chunk, stage_taps)
        output_assembled = _crossfade_append(
            output_assembled,
            np.asarray(stage2_chunk, dtype=np.float64),
            crossfade_output,
        )
        if config.evaluate_stage1_metrics:
            reference_chunk = sp_signal.resample_poly(
                np.asarray(chunk, dtype=np.float64),
                up=2,
                down=1,
                window=("kaiser", 8.0),
            )
            stage1_assembled = _crossfade_append(
                stage1_assembled,
                np.asarray(stage1_chunk, dtype=np.float64),
                crossfade_stage1,
            )
            stage1_ref_assembled = _crossfade_append(
                stage1_ref_assembled,
                np.asarray(reference_chunk, dtype=np.float64),
                crossfade_stage1,
            )

    latency = time.perf_counter() - start_time
    duration_sec = input_signal.shape[0] / float(config.source_sample_rate)
    throughput = duration_sec / max(latency, 1.0e-12)
    performance = PipelinePerformance(
        latency_sec=float(latency),
        input_duration_sec=float(duration_sec),
        throughput_x_realtime=float(throughput),
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


def _iterate_chunks(
    signal: np.ndarray, chunk_samples: int, crossfade_samples: int
) -> tuple[np.ndarray, ...]:
    """Split a signal into overlap chunks."""
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive.")
    if crossfade_samples < 0:
        raise ValueError("crossfade_samples must be non-negative.")
    if crossfade_samples >= chunk_samples:
        raise ValueError("crossfade_samples must be smaller than chunk_samples.")
    if signal.shape[0] <= chunk_samples:
        return (np.asarray(signal, dtype=np.float64),)

    hop = chunk_samples - crossfade_samples
    chunks: list[np.ndarray] = []
    start = 0
    while start < signal.shape[0]:
        end = min(start + chunk_samples, signal.shape[0])
        chunks.append(np.asarray(signal[start:end], dtype=np.float64))
        if end >= signal.shape[0]:
            break
        start += hop
    return tuple(chunks)


def _crossfade_append(
    left: np.ndarray, right: np.ndarray, crossfade_samples: int
) -> np.ndarray:
    """Append two signals with optional linear crossfade overlap."""
    if left.size == 0:
        return np.asarray(right, dtype=np.float64)
    if crossfade_samples <= 0:
        return np.concatenate([left, right])

    overlap = min(crossfade_samples, left.shape[0], right.shape[0])
    if overlap <= 0:
        return np.concatenate([left, right])

    fade_in = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float64)
    fade_out = 1.0 - fade_in
    blended = left[-overlap:] * fade_out + right[:overlap] * fade_in
    return np.concatenate([left[:-overlap], blended, right[overlap:]])


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
