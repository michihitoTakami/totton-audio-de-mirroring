"""Generate Issue #109 baseline outputs with per-channel Stage1 inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.io import wavfile

from totton_audio_de_mirroring.inference.pipeline import load_nmse_stage1_processor


class Stage1ProcessorProtocol(Protocol):
    """Protocol for Stage 1 processor used in baseline generation.

    Physical Basis:
        Stage 1 inference is applied per channel to preserve stereo cues while
        keeping each channel's mirror suppression behavior consistent.
    """

    def process(
        self,
        signal: np.ndarray,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray: ...


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Physical Basis:
        Deterministic I/O paths and checkpoint selection ensure reproducible
        baseline output generation for metric comparison.
    """

    parser = argparse.ArgumentParser(
        description="Generate Issue #109 baseline_nn outputs from 44.1kHz stimuli."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("third_party/microstructure-metrics/test_signals_88k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/issue109/baseline_nn"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/checkpoints/stage1_best.pt"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data_generation.yaml"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate baseline_nn outputs for all 44.1kHz test stimuli.

    Raises:
        FileNotFoundError: If input directory is missing.
        RuntimeError: If signal processing output shape is invalid.

    Physical Basis:
        Channel-wise inference prevents collapsing stereo information (ITD/ILD/
        side components) that is required by TFS and binaural metrics.
    """

    args = parse_args()
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = load_nmse_stage1_processor(
        checkpoint_path=args.checkpoint,
        data_config_path=args.data_config,
        device=args.device,
    )

    for wav_path in sorted(args.input_dir.glob("*_44100_hz_24bit_v1.wav")):
        output_name = wav_path.name.replace("_44100_hz_24bit_v1.wav", "_88200_full.wav")
        output_path = args.output_dir / output_name
        if output_path.exists() and not args.overwrite:
            continue
        sample_rate, signal = wavfile.read(wav_path)
        if int(sample_rate) != 44_100:
            continue
        processed = run_per_channel_inference(
            processor=processor,
            signal=np.asarray(signal),
            source_sample_rate=44_100,
            target_sample_rate=88_200,
        )
        wavfile.write(output_path, 88_200, processed.astype(np.float32))


def run_per_channel_inference(
    *,
    processor: Stage1ProcessorProtocol,
    signal: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """Run Stage1 inference channel-by-channel and re-combine.

    Args:
        processor: Stage 1 processor implementation.
        signal: Input waveform (`[N]` or `[N, C]`).
        source_sample_rate: Source sample rate in Hz.
        target_sample_rate: Target sample rate in Hz.

    Returns:
        Processed waveform (`[N*2]` for mono, `[N*2, C]` for multi-channel).

    Raises:
        ValueError: If input shape is invalid.
        RuntimeError: If per-channel output lengths mismatch.

    Physical Basis:
        Running inference independently on each channel preserves stereo phase
        relationships better than pre-mix to mono then duplicate.
    """

    channels = to_float_channels(signal)
    outputs: list[np.ndarray] = []
    for channel_index in range(channels.shape[1]):
        channel_signal = channels[:, channel_index]
        enhanced = processor.process(
            channel_signal,
            source_sample_rate=source_sample_rate,
            target_sample_rate=target_sample_rate,
        )
        outputs.append(np.asarray(enhanced, dtype=np.float64))
    lengths = {item.shape[0] for item in outputs}
    if len(lengths) != 1:
        raise RuntimeError(f"Per-channel output length mismatch: {sorted(lengths)}")
    if channels.shape[1] == 1:
        return outputs[0]
    return np.stack(outputs, axis=1)


def to_float_channels(signal: np.ndarray) -> np.ndarray:
    """Convert waveform to float64 channel-first matrix `[N, C]`.

    Args:
        signal: Input waveform array.

    Returns:
        Float64 waveform with explicit channel dimension.

    Raises:
        ValueError: If signal rank is not 1D or 2D.

    Physical Basis:
        Stable amplitude normalization is required to avoid clipping and to keep
        metric comparisons consistent across integer/float WAV sources.
    """

    waveform = np.asarray(signal)
    if waveform.ndim == 1:
        waveform_2d = waveform[:, np.newaxis]
    elif waveform.ndim == 2:
        waveform_2d = waveform
    else:
        raise ValueError(f"signal must be 1D or 2D, got {waveform.ndim}D")
    if np.issubdtype(waveform_2d.dtype, np.integer):
        if waveform_2d.dtype == np.int32:
            scale = float(2**31)
        else:
            scale = float(np.iinfo(waveform_2d.dtype).max)
        return waveform_2d.astype(np.float64) / scale
    return waveform_2d.astype(np.float64)


if __name__ == "__main__":
    main()
