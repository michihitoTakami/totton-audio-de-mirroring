"""Compare full-song inference across Stage 1 backends."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml  # type: ignore[import-untyped]

from totton_audio_de_mirroring.inference import (
    PipelineConfig,
    ReferenceStage1Processor,
    load_nmse_stage1_processor,
    load_onnx_stage1_processor,
    run_stage1_stage2_pipeline,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Physical Basis:
        Full-song comparisons must keep input/audio-path parameters fixed to
        isolate Stage 1 backend effects on quality and throughput.
    """
    parser = argparse.ArgumentParser(
        description="Compare full-song outputs across reference/NMSE/ONNX Stage1 backends."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1_stage2_pipeline.yaml"),
        help="Pipeline YAML config path.",
    )
    parser.add_argument(
        "--input-wav",
        type=Path,
        required=True,
        help="Input mono/stereo WAV at source sample rate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/compare_full_song"),
        help="Directory to write output WAVs and benchmark JSON.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Stage1 checkpoint path for NMSE mode.",
    )
    parser.add_argument(
        "--onnx-model-path",
        type=Path,
        default=None,
        help="Exported ONNX model path for ONNX mode.",
    )
    parser.add_argument(
        "--onnx-device",
        type=str,
        default="cpu",
        help="ONNX Runtime device (cpu/cuda).",
    )
    return parser.parse_args()


def main() -> None:
    """Run full-song comparison and write outputs.

    Raises:
        FileNotFoundError: If config or required model path is missing.
        RuntimeError: If audio I/O fails.

    Physical Basis:
        Comparing complete tracks exposes long-form boundary/stability issues
        that may not appear in short synthetic benchmarks.
    """
    args = parse_args()
    config_raw = _load_yaml(args.config)
    pipeline_config = _build_pipeline_config(config_raw.get("pipeline", {}))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    signal = _read_wav_mono(args.input_wav, pipeline_config.source_sample_rate)
    summaries: dict[str, Any] = {}

    reference_result = run_stage1_stage2_pipeline(
        signal=signal,
        stage1_processor=ReferenceStage1Processor(),
        config=pipeline_config,
    )
    reference_path = output_dir / "reference.wav"
    sf.write(
        reference_path,
        np.asarray(reference_result.output_signal, dtype=np.float32),
        pipeline_config.output_sample_rate,
    )
    summaries["reference"] = _summarize_result(reference_result, reference_path)

    if args.checkpoint_path is not None:
        nmse_processor = load_nmse_stage1_processor(
            checkpoint_path=args.checkpoint_path,
            data_config_path=Path("configs/data_generation.yaml"),
            device="cpu",
        )
        nmse_result = run_stage1_stage2_pipeline(
            signal=signal,
            stage1_processor=nmse_processor,
            config=pipeline_config,
        )
        nmse_path = output_dir / "nmse.wav"
        sf.write(
            nmse_path,
            np.asarray(nmse_result.output_signal, dtype=np.float32),
            pipeline_config.output_sample_rate,
        )
        summaries["nmse"] = _summarize_result(nmse_result, nmse_path)

    if args.onnx_model_path is not None:
        onnx_processor = load_onnx_stage1_processor(
            model_path=args.onnx_model_path,
            data_config_path=Path("configs/data_generation.yaml"),
            device=args.onnx_device,
        )
        onnx_result = run_stage1_stage2_pipeline(
            signal=signal,
            stage1_processor=onnx_processor,
            config=pipeline_config,
        )
        onnx_path = output_dir / "onnx.wav"
        sf.write(
            onnx_path,
            np.asarray(onnx_result.output_signal, dtype=np.float32),
            pipeline_config.output_sample_rate,
        )
        summaries["onnx"] = _summarize_result(onnx_result, onnx_path)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote comparison summary: {summary_path}")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping.")
    return loaded


def _build_pipeline_config(raw: dict[str, Any]) -> PipelineConfig:
    if not isinstance(raw, dict):
        raise ValueError("pipeline config must be a mapping.")
    return PipelineConfig(
        source_sample_rate=int(raw.get("source_sample_rate", 44_100)),
        stage1_sample_rate=int(raw.get("stage1_sample_rate", 88_200)),
        output_sample_rate=int(raw.get("output_sample_rate", 705_600)),
        stage2_config_dir=Path(raw.get("stage2_config_dir", "cpp/configs")),
        stage2_num_stages=int(raw.get("stage2_num_stages", 3)),
        stage2_backend=str(raw.get("stage2_backend", "cpp")),
        stage2_cpp_project_dir=Path(raw.get("stage2_cpp_project_dir", "cpp")),
        stage2_cpp_build_dir=Path(raw.get("stage2_cpp_build_dir", "cpp/build")),
        chunk_duration_sec=float(raw.get("chunk_duration_sec", 0.25)),
        crossfade_duration_sec=float(raw.get("crossfade_duration_sec", 0.05)),
        stage1_energy_cap=float(raw.get("stage1_energy_cap", 1.0e-3)),
        evaluate_stage1_metrics=bool(raw.get("evaluate_stage1_metrics", True)),
    )


def _read_wav_mono(path: Path, sample_rate: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input wav not found: {path}")
    try:
        signal, loaded_sr = sf.read(path, dtype="float64", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read wav input: {exc}") from exc
    if int(loaded_sr) != sample_rate:
        raise ValueError(
            f"Input sample rate mismatch: expected {sample_rate}, got {loaded_sr}."
        )
    mono = np.asarray(signal, dtype=np.float64)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=1)
    if mono.ndim != 1 or mono.size == 0:
        raise ValueError("Input signal must be non-empty mono audio.")
    return mono


def _summarize_result(result: Any, output_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "output_wav": str(output_path),
        "performance": asdict(result.performance),
        "num_output_samples": int(result.output_signal.shape[0]),
    }
    if result.stage1_metrics is not None:
        payload["stage1_metrics"] = asdict(result.stage1_metrics)
    return payload


if __name__ == "__main__":
    main()
