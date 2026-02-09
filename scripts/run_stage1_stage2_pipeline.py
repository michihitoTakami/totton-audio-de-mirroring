"""CLI for Stage 1 (NMSE) -> Stage 2 (HIE) integrated inference."""

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
        Fixed, explicit runtime parameters are required for reproducible
        latency/memory and hard-metric comparisons.
    """
    parser = argparse.ArgumentParser(
        description="Run 44.1kHz -> 88.2kHz(Stage1) -> 705.6kHz(Stage2) pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1_stage2_pipeline.yaml"),
        help="YAML config path.",
    )
    parser.add_argument("--input-npy", type=Path, default=None)
    parser.add_argument("--input-wav", type=Path, default=None)
    parser.add_argument("--output-npy", type=Path, default=None)
    parser.add_argument("--output-wav", type=Path, default=None)
    parser.add_argument(
        "--benchmark-duration-sec",
        type=float,
        default=None,
        help="Generate synthetic sine input when explicit input file is omitted.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON payload only.",
    )
    return parser.parse_args()


def main() -> None:
    """Run integrated pipeline and emit metrics/output paths.

    Raises:
        FileNotFoundError: If required config or input files are missing.
        RuntimeError: If audio IO fails.

    Physical Basis:
        End-to-end execution validates Stage 1 safety constraints and Stage 2
        interpolation behavior on the same signal path used in deployment.
    """
    args = parse_args()
    config_raw = _load_yaml(args.config)
    pipeline_config = _build_pipeline_config(config_raw.get("pipeline", {}))
    stage1_processor = _build_stage1_processor(config_raw.get("stage1", {}))
    input_signal = _load_input_signal(args, pipeline_config.source_sample_rate)

    result = run_stage1_stage2_pipeline(
        input_signal,
        stage1_processor=stage1_processor,
        config=pipeline_config,
    )

    if args.output_npy is not None:
        _write_npy(args.output_npy, result.output_signal)
    if args.output_wav is not None:
        _write_wav(
            args.output_wav, result.output_signal, pipeline_config.output_sample_rate
        )

    payload = _build_payload(result, pipeline_config, args)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    _print_summary(payload)


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


def _build_stage1_processor(raw: dict[str, Any]) -> Any:
    if not isinstance(raw, dict):
        raise ValueError("stage1 config must be a mapping.")
    mode = str(raw.get("mode", "reference")).strip().lower()
    if mode == "reference":
        return ReferenceStage1Processor()
    if mode == "nmse":
        checkpoint_raw = raw.get("checkpoint_path")
        if checkpoint_raw is None:
            raise ValueError("stage1.mode=nmse requires checkpoint_path.")
        checkpoint_path = Path(str(checkpoint_raw))
        data_config_path = Path(
            raw.get("data_config_path", "configs/data_generation.yaml")
        )
        if checkpoint_path.as_posix().strip() in {"", "."}:
            raise ValueError("stage1.mode=nmse requires checkpoint_path.")
        device = str(raw.get("device", "cpu"))
        return load_nmse_stage1_processor(
            checkpoint_path=checkpoint_path,
            data_config_path=data_config_path,
            device=device,
        )
    if mode == "onnx":
        model_raw = raw.get("model_path")
        if model_raw is None:
            raise ValueError("stage1.mode=onnx requires model_path.")
        model_path = Path(str(model_raw))
        if model_path.as_posix().strip() in {"", "."}:
            raise ValueError("stage1.mode=onnx requires model_path.")
        data_config_path = Path(
            str(raw.get("data_config_path", "configs/data_generation.yaml"))
        )
        device = str(raw.get("device", "cpu"))
        energy_cap_raw = raw.get("energy_cap")
        energy_cap = None if energy_cap_raw is None else float(energy_cap_raw)
        iir_order = int(raw.get("iir_order", 6))
        return load_onnx_stage1_processor(
            model_path=model_path,
            data_config_path=data_config_path,
            device=device,
            energy_cap=energy_cap,
            iir_order=iir_order,
        )
    raise ValueError(f"Unsupported stage1.mode: {mode}")


def _load_input_signal(args: argparse.Namespace, sample_rate: int) -> np.ndarray:
    sources = [args.input_npy is not None, args.input_wav is not None]
    has_file_input = any(sources)

    if has_file_input and sum(sources) != 1:
        raise ValueError("Specify exactly one of --input-npy or --input-wav.")
    if not has_file_input and args.benchmark_duration_sec is None:
        raise ValueError("Specify input file or --benchmark-duration-sec.")

    if args.input_npy is not None:
        return _read_npy(args.input_npy)
    if args.input_wav is not None:
        return _read_wav(args.input_wav, sample_rate)
    return _generate_benchmark_input(args.benchmark_duration_sec, sample_rate)


def _read_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        signal = np.asarray(np.load(path), dtype=np.float64)
    except Exception as exc:
        raise RuntimeError(f"Failed to load npy input: {exc}") from exc
    _validate_mono_signal(signal)
    return signal


def _read_wav(path: Path, sample_rate: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input wav not found: {path}")
    try:
        signal, sr = sf.read(path, dtype="float64", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read wav input: {exc}") from exc
    if int(sr) != sample_rate:
        raise ValueError(
            f"Input sample rate mismatch: expected {sample_rate}, got {sr}"
        )
    mono = np.asarray(signal, dtype=np.float64)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=1)
    _validate_mono_signal(mono)
    return mono


def _write_npy(path: Path, signal: np.ndarray) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(signal, dtype=np.float64))
    except Exception as exc:
        raise RuntimeError(f"Failed to write output npy: {exc}") from exc


def _write_wav(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.asarray(signal, dtype=np.float32), sample_rate)
    except Exception as exc:
        raise RuntimeError(f"Failed to write output wav: {exc}") from exc


def _generate_benchmark_input(
    duration_sec: float | None, sample_rate: int
) -> np.ndarray:
    if duration_sec is None or duration_sec <= 0.0:
        raise ValueError("benchmark_duration_sec must be positive.")
    num_samples = int(round(duration_sec * sample_rate))
    if num_samples <= 0:
        raise ValueError("benchmark_duration_sec produced zero samples.")
    time_axis = np.arange(num_samples, dtype=np.float64) / float(sample_rate)
    signal = 0.2 * np.sin(2.0 * np.pi * 997.0 * time_axis)
    return np.asarray(signal, dtype=np.float64)


def _build_payload(
    result: Any, config: PipelineConfig, args: argparse.Namespace
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_sample_rate": config.source_sample_rate,
        "stage1_sample_rate": config.stage1_sample_rate,
        "output_sample_rate": config.output_sample_rate,
        "stage2_num_stages": config.stage2_num_stages,
        "stage2_backend": config.stage2_backend,
        "chunk_duration_sec": config.chunk_duration_sec,
        "crossfade_duration_sec": config.crossfade_duration_sec,
        "num_output_samples": int(result.output_signal.shape[0]),
        "performance": asdict(result.performance),
        "output_npy": str(args.output_npy) if args.output_npy is not None else None,
        "output_wav": str(args.output_wav) if args.output_wav is not None else None,
    }
    if result.stage1_metrics is not None:
        payload["stage1_metrics"] = asdict(result.stage1_metrics)
    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    perf = payload["performance"]
    print(
        "Pipeline complete: "
        f"{payload['source_sample_rate']} -> {payload['stage1_sample_rate']} -> "
        f"{payload['output_sample_rate']} Hz"
    )
    print(
        f"Latency={perf['latency_sec']:.3f}s, "
        f"Throughput={perf['throughput_x_realtime']:.2f}x realtime, "
        f"PeakMemory={perf['peak_memory_mb']:.1f}MB"
    )
    if "stage1_metrics" in payload:
        metrics = payload["stage1_metrics"]
        print(
            "Stage1 cap violated: "
            f"{metrics['hb_energy_cap_violated']} "
            f"(energy={metrics['hb_energy']:.4e}, cap={metrics['hb_energy_cap']:.4e})"
        )


def _validate_mono_signal(signal: np.ndarray) -> None:
    if signal.ndim != 1:
        raise ValueError(f"Input must be mono 1D, got {signal.ndim}D.")
    if signal.size == 0:
        raise ValueError("Input signal is empty.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Input signal contains non-finite values.")


if __name__ == "__main__":
    main()
