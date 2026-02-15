"""End-user batch CLI for Stage 1 -> Stage 2 offline upsampling."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml  # type: ignore[import-untyped]

from totton_audio_de_mirroring.inference import (
    PipelineConfig,
    PipelineResult,
    ReferenceStage1Processor,
    Stage1Processor,
    load_nmse_stage1_processor,
    load_onnx_stage1_processor,
    load_tensorrt_stage1_processor,
    run_stage1_stage2_pipeline,
)

LOGGER = logging.getLogger("totton_audio_de_mirroring.cli")
AUDIO_FORMATS = ("wav", "flac")
OUTPUT_FORMATS = ("wav", "flac", "metadata")
FLAC_MAX_SAMPLE_RATE = 655_350


@dataclass(frozen=True)
class CliSettings:
    """CLI-specific defaults loaded from YAML config.

    Args:
        output_formats: Default enabled formats in config.
        fail_fast: Whether to stop the batch on first failure.
        log_level: Logging level string.

    Physical Basis:
        Offline batch processing prioritizes reproducibility and diagnostics
        over realtime behavior; explicit config keeps runs repeatable.
    """

    output_formats: tuple[str, ...] = ("wav",)
    fail_fast: bool = False
    log_level: str = "INFO"


@dataclass(frozen=True)
class OutputTargets:
    """Resolved output paths for one input file.

    Args:
        audio_outputs: Mapping from audio format to output path.
        metadata_output: Optional JSON metadata output path.

    Physical Basis:
        Artifact paths are pre-resolved before processing so failures in path
        logic cannot modify signal-processing behavior.
    """

    audio_outputs: dict[str, Path]
    metadata_output: Path | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional explicit argv for tests.

    Returns:
        Parsed namespace.

    Physical Basis:
        Reproducible audio processing requires explicit input lists, output
        policy, and processing backend configuration.
    """
    parser = argparse.ArgumentParser(
        prog="totton-upsample",
        description=(
            "Batch upsample audio via Stage1 (44.1kHz->88.2kHz) and "
            "Stage2 (88.2kHz->705.6kHz)."
        ),
    )
    parser.add_argument("inputs", nargs="+", help="Input files or glob patterns.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output file (single input) or output directory (batch).",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/stage1_stage2_pipeline.yaml"),
        help="YAML config path with pipeline/stage1/cli sections.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override stage1 device from config (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--output-format",
        action="append",
        choices=OUTPUT_FORMATS,
        default=None,
        help="Repeatable output format option: wav, flac, metadata.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one file fails.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level override (DEBUG/INFO/WARNING/ERROR).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run end-user batch CLI and exit with a status code.

    Raises:
        SystemExit: With CLI status code.

    Physical Basis:
        Batch/offline workflow improves usability for long datasets while
        preserving the fixed Stage1->Stage2 signal path.
    """
    raise SystemExit(run_cli())


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Execute CLI logic.

    Args:
        argv: Optional explicit argv for tests.

    Returns:
        Process exit code. `0` means full success.

    Physical Basis:
        Per-file isolation prevents one damaged input from invalidating a
        complete offline batch run.
    """
    args = parse_args(argv)
    try:
        config_raw = _load_yaml_mapping(args.config)
        settings = _build_cli_settings(config_raw.get("cli", {}))

        log_level = str(args.log_level or settings.log_level).upper()
        _configure_logging(log_level)

        pipeline_config = _build_pipeline_config(config_raw.get("pipeline", {}))
        stage1_processor = _build_stage1_processor(
            config_raw.get("stage1", {}),
            device_override=args.device,
        )

        selected_formats = _resolve_output_formats(
            args.output_format,
            default_formats=settings.output_formats,
        )
        input_paths = _resolve_input_paths(args.inputs)
        targets = _resolve_output_targets(
            input_paths=input_paths,
            output_arg=args.output,
            output_formats=selected_formats,
        )

        fail_fast = bool(args.fail_fast or settings.fail_fast)
        processed = 0
        failed = 0

        LOGGER.info("Processing started: %d file(s)", len(input_paths))
        _print_progress(0, len(input_paths))

        for index, input_path in enumerate(input_paths, start=1):
            LOGGER.info("[%d/%d] %s", index, len(input_paths), input_path)
            try:
                result = _process_one_file(
                    input_path=input_path,
                    pipeline_config=pipeline_config,
                    stage1_processor=stage1_processor,
                )
                _write_outputs(
                    result=result,
                    input_path=input_path,
                    targets=targets[input_path],
                    output_sample_rate=pipeline_config.output_sample_rate,
                )
                processed += 1
            except Exception as exc:
                failed += 1
                LOGGER.error("Failed to process %s: %s", input_path, exc)
                if fail_fast:
                    _print_progress(index, len(input_paths))
                    LOGGER.error("Stopped by fail-fast mode.")
                    return 1
            finally:
                _print_progress(index, len(input_paths))

        LOGGER.info("Done. success=%d failure=%d", processed, failed)
        return 0 if failed == 0 else 1
    except Exception as exc:
        _print_user_error(str(exc))
        return 1


def _configure_logging(level: str) -> None:
    """Initialize root logging for CLI run."""
    numeric_level = getattr(logging, level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load YAML file as mapping.

    Args:
        path: YAML path.

    Returns:
        Loaded mapping. Empty mapping for empty YAML.

    Raises:
        FileNotFoundError: If path does not exist.
        RuntimeError: If YAML parsing fails.
        ValueError: If root is not a mapping.
    """
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


def _build_cli_settings(raw: Any) -> CliSettings:
    """Build CLI settings from config mapping."""
    if raw is None:
        return CliSettings()
    if not isinstance(raw, dict):
        raise ValueError("cli config must be a mapping.")

    output_raw = raw.get("output_formats", ["wav"])
    if not isinstance(output_raw, list):
        raise ValueError("cli.output_formats must be a list.")
    normalized = tuple(str(item).strip().lower() for item in output_raw)
    _validate_output_formats(normalized)

    return CliSettings(
        output_formats=normalized,
        fail_fast=bool(raw.get("fail_fast", False)),
        log_level=str(raw.get("log_level", "INFO")),
    )


def _build_pipeline_config(raw: Any) -> PipelineConfig:
    """Build PipelineConfig from config mapping."""
    if raw is None:
        raw = {}
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
        overlap_ratio=float(raw.get("overlap_ratio", 0.5)),
        chunk_window=str(raw.get("chunk_window", "hann")),
        crossfade_duration_sec=_coerce_optional_float(
            raw.get("crossfade_duration_sec")
        ),
        stage1_energy_cap=float(raw.get("stage1_energy_cap", 1.0e-3)),
        evaluate_stage1_metrics=bool(raw.get("evaluate_stage1_metrics", True)),
    )


def _build_stage1_processor(
    raw: Any,
    *,
    device_override: str | None,
) -> Stage1Processor:
    """Build Stage 1 processor from config mapping."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("stage1 config must be a mapping.")

    mode = str(raw.get("mode", "reference")).strip().lower()
    if mode == "reference":
        return ReferenceStage1Processor()
    if mode == "nmse":
        checkpoint_raw = raw.get("checkpoint_path")
        if checkpoint_raw is None:
            raise ValueError("stage1.mode=nmse requires checkpoint_path.")
        device = str(device_override or raw.get("device", "cpu"))
        data_config_path = Path(
            str(raw.get("data_config_path", "configs/data_generation.yaml"))
        )
        return load_nmse_stage1_processor(
            checkpoint_path=Path(str(checkpoint_raw)),
            data_config_path=data_config_path,
            device=device,
        )
    if mode == "onnx":
        model_raw = raw.get("model_path")
        if model_raw is None:
            raise ValueError("stage1.mode=onnx requires model_path.")
        device = str(device_override or raw.get("device", "cuda"))
        data_config_path = Path(
            str(raw.get("data_config_path", "configs/data_generation.yaml"))
        )
        energy_cap_raw = raw.get("energy_cap")
        energy_cap = None if energy_cap_raw is None else float(energy_cap_raw)
        iir_order = int(raw.get("iir_order", 6))
        allow_cpu_fallback = bool(raw.get("allow_cpu_fallback", False))
        return load_onnx_stage1_processor(
            model_path=Path(str(model_raw)),
            data_config_path=data_config_path,
            device=device,
            energy_cap=energy_cap,
            iir_order=iir_order,
            allow_cpu_fallback=allow_cpu_fallback,
        )
    if mode == "tensorrt":
        engine_raw = raw.get("engine_path")
        if engine_raw is None:
            raise ValueError("stage1.mode=tensorrt requires engine_path.")
        device = str(device_override or raw.get("device", "cuda"))
        data_config_path = Path(
            str(raw.get("data_config_path", "configs/data_generation.yaml"))
        )
        energy_cap_raw = raw.get("energy_cap")
        energy_cap = None if energy_cap_raw is None else float(energy_cap_raw)
        iir_order = int(raw.get("iir_order", 6))
        return load_tensorrt_stage1_processor(
            engine_path=Path(str(engine_raw)),
            data_config_path=data_config_path,
            device=device,
            energy_cap=energy_cap,
            iir_order=iir_order,
        )
    raise ValueError(f"Unsupported stage1.mode: {mode}")


def _resolve_output_formats(
    cli_formats: list[str] | None,
    *,
    default_formats: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve enabled output formats from CLI/config."""
    if cli_formats is None:
        selected = default_formats
    else:
        selected = tuple(str(item).strip().lower() for item in cli_formats)
    if len(selected) == 0:
        raise ValueError("At least one output format must be enabled.")
    _validate_output_formats(selected)
    # Preserve order while deduplicating.
    return tuple(dict.fromkeys(selected))


def _validate_output_formats(formats: tuple[str, ...]) -> None:
    """Validate output format names."""
    for fmt in formats:
        if fmt not in OUTPUT_FORMATS:
            raise ValueError(f"Unsupported output format: {fmt}")


def _resolve_input_paths(patterns: Sequence[str]) -> list[Path]:
    """Expand files/globs and validate input list.

    Args:
        patterns: Input path or glob expressions.

    Returns:
        Sorted unique input paths.

    Raises:
        ValueError: If nothing matches.
    """
    resolved: set[Path] = set()
    for raw_pattern in patterns:
        pattern = raw_pattern.strip()
        if pattern == "":
            continue
        has_glob = any(char in pattern for char in "*?[]")
        matches = (
            [Path(value) for value in glob(pattern)] if has_glob else [Path(pattern)]
        )
        for path in matches:
            if path.is_file():
                resolved.add(path.resolve())
    ordered = sorted(resolved)
    if not ordered:
        raise ValueError("No input files matched.")
    return ordered


def _resolve_output_targets(
    *,
    input_paths: Sequence[Path],
    output_arg: Path,
    output_formats: tuple[str, ...],
) -> dict[Path, OutputTargets]:
    """Resolve per-input output targets from CLI arguments."""
    audio_formats = tuple(fmt for fmt in output_formats if fmt in AUDIO_FORMATS)
    want_metadata = "metadata" in output_formats

    single_file_mode = (
        len(input_paths) == 1
        and len(audio_formats) == 1
        and not want_metadata
        and output_arg.suffix.lower() in {".wav", ".flac"}
    )

    result: dict[Path, OutputTargets] = {}
    if single_file_mode:
        fmt = audio_formats[0]
        if output_arg.suffix.lower().lstrip(".") != fmt:
            raise ValueError(
                "Single-file output extension must match selected output format."
            )
        output_arg.parent.mkdir(parents=True, exist_ok=True)
        result[input_paths[0]] = OutputTargets(
            audio_outputs={fmt: output_arg}, metadata_output=None
        )
        return result

    output_dir = output_arg
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        stem = input_path.stem
        audio_outputs = {fmt: output_dir / f"{stem}.{fmt}" for fmt in audio_formats}
        metadata_path = output_dir / f"{stem}.json" if want_metadata else None
        result[input_path] = OutputTargets(
            audio_outputs=audio_outputs,
            metadata_output=metadata_path,
        )
    return result


def _process_one_file(
    *,
    input_path: Path,
    pipeline_config: PipelineConfig,
    stage1_processor: Stage1Processor,
) -> PipelineResult:
    """Run full pipeline for one input wav/flac file."""
    signal = _read_audio_mono(input_path, pipeline_config.source_sample_rate)
    return run_stage1_stage2_pipeline(
        signal=signal,
        stage1_processor=stage1_processor,
        config=pipeline_config,
    )


def _read_audio_mono(path: Path, expected_sample_rate: int) -> np.ndarray:
    """Read audio file as finite mono float64 array."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        signal, sample_rate = sf.read(path, dtype="float64", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio: {exc}") from exc
    if int(sample_rate) != expected_sample_rate:
        raise ValueError(
            f"Input sample rate mismatch: expected {expected_sample_rate}, got {sample_rate}."
        )

    mono = np.asarray(signal, dtype=np.float64)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=1)
    if mono.ndim != 1 or mono.size == 0:
        raise ValueError("Input signal must be non-empty mono audio.")
    if not np.all(np.isfinite(mono)):
        raise ValueError("Input signal contains non-finite values.")
    return mono


def _write_outputs(
    *,
    result: PipelineResult,
    input_path: Path,
    targets: OutputTargets,
    output_sample_rate: int,
) -> None:
    """Write audio and metadata outputs for one processed file."""
    output_signal = np.asarray(result.output_signal, dtype=np.float32)
    for fmt, output_path in targets.audio_outputs.items():
        _write_audio(output_path, output_signal, output_sample_rate, fmt)
    if targets.metadata_output is not None:
        _write_metadata(
            path=targets.metadata_output,
            input_path=input_path,
            result=result,
            audio_outputs=targets.audio_outputs,
            output_sample_rate=output_sample_rate,
        )


def _write_audio(path: Path, signal: np.ndarray, sample_rate: int, fmt: str) -> None:
    """Write one output audio file."""
    if fmt not in AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: {fmt}")
    if fmt == "flac" and sample_rate > FLAC_MAX_SAMPLE_RATE:
        raise ValueError(
            "FLAC output is not supported above 655350 Hz; "
            "use WAV or lower output sample rate."
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, signal, sample_rate, format=fmt.upper())
    except Exception as exc:
        raise RuntimeError(f"Failed to write {fmt} output: {exc}") from exc


def _write_metadata(
    *,
    path: Path,
    input_path: Path,
    result: PipelineResult,
    audio_outputs: dict[str, Path],
    output_sample_rate: int,
) -> None:
    """Write JSON metadata sidecar."""
    payload: dict[str, Any] = {
        "input": str(input_path),
        "outputs": {fmt: str(out_path) for fmt, out_path in audio_outputs.items()},
        "output_sample_rate": output_sample_rate,
        "num_output_samples": int(result.output_signal.shape[0]),
        "performance": asdict(result.performance),
    }
    if result.stage1_metrics is not None:
        payload["stage1_metrics"] = asdict(result.stage1_metrics)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to write metadata: {exc}") from exc


def _print_progress(done: int, total: int) -> None:
    """Render a single-line progress bar to stderr."""
    if total <= 0:
        return
    ratio = min(max(done / total, 0.0), 1.0)
    width = 28
    filled = int(round(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if done >= total else ""
    print(f"\r[{bar}] {done}/{total}", end=end, flush=True, file=sys.stderr)


def _print_user_error(message: str) -> None:
    """Print user-friendly fatal error message."""
    print(f"Error: {message}", file=sys.stderr)


def _coerce_optional_float(value: Any) -> float | None:
    """Convert optional values to float."""
    if value is None:
        return None
    return float(value)


__all__ = ["main", "run_cli", "parse_args"]
