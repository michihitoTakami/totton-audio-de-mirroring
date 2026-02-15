"""Build TensorRT engines from Stage 1 ONNX model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TensorRtBuildMode:
    """TensorRT build mode descriptor.

    Args:
        name: Human-readable mode name.
        enable_fp16: Enable FP16 builder flag.
        mixed_precision: Whether to keep IO tensors in FP32 while enabling FP16.

    Physical Basis:
        Mixed precision keeps quality-sensitive boundaries in FP32 while
        accelerating network internals with FP16 kernels where safe.
    """

    name: str
    enable_fp16: bool
    mixed_precision: bool


_BUILD_MODES: dict[str, TensorRtBuildMode] = {
    "fp32": TensorRtBuildMode("fp32", enable_fp16=False, mixed_precision=False),
    "pure_fp16": TensorRtBuildMode(
        "pure_fp16", enable_fp16=True, mixed_precision=False
    ),
    "mixed": TensorRtBuildMode("mixed", enable_fp16=True, mixed_precision=True),
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for TensorRT export.

    Physical Basis:
        Explicit shape profile and precision mode selection is required
        to compare quality/performance tradeoffs reproducibly.
    """
    parser = argparse.ArgumentParser(
        description="Build TensorRT engine(s) from Stage1 ONNX model."
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("data/checkpoints/stage1_best.onnx"),
        help="Input ONNX model path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/checkpoints/tensorrt"),
        help="Output directory for serialized engines.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="mixed",
        help="Comma-separated precision modes: fp32,pure_fp16,mixed",
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=2048,
        help="TensorRT workspace size in MB.",
    )
    parser.add_argument(
        "--freq-bins",
        type=int,
        default=513,
        help="Static STFT frequency bins for input profile.",
    )
    parser.add_argument(
        "--min-time-frames",
        type=int,
        default=16,
        help="Optimization profile minimum time frames.",
    )
    parser.add_argument(
        "--opt-time-frames",
        type=int,
        default=96,
        help="Optimization profile optimum time frames.",
    )
    parser.add_argument(
        "--max-time-frames",
        type=int,
        default=512,
        help="Optimization profile maximum time frames.",
    )
    parser.add_argument(
        "--strict-mixed-io-fp32",
        action="store_true",
        help="For mixed mode, force model input/output tensor dtype to FP32.",
    )
    return parser.parse_args()


def main() -> None:
    """Build one or more TensorRT engines from ONNX model.

    Raises:
        FileNotFoundError: If ONNX path is missing.
        ValueError: If arguments are invalid.
        RuntimeError: If TensorRT parsing/build fails.

    Physical Basis:
        Engine generation must preserve Stage1 signal intent while enabling
        deployment-time backend acceleration on Jetson-class devices.
    """
    args = parse_args()
    _validate_args(args)
    selected_modes = _parse_modes(args.modes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode_name in selected_modes:
        mode = _BUILD_MODES[mode_name]
        output_path = args.output_dir / f"stage1_{mode.name}.engine"
        _build_engine(
            onnx_path=args.onnx_path,
            output_path=output_path,
            mode=mode,
            workspace_mb=args.workspace_mb,
            freq_bins=args.freq_bins,
            min_time_frames=args.min_time_frames,
            opt_time_frames=args.opt_time_frames,
            max_time_frames=args.max_time_frames,
            strict_mixed_io_fp32=bool(args.strict_mixed_io_fp32),
        )
        print(f"Built TensorRT engine ({mode.name}): {output_path}")


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI argument values."""
    if not args.onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_path}")
    if args.workspace_mb <= 0:
        raise ValueError(f"workspace_mb must be positive, got {args.workspace_mb}.")
    if args.freq_bins <= 0:
        raise ValueError(f"freq_bins must be positive, got {args.freq_bins}.")
    if args.min_time_frames <= 0:
        raise ValueError(
            f"min_time_frames must be positive, got {args.min_time_frames}."
        )
    if args.opt_time_frames < args.min_time_frames:
        raise ValueError("opt_time_frames must be >= min_time_frames.")
    if args.max_time_frames < args.opt_time_frames:
        raise ValueError("max_time_frames must be >= opt_time_frames.")


def _parse_modes(raw_modes: str) -> list[str]:
    """Parse comma-separated mode list and validate entries."""
    modes = [item.strip().lower() for item in raw_modes.split(",") if item.strip()]
    if len(modes) == 0:
        raise ValueError("At least one mode must be specified in --modes.")
    unknown = [mode for mode in modes if mode not in _BUILD_MODES]
    if unknown:
        raise ValueError(
            "Unsupported mode(s): "
            + ", ".join(unknown)
            + ". Supported: fp32,pure_fp16,mixed."
        )
    # Preserve order while deduplicating.
    return list(dict.fromkeys(modes))


def _build_engine(
    *,
    onnx_path: Path,
    output_path: Path,
    mode: TensorRtBuildMode,
    workspace_mb: int,
    freq_bins: int,
    min_time_frames: int,
    opt_time_frames: int,
    max_time_frames: int,
    strict_mixed_io_fp32: bool,
) -> None:
    """Build and serialize one TensorRT engine.

    Args:
        onnx_path: Input ONNX path.
        output_path: Output engine path.
        mode: Build precision mode.
        workspace_mb: Workspace memory size in MB.
        freq_bins: STFT frequency bins.
        min_time_frames: Profile minimum frames.
        opt_time_frames: Profile optimum frames.
        max_time_frames: Profile maximum frames.
        strict_mixed_io_fp32: Force FP32 IO tensors in mixed mode.

    Raises:
        RuntimeError: If parser/build returns errors.

    Physical Basis:
        Dynamic time-frame profile supports chunk-size variation while
        fixed frequency bins preserve the STFT configuration contract.
    """
    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = onnx_path.read_bytes()
    if not bool(parser.parse(onnx_bytes)):
        errors = _collect_parser_errors(parser)
        raise RuntimeError(f"TensorRT ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_mb) * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    if mode.enable_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if mode.mixed_precision and hasattr(
        trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"
    ):
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    input_tensor = network.get_input(0)
    input_name = str(input_tensor.name)
    output_tensor = network.get_output(0)
    if mode.mixed_precision and strict_mixed_io_fp32:
        input_tensor.dtype = trt.float32
        output_tensor.dtype = trt.float32
        if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    profile = builder.create_optimization_profile()
    min_shape = (1, 1, int(freq_bins), int(min_time_frames))
    opt_shape = (1, 1, int(freq_bins), int(opt_time_frames))
    max_shape = (1, 1, int(freq_bins), int(max_time_frames))
    profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(serialized))


def _collect_parser_errors(parser: Any) -> str:
    """Collect TensorRT parser error messages as one line."""
    messages: list[str] = []
    num_errors = int(getattr(parser, "num_errors", 0))
    for index in range(num_errors):
        messages.append(str(parser.get_error(index)))
    return "; ".join(messages) if messages else "unknown parser error"


def _import_tensorrt() -> Any:
    """Import TensorRT module with actionable error message."""
    try:
        import tensorrt as trt  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Failed to import tensorrt. Install TensorRT Python bindings first."
        ) from exc
    return trt


if __name__ == "__main__":
    main()
