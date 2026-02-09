"""Export Stage 1 NMSE checkpoint to ONNX format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import torch

from totton_audio_de_mirroring.inference.pipeline import load_nmse_stage1_processor
from totton_audio_de_mirroring.models.nmse import NMSE, _pad_to_multiple


def parse_args() -> argparse.Namespace:
    """Parse export CLI arguments.

    Physical Basis:
        Explicit export options make model conversion reproducible across
        environments and preserve Stage 1 signal semantics.
    """
    parser = argparse.ArgumentParser(
        description="Export Stage1 NMSE checkpoint to ONNX."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("data/checkpoints/stage1_best.pt"),
        help="Path to Stage 1 checkpoint (.pt).",
    )
    parser.add_argument(
        "--data-config-path",
        type=Path,
        default=Path("configs/data_generation.yaml"),
        help="Path to data generation config used for filter construction.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/checkpoints/stage1_best.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version (17+ recommended).",
    )
    parser.add_argument(
        "--dummy-samples",
        type=int,
        default=22_050,
        help="Dummy Stage 1 waveform length used to derive STFT frame count.",
    )
    parser.add_argument(
        "--disable-dynamic-axes",
        action="store_true",
        help="Disable dynamic batch/time axes in exported ONNX.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used for checkpoint loading/export.",
    )
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="Run onnx.checker.check_model on exported file.",
    )
    parser.add_argument(
        "--verify-ort",
        action="store_true",
        help="Run ONNX Runtime inference and compare against PyTorch U-Net output.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-5,
        help="Max absolute error threshold for --verify-ort.",
    )
    return parser.parse_args()


def main() -> None:
    """Export checkpoint and optionally run checker/runtime verification.

    Raises:
        ValueError: If arguments are invalid.
        FileNotFoundError: If checkpoint/config path is missing.
        RuntimeError: If export/check/verification fails.

    Physical Basis:
        Numerical parity checks against PyTorch are required to ensure
        ONNX deployment preserves Stage 1 suppression behavior.
    """
    args = parse_args()
    _validate_args(args)

    processor = load_nmse_stage1_processor(
        checkpoint_path=args.checkpoint_path,
        data_config_path=args.data_config_path,
        device=args.device,
    )
    nmse_model = cast(NMSE, processor.model)
    nmse_model.eval()
    unet_model = nmse_model.unet
    unet_model.eval()

    dummy_wave = torch.zeros((1, args.dummy_samples), dtype=torch.float32)
    with torch.no_grad():
        dummy_stft = nmse_model._stft(dummy_wave)
    dummy_magnitude = torch.abs(dummy_stft).unsqueeze(1)
    downsample_multiple = 2 ** len(unet_model.down_blocks)
    dummy_input, _, _ = _pad_to_multiple(dummy_magnitude, multiple=downsample_multiple)
    dynamic_axes: dict[str, dict[int, str]] | None = None
    if not args.disable_dynamic_axes:
        dynamic_axes = {
            "input_magnitude": {0: "batch_size", 2: "freq_bins", 3: "time_frames"},
            "output_mask": {0: "batch_size", 2: "freq_bins", 3: "time_frames"},
        }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model=unet_model,
            args=(dummy_input,),
            f=args.output_path.as_posix(),
            export_params=True,
            dynamo=False,
            opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=["input_magnitude"],
            output_names=["output_mask"],
            dynamic_axes=dynamic_axes,
        )
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed: {exc}") from exc

    if args.check_model:
        _run_onnx_checker(args.output_path)
    if args.verify_ort:
        _verify_with_onnxruntime(
            model=unet_model,
            output_path=args.output_path,
            dummy_input=dummy_input,
            tolerance=args.tolerance,
        )

    print(f"Exported ONNX model: {args.output_path}")


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.opset_version <= 0:
        raise ValueError(f"opset_version must be positive, got {args.opset_version}.")
    if args.opset_version < 17:
        raise ValueError("opset_version must be >= 17.")
    if args.dummy_samples <= 0:
        raise ValueError(f"dummy_samples must be positive, got {args.dummy_samples}.")
    if args.tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {args.tolerance}.")


def _run_onnx_checker(output_path: Path) -> None:
    """Run ONNX checker for exported model."""
    try:
        import onnx  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Failed to import onnx. Install 'onnx' to use --check-model."
        ) from exc

    try:
        model_proto = onnx.load(output_path.as_posix())
        onnx.checker.check_model(model_proto)
    except Exception as exc:
        raise RuntimeError(f"onnx.checker failed: {exc}") from exc


def _verify_with_onnxruntime(
    *,
    model: torch.nn.Module,
    output_path: Path,
    dummy_input: torch.Tensor,
    tolerance: float,
) -> None:
    """Compare ONNX Runtime output with PyTorch output."""
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Failed to import onnxruntime. Install 'onnxruntime' to use --verify-ort."
        ) from exc

    with torch.no_grad():
        torch_output = model(dummy_input).detach().cpu().numpy()

    try:
        session = ort.InferenceSession(
            output_path.as_posix(),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize ONNX Runtime: {exc}") from exc

    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(f"Expected single ONNX input, got {len(inputs)}.")
    input_name = getattr(inputs[0], "name", "")
    if not isinstance(input_name, str) or input_name.strip() == "":
        raise RuntimeError("ONNX input name is missing.")

    onnx_output = session.run(None, {input_name: dummy_input.numpy()})
    if len(onnx_output) != 1:
        raise RuntimeError(f"Expected single ONNX output, got {len(onnx_output)}.")
    np_output = np.asarray(onnx_output[0], dtype=np.float32)
    max_abs_error = float(np.max(np.abs(np_output - torch_output)))
    if max_abs_error > tolerance:
        raise RuntimeError(
            "ONNX Runtime verification failed: "
            f"max abs error {max_abs_error:.6e} exceeds tolerance {tolerance:.6e}."
        )
    print(f"ONNX Runtime verification passed (max_abs_error={max_abs_error:.6e})")


if __name__ == "__main__":
    main()
