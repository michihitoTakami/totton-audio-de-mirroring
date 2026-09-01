"""Export a CAPB Stage 1 checkpoint to ONNX (waveform contract).

The exported graph consumes/produces raw waveforms (``input_waveform`` ->
``output_waveform``, 2x samples) with a dynamic time axis, and embeds the
checkpoint's ``expected_input_rate`` as ONNX custom metadata so downstream
consumers (totton-audio-nn) can gate model selection per input-rate family.

Usage:
    uv run --with onnx --with onnxruntime python scripts/export_capb_to_onnx.py \
        --checkpoint data/checkpoints/capb_48k/run3/capb_best.pt \
        --output data/checkpoints/capb_48k/run3/capb_stage1_48k_to_96k.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from totton_audio_de_mirroring.models.capb import (
    CAPB,
    DEFAULT_TARGET_SAMPLE_RATE,
    capb_from_checkpoint,
)

DEFAULT_OPSET_VERSION = 17
DEFAULT_TOLERANCE = 1.0e-4
# Three deliberately unequal lengths exercise the dynamic time axis: one
# shorter than the controller stride context, one mid, one full second.
DEFAULT_VERIFY_LENGTHS = (4_096, 22_050, 48_000)
METADATA_KEY_EXPECTED_INPUT_RATE = "expected_input_rate"
METADATA_KEY_CUDA_PRECISION = "cuda_compute_precision"
METADATA_KEY_PROTOTYPE_PROFILE = "prototype_profile"
METADATA_KEY_PROTOTYPE_HASH = "prototype_hash"
METADATA_KEY_FIR_COMPUTE_DTYPE = "fir_compute_dtype"


def main() -> None:
    """Export and verify a CAPB checkpoint as an ONNX model."""
    args = _parse_args()
    checkpoint = _load_checkpoint(args.checkpoint)
    model = capb_from_checkpoint(checkpoint)

    expected_input_rate = _resolve_expected_input_rate(checkpoint)
    _export_onnx(model, args.output, args.opset_version)
    _write_metadata(args.output, expected_input_rate, model)
    _check_model(args.output)
    max_error = _verify_parity(
        model, args.output, tuple(args.verify_lengths), args.tolerance, args.seed
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "opset_version": args.opset_version,
        "expected_input_rate": expected_input_rate,
        "prototype_profile": model.prototype_profile,
        "prototype_hash": model.prototype_hash,
        "fir_compute_dtype": model.fir_compute_dtype,
        "verify_lengths": list(args.verify_lengths),
        "max_abs_error": max_error,
        "tolerance": args.tolerance,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset-version", type=int, default=DEFAULT_OPSET_VERSION)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--verify-lengths",
        type=int,
        nargs=3,
        default=list(DEFAULT_VERIFY_LENGTHS),
        metavar=("LEN1", "LEN2", "LEN3"),
        help="Three input lengths (samples) for PyTorch-vs-ONNX parity.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.opset_version < DEFAULT_OPSET_VERSION:
        parser.error(f"--opset-version must be >= {DEFAULT_OPSET_VERSION}.")
    if args.tolerance <= 0.0:
        parser.error("--tolerance must be positive.")
    if any(length <= 0 for length in args.verify_lengths):
        parser.error("--verify-lengths must be positive.")
    return args


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid checkpoint format: {path}")
    return checkpoint


def _resolve_expected_input_rate(checkpoint: dict[str, Any]) -> int:
    """Return the source rate the exported model expects.

    Physical Basis:
        Legacy (run9-era) checkpoints predate the rate metadata keys and
        are all 44.1k-family, so the fallback divides the default target
        rate by the fixed 2x ratio.
    """
    rate = checkpoint.get("expected_input_rate")
    if rate is not None:
        return int(rate)
    target = int(checkpoint.get("target_sample_rate", DEFAULT_TARGET_SAMPLE_RATE))
    return target // 2


def _export_onnx(model: CAPB, output: Path, opset_version: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 8_192, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        str(output),
        input_names=["input_waveform"],
        output_names=["output_waveform"],
        dynamic_axes={
            "input_waveform": {0: "batch", 1: "time"},
            "output_waveform": {0: "batch", 1: "time_out"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )


def _write_metadata(output: Path, expected_input_rate: int, model: CAPB) -> None:
    """Embed rate, prototype identity, and FIR precision as metadata."""
    try:
        import onnx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import onnx. Run via 'uv run --with onnx ...'."
        ) from exc

    onnx_model = onnx.load(str(output))
    metadata = {entry.key: entry.value for entry in onnx_model.metadata_props}
    metadata.update(_capb_metadata(expected_input_rate, model))
    del onnx_model.metadata_props[:]
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(onnx_model, str(output))


def _capb_metadata(expected_input_rate: int, model: CAPB) -> dict[str, str]:
    """Return the custom metadata required by the waveform contract.

    Physical Basis:
        Rate, FIR identity, and strict arithmetic mode bind the exported graph
        to the exact validated prototype bank and prevent silent cross-family
        or reduced-precision substitution downstream.
    """
    if expected_input_rate <= 0:
        raise ValueError("expected_input_rate must be positive.")
    return {
        METADATA_KEY_EXPECTED_INPUT_RATE: str(expected_input_rate),
        METADATA_KEY_CUDA_PRECISION: "strict_fp32",
        METADATA_KEY_PROTOTYPE_PROFILE: model.prototype_profile,
        METADATA_KEY_PROTOTYPE_HASH: model.prototype_hash,
        METADATA_KEY_FIR_COMPUTE_DTYPE: model.fir_compute_dtype,
    }


def _check_model(output: Path) -> None:
    import onnx  # type: ignore[import-not-found]

    onnx.checker.check_model(onnx.load(str(output)))


def _verify_parity(
    model: CAPB,
    output: Path,
    lengths: tuple[int, ...],
    tolerance: float,
    seed: int,
) -> float:
    """Compare PyTorch and onnxruntime outputs at several input lengths.

    Returns:
        Worst max-abs difference across all lengths.

    Raises:
        RuntimeError: If any length exceeds the tolerance.

    Physical Basis:
        The CAPB graph is pure convolution/softmax DSP, so PyTorch and ORT
        should agree to float32 rounding; multiple lengths exercise the
        dynamic time axis (controller striding and blend interpolation).
    """
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import onnxruntime. Run via 'uv run --with onnxruntime ...'."
        ) from exc

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    worst = 0.0
    for length in lengths:
        wave = rng.uniform(-0.5, 0.5, size=(1, length)).astype(np.float32)
        with torch.no_grad():
            torch_out = model(torch.from_numpy(wave)).numpy()
        (onnx_out,) = session.run(["output_waveform"], {"input_waveform": wave})
        if onnx_out.shape != torch_out.shape:
            raise RuntimeError(
                f"Shape mismatch at length {length}: "
                f"{onnx_out.shape} vs {torch_out.shape}."
            )
        error = float(np.max(np.abs(onnx_out - torch_out)))
        worst = max(worst, error)
        if error > tolerance:
            raise RuntimeError(
                f"Parity failure at length {length}: max abs error "
                f"{error:.3e} > tolerance {tolerance:.3e}."
            )
    return worst


if __name__ == "__main__":
    main()
