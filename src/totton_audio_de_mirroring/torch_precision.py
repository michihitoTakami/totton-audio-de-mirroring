"""Torch execution-precision policy for CAPB DSP paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class TorchPrecisionRecord:
    """Machine-readable Torch precision settings.

    Physical Basis:
        TF32 truncates float32 convolution inputs to a shorter mantissa. That
        is useful for throughput-oriented neural networks, but CAPB also uses
        convolution for long fixed FIRs whose low-level distortion products
        are part of the audio contract.
    """

    device: str
    precision_mode: str
    allow_tf32: bool
    deterministic: bool
    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    gpu_name: str | None

    def to_dict(self) -> dict[str, str | bool | int | None]:
        """Return a JSON-compatible copy of the record."""
        return asdict(self)


def configure_torch_precision(
    device: str | torch.device,
    *,
    allow_tf32: bool = False,
    deterministic: bool = False,
) -> TorchPrecisionRecord:
    """Configure and describe Torch precision before CAPB execution.

    Args:
        device: Torch device used by the caller.
        allow_tf32: Whether CUDA convolution and matrix multiplication may use
            TF32. CAPB release paths must leave this disabled.
        deterministic: Whether to require deterministic Torch algorithms.

    Returns:
        The effective precision settings and runtime versions.

    Raises:
        ValueError: If a CUDA device is requested but unavailable.

    Physical Basis:
        A CAPB output is a convex blend of fixed FIR responses. Strict IEEE
        float32 accumulation keeps GPU FIR distortion near the CPU float32
        floor; TF32 can raise coherent harmonic products by tens of decibels.
    """
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA precision was requested but CUDA is unavailable.")

    effective_tf32 = allow_tf32 and torch_device.type == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = effective_tf32
    torch.backends.cudnn.allow_tf32 = effective_tf32
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)

    gpu_name = None
    if torch_device.type == "cuda":
        index = torch_device.index if torch_device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(index)
    return TorchPrecisionRecord(
        device=str(torch_device),
        precision_mode="tf32" if effective_tf32 else "strict_fp32",
        allow_tf32=effective_tf32,
        deterministic=deterministic,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        gpu_name=gpu_name,
    )
