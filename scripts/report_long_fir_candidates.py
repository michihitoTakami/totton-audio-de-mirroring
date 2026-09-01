"""Report structural quality and long-tail risks of CAPB FIR profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional

from totton_audio_de_mirroring.evaluation.long_fir import (
    evaluate_long_echo,
    evaluate_phase_alignment,
    validate_phase_alignment,
)
from totton_audio_de_mirroring.models.proto_bank import (
    RELEASE_PROTOTYPE_PROFILE,
    build_prototype_bank_for_profile,
    supported_prototype_profiles,
    upsample_with_kernel,
)

RATE_FAMILIES: dict[str, tuple[int, int]] = {
    "44k1": (44_100, 88_200),
    "48k": (48_000, 96_000),
}
DEFAULT_PROFILES = tuple(
    profile
    for profile in supported_prototype_profiles()
    if profile != RELEASE_PROTOTYPE_PROFILE
)
FFT_SIZE = 1 << 20
IMAGE_TARGET_DB = -130.0
RESPONSE_20K_TOLERANCE_DB = 1.0e-3


def main() -> None:
    """Generate a machine-readable long-FIR structural report."""
    args = _parse_args()
    payload = {
        "criteria": {
            "sharp_image_max_db": IMAGE_TARGET_DB,
            "response_20k_abs_max_db": RESPONSE_20K_TOLERANCE_DB,
            "fixed_fir_float32_error_is_diagnostic_only": True,
        },
        "families": {
            family: _evaluate_family(family, tuple(args.profiles))
            for family in args.rate_family
        },
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        except OSError as exc:
            raise RuntimeError(f"Failed to write report {args.output}: {exc}") from exc
    print(rendered, end="")


def _parse_args() -> argparse.Namespace:
    """Parse report command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rate-family",
        choices=tuple(RATE_FAMILIES),
        action="append",
        default=None,
        help="Rate family to inspect; repeat to select both.",
    )
    parser.add_argument(
        "--profiles",
        choices=DEFAULT_PROFILES,
        nargs="+",
        default=list(DEFAULT_PROFILES),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.rate_family is None:
        args.rate_family = list(RATE_FAMILIES)
    return args


def _evaluate_family(family: str, profiles: tuple[str, ...]) -> dict[str, Any]:
    """Evaluate release and experimental banks for one rate family."""
    source_rate, target_rate = RATE_FAMILIES[family]
    baseline = build_prototype_bank_for_profile(target_rate, RELEASE_PROTOTYPE_PROFILE)
    baseline_sharp = baseline.kernels[baseline.names.index("sharp")]
    rows: dict[str, Any] = {}
    for profile in (RELEASE_PROTOTYPE_PROFILE, *profiles):
        bank = build_prototype_bank_for_profile(target_rate, profile)
        phase = evaluate_phase_alignment(bank)
        phase_passed = True
        try:
            validate_phase_alignment(phase)
        except ValueError:
            phase_passed = False
        sharp = bank.kernels[bank.names.index("sharp")]
        response64 = _response_metrics(sharp, target_rate)
        response32 = _response_metrics(
            sharp.astype(np.float32).astype(np.float64), target_rate
        )
        rows[profile] = {
            "kernel_length": int(sharp.size),
            "support_ms": float(1_000.0 * (sharp.size - 1) / target_rate),
            "phase": phase.to_dict(),
            "phase_passed": phase_passed,
            "response_float64": response64,
            "response_float32_coefficients": response32,
            "fixed_fir_float32_error": _fixed_fir_float32_error(sharp, source_rate),
            "sharp_echo": _sharp_echo(sharp, source_rate, target_rate),
            "zero_padding_control_magnitude_error": _padding_control_error(
                baseline_sharp, sharp.size
            ),
            "structural_passed": bool(
                phase_passed
                and response32["image_max_db"] <= IMAGE_TARGET_DB
                and abs(response32["response_20k_db"]) <= RESPONSE_20K_TOLERANCE_DB
            ),
        }
    _add_release_comparisons(rows)
    return {
        "source_sample_rate": source_rate,
        "target_sample_rate": target_rate,
        "release_kernel_length": int(baseline.kernels.shape[1]),
        "profiles": rows,
    }


def _add_release_comparisons(rows: dict[str, Any]) -> None:
    """Annotate experimental rows with release-relative Go/No-Go metrics."""
    release = rows[RELEASE_PROTOTYPE_PROFILE]
    release_image = release["response_float32_coefficients"]["image_max_db"]
    release_floor = release["fixed_fir_float32_error"]["relative_rms_db"]
    release["screening_eligible"] = False
    for profile, row in rows.items():
        if profile == RELEASE_PROTOTYPE_PROFILE:
            continue
        image_improvement = (
            release_image - row["response_float32_coefficients"]["image_max_db"]
        )
        floor_improvement = (
            release_floor - row["fixed_fir_float32_error"]["relative_rms_db"]
        )
        row["image_improvement_db"] = image_improvement
        row["fixed_fir_floor_improvement_db"] = floor_improvement
        row["screening_eligible"] = bool(row["structural_passed"])


def _response_metrics(kernel: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Measure normalized passband and image response of one sharp kernel."""
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0 / sample_rate)
    response = np.abs(np.fft.rfft(kernel, n=FFT_SIZE)) / 2.0
    input_nyquist = sample_rate / 4.0
    passband = frequencies <= 19_000.0
    image = frequencies >= input_nyquist + 500.0
    index_20k = int(np.argmin(np.abs(frequencies - 20_000.0)))
    return {
        "passband_deviation_db": _db(np.max(np.abs(response[passband] - 1.0))),
        "response_20k_db": _db(response[index_20k]),
        "image_max_db": _db(np.max(response[image])),
    }


def _sharp_echo(
    kernel: np.ndarray, source_rate: int, target_rate: int
) -> dict[str, float]:
    """Measure the fixed sharp endpoint around an isolated impulse."""
    source = np.zeros(source_rate, dtype=np.float64)
    center = source.size // 2
    source[center] = 0.5
    output = upsample_with_kernel(source, kernel, 2)
    return evaluate_long_echo(
        output, center_index=center * 2, sample_rate=target_rate
    ).to_dict()


def _padding_control_error(kernel: np.ndarray, target_length: int) -> float:
    """Return magnitude-response error after symmetric zero-padding."""
    if target_length < kernel.size or (target_length - kernel.size) % 2 != 0:
        raise ValueError("target_length must preserve the kernel center.")
    pad = (target_length - kernel.size) // 2
    padded = np.pad(kernel, (pad, pad))
    original_magnitude = np.abs(np.fft.rfft(kernel, n=FFT_SIZE))
    padded_magnitude = np.abs(np.fft.rfft(padded, n=FFT_SIZE))
    return float(np.max(np.abs(original_magnitude - padded_magnitude)))


def _fixed_fir_float32_error(kernel: np.ndarray, source_rate: int) -> dict[str, float]:
    """Measure strict-FP32 FIR error against the float64 implementation.

    Args:
        kernel: Float64 sharp-prototype coefficients.
        source_rate: Input rate used to synthesize the deterministic probe.

    Returns:
        Relative RMS and peak errors in decibels.

    Physical Basis:
        Comparing identical convolution topology in float32 and float64
        separates arithmetic/quantization floor from interpolation-image
        leakage. Longer kernels need not improve this floor because they
        accumulate more rounded products.
    """
    num_samples = 8_192
    time = np.arange(num_samples, dtype=np.float64) / source_rate
    source = sum(
        amplitude * np.sin(2.0 * np.pi * frequency * time)
        for frequency, amplitude in ((997.0, 0.4), (7_000.0, 0.2), (19_000.0, 0.1))
    )
    reference = upsample_with_kernel(source, kernel, 2)
    source_tensor = torch.from_numpy(source.astype(np.float32)).reshape(1, 1, -1)
    stuffed = torch.zeros(1, 1, num_samples * 2, dtype=torch.float32)
    stuffed[:, :, ::2] = source_tensor
    kernel_tensor = torch.from_numpy(kernel.astype(np.float32)).reshape(1, 1, -1)
    output = torch_functional.conv1d(stuffed, kernel_tensor, padding=kernel.size // 2)
    candidate = np.asarray(output.squeeze(), dtype=np.float64)
    trim = kernel.size // 2
    error = candidate[trim:-trim] - reference[trim:-trim]
    reference_region = reference[trim:-trim]
    return {
        "relative_rms_db": _db(
            np.sqrt(np.mean(np.square(error)))
            / np.sqrt(np.mean(np.square(reference_region)))
        ),
        "relative_peak_db": _db(
            np.max(np.abs(error)) / np.max(np.abs(reference_region))
        ),
    }


def _db(value: float | np.floating) -> float:
    """Convert a non-negative linear magnitude to decibels."""
    return float(20.0 * np.log10(max(float(value), 1.0e-300)))


if __name__ == "__main__":
    main()
