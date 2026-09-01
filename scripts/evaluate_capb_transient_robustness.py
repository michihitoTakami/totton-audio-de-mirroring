"""Evaluate CAPB pre-echo across controller and OLA offsets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from totton_audio_de_mirroring.inference.chunk_processor import (
    ChunkProcessingConfig,
    HannOverlapAddStreamer,
    iterate_chunk_frames,
)
from totton_audio_de_mirroring.inference.pipeline import (
    CAPBStage1Processor,
    ReferenceStage1Processor,
    Stage1Processor,
)
from totton_audio_de_mirroring.models.capb import (
    SUPPORTED_FIR_COMPUTE_DTYPES,
    capb_candidate_from_checkpoint,
)
from totton_audio_de_mirroring.models.proto_bank import supported_prototype_profiles
from totton_audio_de_mirroring.torch_precision import configure_torch_precision

_SOURCE_RATE = 44_100
_TARGET_RATE = 88_200
_AMPLITUDE = 0.5
_ECHO_GUARD_MS = 0.5
_ECHO_WINDOW_MS = 3.5
_FAR_ECHO_GUARD_MS = 4.0
_FAR_ECHO_WINDOW_MS = 8.0
_PRE_ECHO_RATIO_MAX = 1.44
_PRE_ECHO_FLOOR_REL = 1.0e-3


def parse_args() -> argparse.Namespace:
    """Parse robustness-evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--rate-family", choices=("44k1", "48k"), default="44k1")
    parser.add_argument(
        "--prototype-profile",
        choices=supported_prototype_profiles(),
        default=None,
    )
    parser.add_argument(
        "--fir-compute-dtype",
        choices=SUPPORTED_FIR_COMPUTE_DTYPES,
        default="float32",
    )
    return parser.parse_args()


def evaluate_robustness(
    capb: CAPBStage1Processor,
    source_rate: int = _SOURCE_RATE,
    target_rate: int = _TARGET_RATE,
) -> dict[str, Any]:
    """Evaluate direct controller phases and production OLA boundaries.

    Physical Basis:
        A strided controller can react differently to the same impulse at
        each of its 64 input phases. Production Hann overlap-add adds a second
        coordinate system at chunk hops, so both must satisfy the same G2b
        energy bound used by the frozen probe gates.
    """
    if source_rate <= 0 or target_rate != 2 * source_rate:
        raise ValueError("target_rate must be exactly 2x a positive source_rate.")
    direct = _evaluate_offsets(
        offsets=range(64),
        event_position=lambda offset, _hop: source_rate // 2 + offset,
        candidate=lambda signal: capb.process(signal, source_rate, target_rate),
        reference=lambda signal: ReferenceStage1Processor().process(
            signal, source_rate, target_rate
        ),
        source_rate=source_rate,
        target_rate=target_rate,
    )
    chunking = ChunkProcessingConfig(
        sample_rate=source_rate,
        chunk_duration_sec=0.25,
        overlap_ratio=0.5,
        window="hann",
    )

    def direct_capb(signal: np.ndarray) -> np.ndarray:
        return _process_chunked(signal, capb, chunking)

    reference_processor = ReferenceStage1Processor()

    def chunked_reference(signal: np.ndarray) -> np.ndarray:
        return _process_chunked(signal, reference_processor, chunking)

    boundary = _evaluate_offsets(
        offsets=range(-32, 32),
        event_position=lambda offset, hop: 2 * hop + offset,
        candidate=direct_capb,
        reference=chunked_reference,
        hop_samples=chunking.hop_samples,
        source_rate=source_rate,
        target_rate=target_rate,
    )
    return {
        "all_passed": direct["all_passed"] and boundary["all_passed"],
        "direct_controller_phase": direct,
        "chunk_boundary": boundary,
        "thresholds": {
            "pre_echo_energy_ratio_max": _PRE_ECHO_RATIO_MAX,
            "pre_echo_floor_rel": _PRE_ECHO_FLOOR_REL,
            "guard_ms": _ECHO_GUARD_MS,
            "window_ms": _ECHO_WINDOW_MS,
            "far_guard_ms": _FAR_ECHO_GUARD_MS,
            "far_window_ms": _FAR_ECHO_WINDOW_MS,
        },
    }


def _evaluate_offsets(
    *,
    offsets: range,
    event_position: Callable[[int, int], int],
    candidate: Callable[[np.ndarray], np.ndarray],
    reference: Callable[[np.ndarray], np.ndarray],
    hop_samples: int = 0,
    source_rate: int = _SOURCE_RATE,
    target_rate: int = _TARGET_RATE,
) -> dict[str, Any]:
    """Measure pre-echo margin for a set of source-sample offsets.

    Args:
        offsets: Source-sample offsets to evaluate.
        event_position: Maps an offset and optional hop size to an event index.
        candidate: Candidate 2x processing function.
        reference: Reference 2x processing function.
        hop_samples: Source-rate OLA hop size passed to event_position.
        source_rate: Input sample rate in Hz.
        target_rate: Output sample rate in Hz; must equal 2x source_rate.

    Returns:
        Serializable worst-case result and one row per offset.

    Raises:
        ValueError: If the rate contract, event position, or output is invalid.

    Physical Basis:
        Exhaustive controller and OLA phase offsets prevent a favorable event
        alignment from hiding a pre-echo regression.
    """
    if source_rate <= 0 or target_rate != 2 * source_rate:
        raise ValueError("target_rate must be exactly 2x a positive source_rate.")
    rows: list[dict[str, float | int | bool]] = []
    for offset in offsets:
        source = np.zeros(source_rate, dtype=np.float64)
        event = event_position(offset, hop_samples)
        if not 0 <= event < source.size:
            raise ValueError(f"Event position {event} is outside the source buffer.")
        source[event] = _AMPLITUDE
        candidate_output = candidate(source)
        reference_output = reference(source)
        if candidate_output.size != source.size * 2:
            raise ValueError("Candidate output length must equal 2x input length.")
        if not np.all(np.isfinite(candidate_output)):
            raise ValueError("Candidate output contains non-finite values.")
        center = event * 2
        after = _pre_echo_energy(candidate_output, center, target_rate)
        before = _pre_echo_energy(reference_output, center, target_rate)
        far_after = _pre_echo_energy(
            candidate_output,
            center,
            target_rate,
            guard_ms=_FAR_ECHO_GUARD_MS,
            window_ms=_FAR_ECHO_WINDOW_MS,
        )
        far_before = _pre_echo_energy(
            reference_output,
            center,
            target_rate,
            guard_ms=_FAR_ECHO_GUARD_MS,
            window_ms=_FAR_ECHO_WINDOW_MS,
        )
        threshold = max(
            _PRE_ECHO_RATIO_MAX * before,
            (_PRE_ECHO_FLOOR_REL * _AMPLITUDE) ** 2,
        )
        far_threshold = max(
            _PRE_ECHO_RATIO_MAX * far_before,
            (_PRE_ECHO_FLOOR_REL * _AMPLITUDE) ** 2,
        )
        near_margin_db = _margin_db(after, threshold)
        far_margin_db = _margin_db(far_after, far_threshold)
        rows.append(
            {
                "offset_samples": offset,
                "pre_echo_energy_before": before,
                "pre_echo_energy_after": after,
                "threshold": threshold,
                "margin_db": max(near_margin_db, far_margin_db),
                "near_margin_db": near_margin_db,
                "far_pre_echo_energy_before": far_before,
                "far_pre_echo_energy_after": far_after,
                "far_threshold": far_threshold,
                "far_margin_db": far_margin_db,
                "passed": after <= threshold and far_after <= far_threshold,
            }
        )
    worst = max(rows, key=lambda row: float(row["margin_db"]))
    return {
        "all_passed": all(bool(row["passed"]) for row in rows),
        "worst": worst,
        "rows": rows,
    }


def _process_chunked(
    signal: np.ndarray,
    processor: Stage1Processor,
    chunking: ChunkProcessingConfig,
) -> np.ndarray:
    """Run the production Stage 1 Hann-OLA topology without Stage 2.

    Args:
        signal: Source-rate mono waveform.
        processor: Stage 1 processor under evaluation.
        chunking: Production-compatible Hann-OLA configuration.

    Returns:
        Contiguous 2x waveform trimmed to the exact rate-converted length.

    Physical Basis:
        Reusing production chunk and overlap geometry exposes boundary-phase
        behavior that direct whole-buffer inference cannot measure.
    """
    streamer = HannOverlapAddStreamer(
        chunk_samples=chunking.chunk_samples * 2,
        overlap_samples=chunking.overlap_samples * 2,
        window=chunking.window,
    )
    segments: list[np.ndarray] = []
    for frame in iterate_chunk_frames(
        signal,
        chunk_samples=chunking.chunk_samples,
        overlap_samples=chunking.overlap_samples,
    ):
        output = processor.process(
            frame.samples, chunking.sample_rate, chunking.sample_rate * 2
        )
        piece = streamer.process_chunk(output)
        if piece.size:
            segments.append(np.asarray(piece, dtype=np.float64))
    tail = streamer.finalize()
    if tail.size:
        segments.append(np.asarray(tail, dtype=np.float64))
    return np.concatenate(segments)[: signal.size * 2]


def _pre_echo_energy(
    signal: np.ndarray,
    center: int,
    sample_rate: int,
    *,
    guard_ms: float = _ECHO_GUARD_MS,
    window_ms: float = _ECHO_WINDOW_MS,
) -> float:
    """Measure mean-square pre-event energy in a physical time window."""
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("signal must be a non-empty 1D array.")
    if sample_rate <= 0 or guard_ms < 0.0 or window_ms <= 0.0:
        raise ValueError("Rate and echo-window values must be positive.")
    guard = int(round(guard_ms * sample_rate / 1_000.0))
    window = int(round(window_ms * sample_rate / 1_000.0))
    if center - guard - window < 0 or center - guard > signal.size:
        raise ValueError("signal does not contain the complete pre-echo window.")
    samples = signal[center - guard - window : center - guard]
    return float(np.mean(np.square(samples)))


def _margin_db(value: float, threshold: float) -> float:
    """Return a finite decibel margin relative to a positive threshold."""
    if value < 0.0 or threshold <= 0.0:
        raise ValueError("value must be non-negative and threshold positive.")
    return float(10.0 * np.log10(max(value, 1.0e-300) / threshold))


def _plot_report(result: dict[str, Any], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    for axis, key, title in (
        (axes[0], "direct_controller_phase", "Direct controller phase"),
        (axes[1], "chunk_boundary", "Hann-OLA hop boundary"),
    ):
        rows = result[key]["rows"]
        axis.plot(
            [row["offset_samples"] for row in rows],
            [row["margin_db"] for row in rows],
            marker="o",
            markersize=3,
        )
        axis.axhline(0.0, color="red", linestyle="--", label="G2b threshold")
        axis.set(title=title, xlabel="offset (source samples)", ylabel="margin (dB)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _render_markdown(result: dict[str, Any], checkpoint: Path) -> str:
    direct = result["direct_controller_phase"]
    boundary = result["chunk_boundary"]
    return f"""# CAPB transient robustness

- Checkpoint: `{checkpoint}`
- Overall: **{"PASS" if result["all_passed"] else "FAIL"}**
- Direct 64-phase worst: {direct["worst"]["margin_db"]:.2f} dB at offset {direct["worst"]["offset_samples"]}
- OLA-boundary worst: {boundary["worst"]["margin_db"]:.2f} dB at offset {boundary["worst"]["offset_samples"]}

Negative margin is below the unchanged G2b threshold. Each offset must pass
both the canonical 0.5--4 ms window and the supplemental 4--12 ms tail window.
"""


def main() -> None:
    """Run robustness evaluation and write reproducible evidence."""
    args = parse_args()
    precision = configure_torch_precision(args.device, allow_tf32=args.allow_tf32)
    source_rate = 44_100 if args.rate_family == "44k1" else 48_000
    target_rate = source_rate * 2
    capb = _load_candidate(
        args.checkpoint, args.device, args.prototype_profile, args.fir_compute_dtype
    )
    result = evaluate_robustness(capb, source_rate, target_rate)
    result["execution"] = precision.to_dict()
    result["candidate_identity"] = {
        "prototype_profile": getattr(capb.model, "prototype_profile", "unknown"),
        "prototype_hash": getattr(capb.model, "prototype_hash", "unknown"),
        "fir_compute_dtype": getattr(capb.model, "fir_compute_dtype", "unknown"),
    }
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "robustness.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        (args.output_dir / "robustness.md").write_text(
            _render_markdown(result, args.checkpoint), encoding="utf-8"
        )
        _plot_report(result, args.output_dir / "offset_margins.png")
    except OSError as error:
        raise RuntimeError(f"Failed to write robustness report: {error}") from error
    print(_render_markdown(result, args.checkpoint))
    if not result["all_passed"]:
        raise SystemExit(1)


def _load_candidate(
    checkpoint_path: Path,
    device: str,
    prototype_profile: str | None,
    fir_compute_dtype: str,
) -> CAPBStage1Processor:
    """Load a release checkpoint or a controller-only FIR candidate.

    Args:
        checkpoint_path: Controller checkpoint to load.
        device: Torch execution device.
        prototype_profile: Optional explicit experimental FIR profile.
        fir_compute_dtype: Fixed-FIR arithmetic dtype for an experiment.

    Returns:
        Stage 1 processor containing the requested model.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        RuntimeError: If checkpoint I/O or deserialization fails.

    Physical Basis:
        Keeping controller weights fixed while replacing only the named FIR
        bank exposes phase-offset and OLA-tail effects attributable to taps.
    """
    from totton_audio_de_mirroring.inference.pipeline import (
        load_capb_stage1_processor,
    )

    if prototype_profile is None:
        return load_capb_stage1_processor(
            checkpoint_path=checkpoint_path,
            device=device,
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Failed to load checkpoint: {error}") from error
    model = capb_candidate_from_checkpoint(
        checkpoint,
        prototype_profile=prototype_profile,
        fir_compute_dtype=fir_compute_dtype,
    )
    torch_device = torch.device(device)
    return CAPBStage1Processor(model=model.to(torch_device), device=torch_device)


if __name__ == "__main__":
    main()
