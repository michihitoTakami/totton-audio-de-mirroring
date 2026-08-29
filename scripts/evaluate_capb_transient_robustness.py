"""Evaluate 44.1 kHz CAPB pre-echo across controller and OLA offsets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from totton_audio_de_mirroring.inference.chunk_processor import (
    ChunkProcessingConfig,
    HannOverlapAddStreamer,
    iterate_chunk_frames,
)
from totton_audio_de_mirroring.inference.pipeline import (
    CAPBStage1Processor,
    ReferenceStage1Processor,
    Stage1Processor,
    load_capb_stage1_processor,
)

_SOURCE_RATE = 44_100
_TARGET_RATE = 88_200
_AMPLITUDE = 0.5
_ECHO_GUARD_MS = 0.5
_ECHO_WINDOW_MS = 3.5
_PRE_ECHO_RATIO_MAX = 1.44
_PRE_ECHO_FLOOR_REL = 1.0e-3


def parse_args() -> argparse.Namespace:
    """Parse robustness-evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def evaluate_robustness(
    capb: CAPBStage1Processor,
) -> dict[str, Any]:
    """Evaluate direct controller phases and production OLA boundaries.

    Physical Basis:
        A strided controller can react differently to the same impulse at
        each of its 64 input phases. Production Hann overlap-add adds a second
        coordinate system at chunk hops, so both must satisfy the same G2b
        energy bound used by the frozen probe gates.
    """
    direct = _evaluate_offsets(
        offsets=range(64),
        event_position=lambda offset, _hop: _SOURCE_RATE // 2 + offset,
        candidate=lambda signal: capb.process(signal, _SOURCE_RATE, _TARGET_RATE),
        reference=lambda signal: ReferenceStage1Processor().process(
            signal, _SOURCE_RATE, _TARGET_RATE
        ),
    )
    chunking = ChunkProcessingConfig(
        sample_rate=_SOURCE_RATE,
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
        },
    }


def _evaluate_offsets(
    *,
    offsets: range,
    event_position: Callable[[int, int], int],
    candidate: Callable[[np.ndarray], np.ndarray],
    reference: Callable[[np.ndarray], np.ndarray],
    hop_samples: int = 0,
) -> dict[str, Any]:
    rows: list[dict[str, float | int | bool]] = []
    for offset in offsets:
        source = np.zeros(_SOURCE_RATE, dtype=np.float64)
        event = event_position(offset, hop_samples)
        source[event] = _AMPLITUDE
        candidate_output = candidate(source)
        reference_output = reference(source)
        if candidate_output.size != source.size * 2:
            raise ValueError("Candidate output length must equal 2x input length.")
        if not np.all(np.isfinite(candidate_output)):
            raise ValueError("Candidate output contains non-finite values.")
        center = event * 2
        after = _pre_echo_energy(candidate_output, center, _TARGET_RATE)
        before = _pre_echo_energy(reference_output, center, _TARGET_RATE)
        threshold = max(
            _PRE_ECHO_RATIO_MAX * before,
            (_PRE_ECHO_FLOOR_REL * _AMPLITUDE) ** 2,
        )
        rows.append(
            {
                "offset_samples": offset,
                "pre_echo_energy_before": before,
                "pre_echo_energy_after": after,
                "threshold": threshold,
                "margin_db": 10.0 * np.log10(max(after, 1.0e-300) / threshold),
                "passed": after <= threshold,
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
    """Run the production Stage 1 Hann-OLA topology without Stage 2."""
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
        output = processor.process(frame.samples, _SOURCE_RATE, _TARGET_RATE)
        piece = streamer.process_chunk(output)
        if piece.size:
            segments.append(np.asarray(piece, dtype=np.float64))
    tail = streamer.finalize()
    if tail.size:
        segments.append(np.asarray(tail, dtype=np.float64))
    return np.concatenate(segments)[: signal.size * 2]


def _pre_echo_energy(signal: np.ndarray, center: int, sample_rate: int) -> float:
    guard = int(round(_ECHO_GUARD_MS * sample_rate / 1_000.0))
    window = int(round(_ECHO_WINDOW_MS * sample_rate / 1_000.0))
    samples = signal[center - guard - window : center - guard]
    return float(np.mean(np.square(samples)))


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

Negative margin is below the unchanged G2b threshold.
"""


def main() -> None:
    """Run robustness evaluation and write reproducible evidence."""
    args = parse_args()
    capb = load_capb_stage1_processor(
        checkpoint_path=args.checkpoint, device=args.device
    )
    result = evaluate_robustness(capb)
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


if __name__ == "__main__":
    main()
