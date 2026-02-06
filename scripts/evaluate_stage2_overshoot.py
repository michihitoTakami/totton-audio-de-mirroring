"""CLI for Stage 2 overshoot evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from totton_audio_de_mirroring.stage2.overshoot import (
    OvershootEvaluation,
    evaluate_stage2_overshoot,
    load_stage_taps,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for overshoot evaluation.

    Physical Basis:
        A fixed and explicit measurement configuration is required to compare
        overshoot behavior across candidate filter designs reproducibly.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 2 FIR cascade overshoot (step/square)."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("cpp/configs"),
        help="Directory containing stage{i}_taps.txt files.",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=3,
        help="Number of 2x stages to load.",
    )
    parser.add_argument(
        "--source-sample-rate",
        type=int,
        default=88_200,
        help="Input sample rate before Stage 2.",
    )
    parser.add_argument(
        "--step-length",
        type=int,
        default=4_096,
        help="Input step probe length in samples.",
    )
    parser.add_argument(
        "--square-frequency-hz",
        type=float,
        default=1_000.0,
        help="Square-wave probe frequency.",
    )
    parser.add_argument(
        "--square-duration-sec",
        type=float,
        default=0.2,
        help="Square-wave probe duration in seconds.",
    )
    parser.add_argument(
        "--settle-fraction",
        type=float,
        default=0.75,
        help="Tail fraction used as settled region.",
    )
    parser.add_argument(
        "--reference-quantile",
        type=float,
        default=0.95,
        help="Quantile used for plateau reference.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON only.",
    )
    return parser.parse_args()


def _to_payload(result: OvershootEvaluation) -> dict[str, Any]:
    """Convert result dataclass into serializable payload.

    Physical Basis:
        Structured output keeps metric definitions explicit so design-space
        comparisons remain traceable and reproducible.
    """
    return {
        "output_sample_rate": result.output_sample_rate,
        "step": {
            "ratio": result.step.ratio,
            "peak": result.step.peak,
            "reference": result.step.reference,
        },
        "square": {
            "ratio": result.square.ratio,
            "peak": result.square.peak,
            "reference": result.square.reference,
        },
    }


def main() -> None:
    """Run Stage 2 overshoot evaluation and print results.

    Physical Basis:
        Overshoot constraints are transients-driven, so the CLI reports both
        step and square-wave metrics to avoid single-probe blind spots.
    """
    args = parse_args()
    stage_taps = load_stage_taps(config_dir=args.config_dir, num_stages=args.num_stages)
    result = evaluate_stage2_overshoot(
        stage_taps=stage_taps,
        source_sample_rate=args.source_sample_rate,
        step_length=args.step_length,
        square_frequency_hz=args.square_frequency_hz,
        square_duration_sec=args.square_duration_sec,
        settle_fraction=args.settle_fraction,
        reference_quantile=args.reference_quantile,
    )
    payload = _to_payload(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"Output sample rate: {payload['output_sample_rate']} Hz")
    print(
        "Step overshoot ratio: "
        f"{payload['step']['ratio']:.6f} "
        f"(peak={payload['step']['peak']:.6f}, ref={payload['step']['reference']:.6f})"
    )
    print(
        "Square overshoot ratio: "
        f"{payload['square']['ratio']:.6f} "
        f"(peak={payload['square']['peak']:.6f}, ref={payload['square']['reference']:.6f})"
    )


if __name__ == "__main__":
    main()
