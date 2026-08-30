"""Compare CAPB strict-FP32 and TF32 CUDA distortion paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from report_capb_distortion import RateCase, _load_model, _measure_distortion
from totton_audio_de_mirroring.torch_precision import configure_torch_precision

_METRICS = (
    "thd_1khz_20khz_db",
    "smpte_imd_db",
    "ccif_imd_db",
    "added_am_sideband_db",
)


def main() -> None:
    """Measure both CUDA precision modes and write causal evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-44k1", type=Path, required=True)
    parser.add_argument("--checkpoint-48k", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = (
        RateCase("44k1", 44_100, args.checkpoint_44k1),
        RateCase("48k", 48_000, args.checkpoint_48k),
    )
    summary: dict[str, Any] = {}
    for mode, allow_tf32 in (("strict_fp32", False), ("tf32", True)):
        execution = configure_torch_precision("cuda", allow_tf32=allow_tf32)
        summary[mode] = {"execution": execution.to_dict()}
        for case in cases:
            model, bank = _load_model(case, "cuda")
            summary[mode][case.label] = _measure_distortion(case, model, bank)[
                "metrics"
            ]
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "precision_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        _plot(summary, args.output_dir / "tf32_precision_comparison.png")
        report = _render_markdown(summary)
        (args.output_dir / "precision_report.md").write_text(report, encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Failed to write precision report: {error}") from error
    print(report)


def _plot(summary: dict[str, Any], output_path: Path) -> None:
    """Plot CAPB coherent-line metrics for both CUDA precision modes."""
    positions = np.arange(len(_METRICS), dtype=np.float64)
    labels = ("THD", "SMPTE", "CCIF", "AM spur")
    figure, axis = plt.subplots(figsize=(11, 5))
    series = (
        ("44k1 strict", "strict_fp32", "44k1", -0.27),
        ("44k1 TF32", "tf32", "44k1", -0.09),
        ("48k strict", "strict_fp32", "48k", 0.09),
        ("48k TF32", "tf32", "48k", 0.27),
    )
    for label, mode, family, offset in series:
        values = summary[mode][family]["capb"]
        axis.bar(
            positions + offset,
            [values[metric] for metric in _METRICS],
            width=0.17,
            label=label,
        )
    axis.set(
        xticks=positions,
        xticklabels=labels,
        ylabel="distortion or spur level (dBc)",
        ylim=(-180.0, -60.0),
        title="CUDA convolution precision (higher is worse)",
    )
    axis.grid(alpha=0.25, axis="y")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _render_markdown(summary: dict[str, Any]) -> str:
    """Render the measured TF32 root-cause comparison."""
    rows = []
    for family in ("44k1", "48k"):
        strict = summary["strict_fp32"][family]["capb"]
        tf32 = summary["tf32"][family]["capb"]
        for metric in _METRICS:
            rows.append(
                f"| {family} | {metric} | {strict[metric]:.2f} dB | "
                f"{tf32[metric]:.2f} dB | {tf32[metric] - strict[metric]:.2f} dB |"
            )
    return "\n".join(
        [
            "# CAPB CUDA precision investigation",
            "",
            "| Family | Metric | strict FP32 | TF32 | TF32 degradation |",
            "|---|---|---:|---:|---:|",
            *rows,
            "",
            "The checkpoints and prototype banks are unchanged between rows. The "
            "measured difference is caused by CUDA convolution precision, not by a "
            "different controller checkpoint.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
