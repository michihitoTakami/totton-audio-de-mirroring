"""Visualize CAPB impulse responses and controller routing for both families."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from totton_audio_de_mirroring.data.reference import upsample_bessel_reference
from totton_audio_de_mirroring.models.capb import CAPB, capb_from_checkpoint
from totton_audio_de_mirroring.models.proto_bank import (
    PrototypeBank,
    build_prototype_bank,
    prototype_specs_for_target_rate,
    upsample_with_kernel,
)
from totton_audio_de_mirroring.torch_precision import configure_torch_precision

_BESSEL_CUTOFF_HZ = 20_000.0
_BESSEL_ORDER = 6
_AMPLITUDE = 0.5
_REFERENCE_ZORDER = 2.0
_CAPB_ZORDER = 10.0


def _comparison_zorder(name: str) -> float:
    """Return a drawing priority that keeps CAPB above references."""
    return _CAPB_ZORDER if name == "capb" else _REFERENCE_ZORDER


@dataclass(frozen=True)
class RateCase:
    """Checkpoint and sample-rate pairing used by the report."""

    label: str
    source_rate: int
    checkpoint: Path

    @property
    def target_rate(self) -> int:
        """Return the fixed Stage 1 output rate."""
        return self.source_rate * 2


def parse_args() -> argparse.Namespace:
    """Parse report inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-44k1", type=Path, required=True)
    parser.add_argument("--checkpoint-48k", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-tf32", action="store_true")
    return parser.parse_args()


def _load_model(case: RateCase, device: str) -> tuple[CAPB, PrototypeBank]:
    """Load a rate-validated controller and its fixed prototype bank."""
    if not case.checkpoint.is_file():
        raise FileNotFoundError(f"CAPB checkpoint not found: {case.checkpoint}")
    try:
        state = torch.load(case.checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Failed to load {case.checkpoint}: {error}") from error
    expected_rate = int(state.get("expected_input_rate", case.source_rate))
    if expected_rate != case.source_rate:
        raise ValueError(
            f"Checkpoint expects {expected_rate} Hz, not {case.source_rate} Hz."
        )
    model = capb_from_checkpoint(state).to(torch.device(device)).eval()
    bank = build_prototype_bank(
        prototype_specs_for_target_rate(case.target_rate), case.target_rate
    )
    return model, bank


def _run_impulse(
    case: RateCase, model: CAPB, bank: PrototypeBank
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Run one isolated source-rate impulse through all relevant paths.

    Physical Basis:
        The impulse is created at the input rate, matching the frozen G2b
        probe. Generating a target-rate impulse and decimating it would smear
        the event and conceal controller-phase sensitivity.
    """
    source = np.zeros(case.source_rate, dtype=np.float64)
    event = source.size // 2
    source[event] = _AMPLITUDE
    with torch.no_grad():
        tensor = torch.from_numpy(source.astype(np.float32)).unsqueeze(0)
        tensor = tensor.to(next(model.parameters()).device)
        output, weights = model(tensor, return_weights=True)
    outputs = {
        "bessel": upsample_bessel_reference(
            source,
            case.source_rate,
            case.target_rate,
            _BESSEL_CUTOFF_HZ,
            _BESSEL_ORDER,
        ),
        "capb": np.asarray(output.squeeze(0).cpu(), dtype=np.float64),
        "sharp": upsample_with_kernel(source, bank.kernels[0], 2),
        "gentle": upsample_with_kernel(source, bank.kernels[-1], 2),
    }
    return outputs, np.asarray(weights.squeeze(0).cpu(), dtype=np.float64), event * 2


def _metrics(
    outputs: dict[str, np.ndarray], center: int, sample_rate: int
) -> dict[str, dict[str, float]]:
    """Measure gate-window pre-echo and symmetric impulse energy."""
    guard = round(0.0005 * sample_rate)
    window = round(0.0035 * sample_rate)
    result: dict[str, dict[str, float]] = {}
    for name, output in outputs.items():
        pre = output[center - guard - window : center - guard]
        local = output[center - window : center + window + 1]
        result[name] = {
            "peak": float(np.max(np.abs(local))),
            "pre_echo_mean_square": float(np.mean(np.square(pre))),
            "local_energy": float(np.sum(np.square(local))),
        }
    return result


def _plot(
    case: RateCase,
    outputs: dict[str, np.ndarray],
    weights: np.ndarray,
    center: int,
    output_path: Path,
) -> None:
    """Plot the impulse neighborhood and controller blend trajectory."""
    radius = round(0.004 * case.target_rate)
    sample_slice = slice(center - radius, center + radius + 1)
    time_ms = (
        (np.arange(center - radius, center + radius + 1) - center)
        / case.target_rate
        * 1_000.0
    )
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    colors = {
        "bessel": "tab:blue",
        "capb": "tab:orange",
        "sharp": "tab:green",
        "gentle": "tab:red",
    }
    for name in ("bessel", "sharp", "gentle", "capb"):
        axes[0].plot(
            time_ms,
            outputs[name][sample_slice],
            label=name,
            color=colors[name],
            zorder=_comparison_zorder(name),
        )
    axes[0].axvspan(-4.0, -0.5, color="gray", alpha=0.12, label="G2b window")
    axes[0].set(
        title=f"{case.label}: isolated input-rate impulse response",
        xlabel="time from impulse (ms)",
        ylabel="amplitude",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=3)

    frame_time = np.linspace(
        -source_duration_ms(case, center),
        source_duration_ms(case, outputs["capb"].size - center),
        weights.shape[-1],
    )
    for index, label in enumerate(("sharp", "middle", "gentle")):
        axes[1].plot(frame_time, weights[index], label=label)
    axes[1].set_xlim(-4.0, 4.0)
    axes[1].set(
        title="Controller convex weights",
        xlabel="time from impulse (ms)",
        ylabel="weight",
        ylim=(-0.02, 1.02),
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def source_duration_ms(case: RateCase, output_samples: int) -> float:
    """Convert target-rate samples to milliseconds."""
    return output_samples / case.target_rate * 1_000.0


def _render_markdown(summary: dict[str, Any]) -> str:
    """Render a compact impulse-response report."""
    lines = ["# CAPB impulse-response report", ""]
    for label, rate_result in summary.items():
        if label == "execution":
            continue
        metrics = rate_result["metrics"]
        lines.extend(
            [
                f"## {label}",
                "",
                f"- Checkpoint: `{rate_result['checkpoint']}`",
                f"- CAPB G2b-window energy: {metrics['capb']['pre_echo_mean_square']:.4e}",
                f"- Gentle G2b-window energy: {metrics['gentle']['pre_echo_mean_square']:.4e}",
                f"- Sharp G2b-window energy: {metrics['sharp']['pre_echo_mean_square']:.4e}",
                "",
                f"![{label} impulse response]({label}/impulse_response.png)",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    """Generate deterministic impulse plots, metrics, and report."""
    args = parse_args()
    precision = configure_torch_precision(args.device, allow_tf32=args.allow_tf32)
    cases = (
        RateCase("44k1", 44_100, args.checkpoint_44k1),
        RateCase("48k", 48_000, args.checkpoint_48k),
    )
    summary: dict[str, Any] = {"execution": precision.to_dict()}
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            output_dir = args.output_dir / case.label
            output_dir.mkdir(parents=True, exist_ok=True)
            model, bank = _load_model(case, args.device)
            outputs, weights, center = _run_impulse(case, model, bank)
            _plot(case, outputs, weights, center, output_dir / "impulse_response.png")
            summary[case.label] = {
                "checkpoint": str(case.checkpoint),
                "source_sample_rate": case.source_rate,
                "metrics": _metrics(outputs, center, case.target_rate),
            }
        (args.output_dir / "impulse_metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "impulse_report.md").write_text(
            _render_markdown(summary), encoding="utf-8"
        )
    except OSError as error:
        raise RuntimeError(f"Failed to write impulse report: {error}") from error
    print(_render_markdown(summary))


if __name__ == "__main__":
    main()
