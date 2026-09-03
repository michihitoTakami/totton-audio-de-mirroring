"""Plot CAPB prototype routing on evaluation-only real recordings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from totton_audio_de_mirroring.evaluation.routing_analysis import (
    summarize_routing,
    transient_strength,
)
from totton_audio_de_mirroring.models.capb import CAPB, capb_from_checkpoint

_RELEASE_CHECKPOINTS = (
    Path("data/checkpoints/capb/run16_v5b_midflat_g03_20260903_44k1/capb_best.pt"),
    Path("data/checkpoints/capb_48k/run16_v5b_midflat_g03_20260903_48k/capb_best.pt"),
)


@dataclass(frozen=True)
class LoadedModel:
    """A labelled, rate-validated CAPB checkpoint."""

    label: str
    expected_input_rate: int
    model: CAPB


def parse_args() -> argparse.Namespace:
    """Parse report arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, action="append", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeat for release and candidate checkpoints; defaults to release pair.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    """Generate whole-file and strongest-transient routing plots."""
    args = parse_args()
    checkpoints = _parse_checkpoint_args(args.checkpoint)
    models = [_load_model(label, path, args.device) for label, path in checkpoints]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"files": []}
    for audio_path in args.audio:
        signal, sample_rate = _load_audio(audio_path)
        matches = [item for item in models if item.expected_input_rate == sample_rate]
        if not matches:
            raise ValueError(
                f"No {sample_rate} Hz checkpoint supplied for {audio_path}."
            )
        analyses = [_analyze_model(item, signal) for item in matches]
        stem = audio_path.stem
        _plot_routing(signal, sample_rate, analyses, args.output_dir / f"{stem}.png")
        _plot_transient_zoom(
            signal,
            sample_rate,
            analyses,
            args.output_dir / f"{stem}_strongest_transient.png",
        )
        report["files"].append(
            {
                "audio": str(audio_path),
                "sample_rate": sample_rate,
                "duration_sec": signal.size / sample_rate,
                "models": [analysis["metrics"] for analysis in analyses],
            }
        )
    try:
        (args.output_dir / "routing_summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
    except OSError as error:
        raise RuntimeError(f"Failed to write routing report: {error}") from error


def _parse_checkpoint_args(values: list[str]) -> list[tuple[str, Path]]:
    if not values:
        return [("release", path) for path in _RELEASE_CHECKPOINTS]
    parsed: list[tuple[str, Path]] = []
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("--checkpoint must use LABEL=PATH syntax.")
        parsed.append((label, Path(raw_path)))
    return parsed


def _load_model(label: str, path: Path, device: str) -> LoadedModel:
    if not path.is_file():
        raise FileNotFoundError(f"CAPB checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Failed to load checkpoint {path}: {error}") from error
    model = capb_from_checkpoint(state).to(torch.device(device)).eval()
    target_rate = int(state.get("target_sample_rate", 88_200))
    expected_rate = int(state.get("expected_input_rate", target_rate // 2))
    return LoadedModel(label=label, expected_input_rate=expected_rate, model=model)


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation audio not found: {path}")
    try:
        audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"Failed to read evaluation audio {path}: {error}"
        ) from error
    mono = np.asarray(np.mean(audio, axis=1), dtype=np.float32)
    if mono.size == 0 or not np.all(np.isfinite(mono)):
        raise ValueError(f"Evaluation audio is empty or non-finite: {path}")
    return mono, int(sample_rate)


def _analyze_model(item: LoadedModel, signal: np.ndarray) -> dict[str, Any]:
    device = next(item.model.parameters()).device
    source = torch.from_numpy(np.array(signal, dtype=np.float32, copy=True)).to(device)
    with torch.no_grad():
        weights = item.model.controller_weights(source.unsqueeze(0))
        weights = weights.squeeze(0).cpu().numpy()
    sharp_index = item.model.prototype_names.index("sharp")
    gentle_index = item.model.prototype_names.index("gentle")
    middle_index = (
        item.model.prototype_names.index("mid")
        if "mid" in item.model.prototype_names
        else None
    )
    summary = summarize_routing(
        signal, weights, sharp_index, gentle_index, middle_index
    )
    return {
        "label": item.label,
        "names": item.model.prototype_names,
        "weights": weights,
        "strength": transient_strength(signal, weights.shape[1]),
        "metrics": {
            "label": item.label,
            "prototype_profile": item.model.prototype_profile,
            "prototype_names": list(item.model.prototype_names),
            **summary.to_dict(),
        },
    }


def _plot_routing(
    signal: np.ndarray,
    sample_rate: int,
    analyses: list[dict[str, Any]],
    output_path: Path,
    xlim: tuple[float, float] | None = None,
) -> None:
    rows = 1 + len(analyses)
    figure, axes = plt.subplots(rows, 1, figsize=(16, 2.8 * rows), sharex=True)
    axes = np.atleast_1d(axes)
    signal_time = np.arange(signal.size) / sample_rate
    axes[0].plot(signal_time, signal, color="black", linewidth=0.35)
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("Evaluation-only source waveform")
    for axis, analysis in zip(axes[1:], analyses, strict=True):
        frame_time = np.linspace(
            0.0, signal.size / sample_rate, analysis["weights"].shape[1]
        )
        axis.fill_between(
            frame_time,
            0.0,
            analysis["strength"],
            color="0.8",
            label="transient strength",
        )
        for index, name in enumerate(analysis["names"]):
            axis.plot(frame_time, analysis["weights"][index], label=name, linewidth=0.8)
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel(analysis["label"])
        axis.legend(loc="upper right", ncols=len(analysis["names"]) + 1)
    axes[-1].set_xlabel("time (s)")
    if xlim is not None:
        axes[-1].set_xlim(*xlim)
    figure.tight_layout()
    _save_figure(figure, output_path)


def _plot_transient_zoom(
    signal: np.ndarray,
    sample_rate: int,
    analyses: list[dict[str, Any]],
    output_path: Path,
) -> None:
    reference = analyses[0]
    event_frame = int(np.argmax(reference["strength"]))
    event_sec = event_frame * signal.size / reference["weights"].shape[1] / sample_rate
    low_sec = max(0.0, event_sec - 0.1)
    high_sec = min(signal.size / sample_rate, event_sec + 0.1)
    _plot_routing(
        signal,
        sample_rate,
        analyses,
        output_path,
        xlim=(low_sec, high_sec),
    )


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    try:
        figure.savefig(output_path, dpi=160)
    except OSError as error:
        raise RuntimeError(f"Failed to write plot {output_path}: {error}") from error
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
