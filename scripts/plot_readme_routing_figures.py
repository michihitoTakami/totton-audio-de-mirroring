"""Render the CAPB overview figure and its per-source routing companion.

Produces two plots:

* ``capb_overview.png`` - the README figure: the three fixed prototypes and
  the blend the controller picks over one percussive recording.
* ``routing_by_source.png`` - the companion for ``docs/routing_examples.md``:
  mean blend weights per source, sustained frames versus transient frames.

The real recordings are evaluation-only material that is not stored in this
repository; pass the directory holding them with ``--audio-dir``.

Usage:
    uv run python scripts/plot_readme_routing_figures.py \
        --audio-dir <dir with the evaluation recordings> \
        --output-dir docs/images
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from matplotlib.axes import Axes

from totton_audio_de_mirroring.data.probe_generators import (
    generate_isolated_click,
    generate_square_wave,
)
from totton_audio_de_mirroring.evaluation.routing_analysis import transient_strength
from totton_audio_de_mirroring.models.capb import CAPB, capb_from_checkpoint

RELEASE_CHECKPOINTS: dict[int, Path] = {
    44_100: Path(
        "data/checkpoints/capb/run17_transient_dwell_20260922_44k1/capb_best.pt"
    ),
    48_000: Path(
        "data/checkpoints/capb_48k/run17_transient_dwell_20260922_48k/capb_best.pt"
    ),
}
REAL_SOURCES: tuple[tuple[str, str], ...] = (
    ("drum loop 110 BPM (44.1k)", "684120__sound_bar_kk__dry-str-pop-beat-110-bpm.wav"),
    (
        "hi-hat shuffle (44.1k)",
        "863823__d_yonqui__stereo-hi-hat-shuffle-loop-120-bpm.wav",
    ),
    ("kitchen foley (44.1k)", "864857__d_yonqui__kitchen-foley-drum-loop-140-bpm.wav"),
    ("electronic loop (48k)", "865041__namegirl__microtonic-138.flac"),
    (
        "water field recording (48k)",
        "868596__sieunk__water-flowing-inside-a-manhole.wav",
    ),
    ("ice cubes in a mug (48k)", "868595__sieunk__putting-ice-cubes-in-a-mug.wav"),
)
OVERVIEW_SOURCE = "drum loop 110 BPM (44.1k)"
EXCERPT_SEC = 8.0
OVERVIEW_SEC = 1.2
RESPONSE_FFT = 1 << 16
RESPONSE_MAX_HZ = 32_000.0
RESPONSE_FLOOR_DB = -130.0
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
FAINT = "#d8d7d2"
PROTOTYPE_COLORS: dict[str, str] = {
    "sharp": "#2a78d6",
    "mid": "#eb6834",
    "gentle": "#1baf7a",
}
LABEL_MIN_FRACTION = 0.08
RISK_QUANTILE = 0.95
SAFE_QUANTILE = 0.50
ACTIVE_FLOOR = 1.0e-3


@dataclass(frozen=True)
class SourceAnalysis:
    """Controller output for one source, ready to plot."""

    label: str
    sample_rate: int
    signal: np.ndarray
    names: tuple[str, ...]
    weights: np.ndarray
    sustained_mean: np.ndarray
    transient_mean: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse figure arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/images"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    """Render the overview figure and the per-source companion figure."""
    args = parse_args()
    if not args.audio_dir.is_dir():
        raise NotADirectoryError(f"Audio directory not found: {args.audio_dir}")
    models = _load_models(args.device)
    analyses = _synthetic_analyses(models) + _real_analyses(models, args.audio_dir)
    overview = [item for item in analyses if item.label == OVERVIEW_SOURCE]
    if not overview:
        raise ValueError(f"Overview source is missing: {OVERVIEW_SOURCE}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_overview(
        models[overview[0].sample_rate],
        overview[0],
        args.output_dir / "capb_overview.png",
    )
    _plot_by_source(analyses, args.output_dir / "routing_by_source.png")


def _load_models(device: str) -> dict[int, CAPB]:
    """Load the recommended checkpoint of each rate family."""
    models: dict[int, CAPB] = {}
    for rate, path in RELEASE_CHECKPOINTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"CAPB checkpoint not found: {path}")
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(f"Failed to load checkpoint {path}: {error}") from error
        models[rate] = capb_from_checkpoint(state).to(torch.device(device)).eval()
    return models


def _synthetic_analyses(models: dict[int, CAPB]) -> list[SourceAnalysis]:
    """Analyze the canonical probe shapes at the 44.1 kHz source rate.

    Args:
        models: Loaded checkpoints keyed by expected input sample rate.

    Returns:
        One analysis per probe shape.

    Physical Basis:
        Steady tones and noise carry no discontinuity, so image rejection is
        the only cost that matters; square edges and isolated clicks are the
        opposite case, where a long kernel exposes ringing around the edge.
    """
    rate = 44_100
    time_axis = np.arange(int(EXCERPT_SEC * rate), dtype=np.float64) / rate
    probes: tuple[tuple[str, np.ndarray], ...] = (
        ("sine 1 kHz", 0.5 * np.sin(2.0 * np.pi * 1_000.0 * time_axis)),
        ("sine 10 kHz", 0.5 * np.sin(2.0 * np.pi * 10_000.0 * time_axis)),
        ("pink noise", _pink_noise(time_axis.size)),
        (
            "square 500 Hz",
            generate_square_wave(500.0, sample_rate=rate, duration_sec=EXCERPT_SEC),
        ),
        ("isolated click", generate_isolated_click(sample_rate=rate, duration_sec=0.5)),
    )
    return [
        _analyze(label, np.asarray(signal, dtype=np.float32), rate, models)
        for label, signal in probes
    ]


def _real_analyses(models: dict[int, CAPB], audio_dir: Path) -> list[SourceAnalysis]:
    """Analyze the evaluation-only recordings found in ``audio_dir``."""
    analyses: list[SourceAnalysis] = []
    for label, filename in REAL_SOURCES:
        signal, rate = _load_excerpt(audio_dir / filename)
        analyses.append(_analyze(label, signal, rate, models))
    return analyses


def _load_excerpt(path: Path) -> tuple[np.ndarray, int]:
    """Read the leading mono excerpt of one evaluation recording."""
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation audio not found: {path}")
    try:
        audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Failed to read {path}: {error}") from error
    mono = np.asarray(np.mean(audio, axis=1), dtype=np.float32)
    mono = mono[: int(EXCERPT_SEC * sample_rate)]
    if mono.size == 0 or not np.all(np.isfinite(mono)):
        raise ValueError(f"Evaluation audio is empty or non-finite: {path}")
    return mono, int(sample_rate)


def _pink_noise(num_samples: int) -> np.ndarray:
    """Generate pink noise scaled to a fixed peak.

    Args:
        num_samples: Positive output length in samples.

    Returns:
        Pink-noise waveform scaled to a 0.3 peak.

    Raises:
        ValueError: If ``num_samples`` is not positive.

    Physical Basis:
        A 1/f spectrum keeps energy across the whole band without any
        envelope discontinuity, which is the stationary broadband case the
        sharp prototype is meant to cover.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    rng = np.random.default_rng(1234)
    spectrum = np.fft.rfft(rng.standard_normal(num_samples))
    frequency = np.fft.rfftfreq(num_samples, 1.0)
    scale = np.concatenate(([1.0], 1.0 / np.sqrt(frequency[1:])))
    noise = np.fft.irfft(spectrum * scale, n=num_samples)
    peak = float(np.max(np.abs(noise)))
    if peak <= 0.0:
        raise ValueError("Pink-noise generation produced a silent signal.")
    return np.asarray(0.3 * noise / peak, dtype=np.float64)


def _analyze(
    label: str, signal: np.ndarray, sample_rate: int, models: dict[int, CAPB]
) -> SourceAnalysis:
    """Run the controller and split its weights into sustained and transient frames.

    Args:
        label: Human-readable source name.
        signal: Mono source-rate waveform.
        sample_rate: Source sample rate in Hz.
        models: Loaded checkpoints keyed by expected input sample rate.

    Returns:
        Weight trajectory plus the two frame-subset means.

    Raises:
        ValueError: If no checkpoint matches the source sample rate.
    """
    if sample_rate not in models:
        raise ValueError(f"No checkpoint for {sample_rate} Hz source {label}.")
    model = models[sample_rate]
    device = next(model.parameters()).device
    source = torch.from_numpy(np.asarray(signal, dtype=np.float32)).to(device)
    with torch.no_grad():
        weights = model.controller_weights(source.unsqueeze(0)).squeeze(0).cpu().numpy()
    sustained, transient = _frame_subset_means(signal, weights)
    return SourceAnalysis(
        label=label,
        sample_rate=sample_rate,
        signal=np.asarray(signal, dtype=np.float64),
        names=tuple(model.prototype_names),
        weights=np.asarray(weights, dtype=np.float64),
        sustained_mean=sustained,
        transient_mean=transient,
    )


def _frame_subset_means(
    signal: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average the weights over sustained and strongest-transient frames.

    Args:
        signal: Mono source-rate waveform.
        weights: Convex weights shaped (prototype, frame).

    Returns:
        Mean weights over the quietest-motion half and the top five percent
        of transient strength, both restricted to non-silent frames.

    Physical Basis:
        Silence has no acoustic consequence, and ringing is exposed by
        envelope changes rather than by sustain, so the two subsets separate
        the image-rejection case from the ringing case.
    """
    frame_count = int(weights.shape[1])
    level = _frame_levels(signal, frame_count)
    active = level > max(float(np.max(level)) * ACTIVE_FLOOR, 1.0e-8)
    if not np.any(active):
        raise ValueError("Signal has no active frames.")
    strength = transient_strength(signal, frame_count)
    active_strength = strength[active]
    risk = float(np.quantile(active_strength, RISK_QUANTILE))
    safe = float(np.quantile(active_strength, SAFE_QUANTILE))
    return (
        _masked_mean(weights, active & (strength <= safe)),
        _masked_mean(weights, active & (strength >= risk)),
    )


def _frame_levels(signal: np.ndarray, frame_count: int) -> np.ndarray:
    """Measure mean absolute level per controller frame."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}.")
    edges = np.linspace(0, signal.size, frame_count + 1).astype(int)
    magnitude = np.abs(np.asarray(signal, dtype=np.float64))
    return np.asarray(
        [
            float(np.mean(magnitude[start:stop])) if stop > start else 0.0
            for start, stop in zip(edges[:-1], edges[1:], strict=True)
        ]
    )


def _masked_mean(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average weights over the selected frames, falling back to all frames."""
    selected = weights[:, mask] if np.any(mask) else weights
    return np.asarray(np.mean(selected, axis=1), dtype=np.float64)


def _prototype_responses(
    model: CAPB, output_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute passband-normalized prototype magnitude responses.

    Args:
        model: Loaded CAPB model holding the fixed kernel bank.
        output_rate: Output sample rate of the interpolation stage in Hz.

    Returns:
        Frequency axis in Hz and magnitude in dB shaped (prototype, bin).

    Physical Basis:
        The kernels carry the interpolation gain, so normalizing each one at
        DC shows the shape that matters here: how far each prototype pushes
        the image band down, and how much passband it gives up for it.
    """
    if output_rate <= 0:
        raise ValueError(f"output_rate must be positive, got {output_rate}.")
    kernels = np.asarray(torch.as_tensor(model.kernels).detach().cpu().numpy())[:, 0, :]
    spectrum = np.fft.rfft(kernels, n=RESPONSE_FFT, axis=1)
    magnitude = np.abs(spectrum) / np.abs(spectrum[:, :1])
    frequency = np.fft.rfftfreq(RESPONSE_FFT, 1.0 / output_rate)
    keep = frequency <= RESPONSE_MAX_HZ
    decibels = 20.0 * np.log10(np.maximum(magnitude[:, keep], 1.0e-12))
    return frequency[keep], decibels


def _plot_overview(model: CAPB, item: SourceAnalysis, output_path: Path) -> None:
    """Draw the README figure: fixed prototypes plus one blend trajectory."""
    figure = plt.figure(figsize=(13.0, 4.8), layout="constrained")
    figure.patch.set_facecolor(SURFACE)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.0, 1.3), height_ratios=(1.0, 1.3))
    response_axis = figure.add_subplot(grid[:, 0])
    wave_axis = figure.add_subplot(grid[0, 1])
    weight_axis = figure.add_subplot(grid[1, 1], sharex=wave_axis)
    _draw_responses(response_axis, model, item.names)
    _draw_timeline(wave_axis, weight_axis, item)
    figure.suptitle(
        "CAPB: three fixed FIRs, and the blend the controller moves",
        color=INK,
        fontsize=14,
    )
    _save_figure(figure, output_path)


def _draw_responses(axis: Axes, model: CAPB, names: tuple[str, ...]) -> None:
    """Draw the fixed prototype magnitude responses of one rate family."""
    ratio = int(model.upsample_ratio)
    output_rate = ratio * 44_100
    frequency, decibels = _prototype_responses(model, output_rate)
    nyquist = 44_100 / 2.0
    axis.axvspan(
        nyquist / 1_000.0,
        RESPONSE_MAX_HZ / 1_000.0,
        color=FAINT,
        alpha=0.6,
        linewidth=0,
    )
    handles = [
        axis.plot(
            frequency / 1_000.0,
            decibels[index],
            color=PROTOTYPE_COLORS[name],
            linewidth=2.0,
        )[0]
        for index, name in enumerate(names)
    ]
    axis.set_xlim(0.0, RESPONSE_MAX_HZ / 1_000.0)
    axis.set_ylim(RESPONSE_FLOOR_DB, 8.0)
    axis.set_xlabel("frequency (kHz)", color=MUTED, fontsize=10)
    axis.set_ylabel("magnitude (dB)", color=MUTED, fontsize=10)
    axis.set_title(
        "Fixed prototypes, never learned (44.1k family)",
        color=INK,
        fontsize=11,
        pad=10,
    )
    axis.text(
        (nyquist + RESPONSE_MAX_HZ) / 2_000.0,
        -20.0,
        "image band",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    axis.grid(color=FAINT, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(
        handles,
        [f"{name} FIR" for name in names],
        loc="lower left",
        frameon=False,
        fontsize=11,
        labelcolor=INK,
    )
    _strip_chrome(axis)


def _draw_timeline(wave_axis: Axes, weight_axis: Axes, item: SourceAnalysis) -> None:
    """Draw the waveform and the blend weights it produces."""
    samples = min(item.signal.size, int(OVERVIEW_SEC * item.sample_rate))
    wave_axis.plot(
        np.arange(samples) / item.sample_rate,
        item.signal[:samples],
        color=MUTED,
        linewidth=0.4,
    )
    wave_axis.set_title(f"Input: {item.label}", color=INK, fontsize=11, pad=10)
    wave_axis.set_ylabel("amplitude", color=MUTED, fontsize=10)
    wave_axis.tick_params(labelbottom=False)
    frame_time = (
        np.arange(item.weights.shape[1])
        * item.signal.size
        / item.weights.shape[1]
        / item.sample_rate
    )
    weight_axis.stackplot(
        frame_time,
        item.weights,
        colors=[PROTOTYPE_COLORS[name] for name in item.names],
    )
    weight_axis.set_ylim(0.0, 1.0)
    weight_axis.set_xlim(0.0, samples / item.sample_rate)
    weight_axis.set_ylabel("blend weight", color=MUTED, fontsize=10)
    weight_axis.set_xlabel("time (s)", color=MUTED, fontsize=10)
    _annotate_hit(weight_axis, item)
    for axis in (wave_axis, weight_axis):
        _strip_chrome(axis)


def _annotate_hit(axis: Axes, item: SourceAnalysis) -> None:
    """Point at the loudest hit of the window, where the blend leaves sharp."""
    samples = min(item.signal.size, int(OVERVIEW_SEC * item.sample_rate))
    event_sec = float(np.argmax(np.abs(item.signal[:samples]))) / item.sample_rate
    axis.annotate(
        "gentle and mid appear\nonly at hits",
        xy=(event_sec, 0.45),
        xytext=(event_sec + 0.10, 0.62),
        color="white",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "white", "linewidth": 1.2},
    )


def _plot_by_source(analyses: list[SourceAnalysis], output_path: Path) -> None:
    """Draw the stacked per-source weight bars for both frame subsets."""
    names = analyses[0].names
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), sharey=True)
    figure.patch.set_facecolor(SURFACE)
    positions = np.arange(len(analyses))[::-1]
    panels = (
        ("Sustained frames", [item.sustained_mean for item in analyses]),
        ("Transient frames (loudest 5%)", [item.transient_mean for item in analyses]),
    )
    for axis, (title, values) in zip(axes, panels, strict=True):
        _stacked_bars(axis, positions, np.asarray(values), names)
        axis.set_title(title, color=INK, fontsize=12, pad=12)
        axis.set_xlabel("blend weight", color=MUTED, fontsize=10)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([item.label for item in analyses], fontsize=10, color=INK)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PROTOTYPE_COLORS[name]) for name in names
    ]
    figure.legend(
        handles,
        list(names),
        loc="lower center",
        ncols=len(names),
        frameon=False,
        fontsize=11,
    )
    figure.suptitle(
        "CAPB run17 - which prototype the controller blends, by source",
        color=INK,
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    _save_figure(figure, output_path)


def _stacked_bars(
    axis: Axes, positions: np.ndarray, values: np.ndarray, names: tuple[str, ...]
) -> None:
    """Draw one horizontal stacked bar per source with direct segment labels."""
    offsets = np.zeros(values.shape[0])
    for index, name in enumerate(names):
        widths = values[:, index]
        axis.barh(
            positions,
            widths,
            left=offsets,
            height=0.62,
            color=PROTOTYPE_COLORS[name],
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        for position, width, offset in zip(positions, widths, offsets, strict=True):
            if width < LABEL_MIN_FRACTION:
                continue
            axis.text(
                offset + width / 2.0,
                position,
                f"{width:.0%}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
        offsets = offsets + widths
    axis.set_xlim(0.0, 1.0)
    _strip_chrome(axis)


def _strip_chrome(axis: Axes) -> None:
    """Apply the shared recessive axis styling."""
    axis.set_facecolor(SURFACE)
    axis.tick_params(colors=MUTED, labelsize=9, length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Write one figure and always release its resources."""
    try:
        figure.savefig(output_path, dpi=150, facecolor=SURFACE)
    except OSError as error:
        raise RuntimeError(f"Failed to write plot {output_path}: {error}") from error
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
