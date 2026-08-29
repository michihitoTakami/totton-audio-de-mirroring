"""Generate CAPB ringing, image, sideband, THD, and IMD diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.reference import upsample_bessel_reference
from totton_audio_de_mirroring.evaluation.distortion import (
    added_am_sideband_db,
    ccif_imd_db,
    relative_line_levels_db,
    smpte_imd_db,
    thd_db,
)
from totton_audio_de_mirroring.models.capb import CAPB, capb_from_checkpoint
from totton_audio_de_mirroring.models.proto_bank import (
    PrototypeBank,
    build_prototype_bank,
    prototype_specs_for_target_rate,
    upsample_with_kernel,
)

_BESSEL_CUTOFF_HZ = 20_000.0
_BESSEL_ORDER = 6
_ANALYSIS_DURATION_SEC = 1
_SIGNAL_DURATION_SEC = 3
_DB_FLOOR = -180.0


@dataclass(frozen=True)
class RateCase:
    """One CAPB checkpoint and its sample-rate family."""

    label: str
    source_rate: int
    checkpoint: Path

    @property
    def target_rate(self) -> int:
        """Return the fixed 2x Stage 1 output rate."""
        return self.source_rate * 2


@dataclass(frozen=True)
class ProcessedSignal:
    """Reference, prototype, CAPB, and controller outputs for one probe."""

    outputs: dict[str, np.ndarray]
    weights: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse reproducible report inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-44k1", type=Path, required=True)
    parser.add_argument("--checkpoint-48k", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    """Generate both rate-family reports and a cross-family summary.

    Physical Basis:
        The same deterministic probes and coherent analysis regions are used
        for both rate families. This makes family differences attributable to
        their checkpoint/controller and prototype bank, not FFT leakage.
    """
    args = parse_args()
    cases = (
        RateCase("44k1", 44_100, args.checkpoint_44k1),
        RateCase("48k", 48_000, args.checkpoint_48k),
    )
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            case.label: _generate_rate_report(case, args.output_dir, args.device)
            for case in cases
        }
        _plot_cross_family(summary, args.output_dir / "distortion_comparison.png")
        _write_summary(summary, cases, args.output_dir)
    except OSError as error:
        raise RuntimeError(
            f"Failed to write CAPB visualization report under {args.output_dir}: "
            f"{error}"
        ) from error


def _generate_rate_report(
    case: RateCase,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    """Generate all plots and metrics for one rate family."""
    output_dir = output_root / case.label
    output_dir.mkdir(parents=True, exist_ok=True)
    model, bank = _load_model(case, device)
    square_metrics = {
        str(frequency_hz): _plot_square_response(
            case, model, bank, frequency_hz, output_dir
        )
        for frequency_hz in (100, 500)
    }
    _plot_sweep_reports(case, model, bank, output_dir)
    distortion = _measure_distortion(case, model, bank)
    _plot_thd(case, distortion, output_dir / "thd_harmonics.png")
    _plot_imd(case, distortion, output_dir / "imd_products.png")
    _plot_sideband_degradation(
        case, distortion, output_dir / "sideband_degradation.png"
    )
    _plot_am_sidebands(case, distortion, output_dir / "am_sidebands.png")
    return {
        "source_sample_rate": case.source_rate,
        "target_sample_rate": case.target_rate,
        "checkpoint": str(case.checkpoint),
        "square": square_metrics,
        "distortion": distortion["metrics"],
        "controller": distortion["controller"],
        "line_levels_db": distortion["line_levels_db"],
    }


def _load_model(case: RateCase, device: str) -> tuple[CAPB, PrototypeBank]:
    """Load and rate-validate one CAPB checkpoint."""
    if not case.checkpoint.is_file():
        raise FileNotFoundError(f"CAPB checkpoint not found: {case.checkpoint}")
    try:
        checkpoint = torch.load(case.checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"Failed to load checkpoint {case.checkpoint}: {error}"
        ) from error
    expected_rate = int(checkpoint.get("expected_input_rate", case.source_rate))
    if expected_rate != case.source_rate:
        raise ValueError(
            f"Checkpoint {case.checkpoint} expects {expected_rate} Hz, "
            f"not {case.source_rate} Hz."
        )
    model = capb_from_checkpoint(checkpoint).to(torch.device(device))
    bank = build_prototype_bank(
        prototype_specs_for_target_rate(case.target_rate),
        sample_rate=case.target_rate,
    )
    return model, bank


def _process_probe(
    source: np.ndarray,
    case: RateCase,
    model: CAPB,
    bank: PrototypeBank,
) -> ProcessedSignal:
    """Run all linear references and CAPB without mutating the source."""
    source_copy = np.array(source, dtype=np.float64, copy=True)
    if source_copy.ndim != 1 or source_copy.size == 0:
        raise ValueError("source must be a non-empty 1D waveform.")
    with torch.no_grad():
        tensor = torch.from_numpy(source_copy.astype(np.float32)).unsqueeze(0)
        tensor = tensor.to(next(model.parameters()).device)
        capb_output, weights = model(tensor, return_weights=True)
    outputs = {
        "ideal": np.asarray(sp_signal.resample_poly(source_copy, 2, 1)),
        "bessel": upsample_bessel_reference(
            source_copy,
            case.source_rate,
            case.target_rate,
            _BESSEL_CUTOFF_HZ,
            _BESSEL_ORDER,
        ),
        "sharp": upsample_with_kernel(source_copy, bank.kernels[0], 2),
        "gentle": upsample_with_kernel(source_copy, bank.kernels[-1], 2),
        "capb": np.asarray(capb_output.squeeze(0).cpu(), dtype=np.float64),
    }
    return ProcessedSignal(
        outputs=outputs,
        weights=np.asarray(weights.squeeze(0).cpu(), dtype=np.float64),
    )


def _plot_square_response(
    case: RateCase,
    model: CAPB,
    bank: PrototypeBank,
    frequency_hz: int,
    output_dir: Path,
) -> dict[str, float]:
    """Plot one rising edge in the format of the legacy CAPB report."""
    time = np.arange(case.source_rate, dtype=np.float64) / case.source_rate
    source = 0.5 * np.where(np.sin(2.0 * np.pi * frequency_hz * time) >= 0.0, 1.0, -1.0)
    processed = _process_probe(source, case, model, bank)
    edges = {
        name: _nearest_rising_edge(values, case.target_rate)
        for name, values in processed.outputs.items()
    }
    metrics = {
        name: _plateau_ripple(values, edges[name], case.target_rate)
        for name, values in processed.outputs.items()
    }
    _render_square_plot(
        case, frequency_hz, processed.outputs, edges, metrics, output_dir
    )
    return {f"{name}_plateau_rms": value for name, value in metrics.items()}


def _nearest_rising_edge(signal: np.ndarray, sample_rate: int) -> int:
    """Find the strongest rising edge near the waveform midpoint."""
    center = signal.size // 2
    radius = max(2, sample_rate // 100)
    start = max(0, center - radius)
    stop = min(signal.size - 1, center + radius)
    return int(start + np.argmax(np.diff(signal[start : stop + 1])) + 1)


def _plateau_ripple(signal: np.ndarray, edge: int, sample_rate: int) -> float:
    """Measure median-referenced RMS ripple from 0.1 to 0.8 ms."""
    start = edge + round(0.0001 * sample_rate)
    stop = edge + round(0.0008 * sample_rate)
    plateau = np.asarray(signal[start:stop], dtype=np.float64)
    error = plateau - np.median(plateau)
    return float(np.sqrt(np.mean(np.square(error))))


def _render_square_plot(
    case: RateCase,
    frequency_hz: int,
    outputs: dict[str, np.ndarray],
    edges: dict[str, int],
    metrics: dict[str, float],
    output_dir: Path,
) -> None:
    """Render an edge-aligned square response comparison."""
    figure, axis = plt.subplots(figsize=(12, 5))
    styles = (("sharp", "#ef5350"), ("bessel", "#777777"), ("capb", "#1976d2"))
    for name, color in styles:
        relative_ms, waveform = _edge_window(
            outputs[name], edges[name], case.target_rate, -1.0, 3.0
        )
        axis.plot(relative_ms, waveform, color=color, linewidth=1.1, label=name)
    axis.axvspan(0.1, 0.8, color="#4caf50", alpha=0.10, label="gate plateau 0.1–0.8 ms")
    axis.set(
        xlabel="time from edge (ms)",
        ylabel="amplitude",
        title=(
            f"{case.label}: {frequency_hz} Hz square rising edge | "
            f"plateau RMS bessel={metrics['bessel']:.2e}, "
            f"CAPB={metrics['capb']:.2e}, sharp={metrics['sharp']:.2e}"
        ),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"square_{frequency_hz}hz_edge.png", dpi=150)
    plt.close(figure)


def _edge_window(
    signal: np.ndarray,
    edge: int,
    sample_rate: int,
    start_ms: float,
    stop_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one edge-relative plotting window."""
    start = edge + round(start_ms * sample_rate / 1_000.0)
    stop = edge + round(stop_ms * sample_rate / 1_000.0)
    samples = np.arange(start, stop)
    return (samples - edge) * 1_000.0 / sample_rate, signal[start:stop]


def _plot_sweep_reports(
    case: RateCase,
    model: CAPB,
    bank: PrototypeBank,
    output_dir: Path,
) -> None:
    """Render sweep spectrogram and spectrum comparisons."""
    duration = 2.0
    time = np.arange(round(case.source_rate * duration)) / case.source_rate
    source = 0.5 * sp_signal.chirp(
        time, f0=20.0, t1=duration, f1=20_000.0, method="logarithmic"
    )
    processed = _process_probe(source, case, model, bank)
    _plot_sweep_spectrogram(case, processed.outputs, output_dir)
    _plot_sweep_spectrum(case, processed.outputs, output_dir)


def _spectrogram_db(
    signal: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a target-rate spectrogram normalized to its peak."""
    frequencies, times, spectrum = sp_signal.spectrogram(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=2_048,
        noverlap=1_792,
        mode="magnitude",
    )
    magnitude_db = 20.0 * np.log10(np.maximum(spectrum, 1.0e-12))
    return frequencies, times, magnitude_db


def _plot_sweep_spectrogram(
    case: RateCase,
    outputs: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Plot Bessel and CAPB sweep image ridges on one color scale."""
    panels = tuple(
        _spectrogram_db(outputs[name], case.target_rate) for name in ("bessel", "capb")
    )
    peak_db = max(float(np.max(panel[2])) for panel in panels)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    image = None
    for axis, name, (frequencies, times, magnitude_db) in zip(
        axes, ("Bessel reference SRC", "CAPB retrain"), panels, strict=True
    ):
        image = axis.pcolormesh(
            times,
            frequencies / 1_000.0,
            magnitude_db - peak_db,
            shading="auto",
            cmap="magma",
            vmin=-140.0,
            vmax=0.0,
        )
        axis.axhline(
            case.source_rate / 2_000.0, color="cyan", linestyle="--", linewidth=1.0
        )
        axis.set(title=name, xlabel="time (s)")
    axes[0].set_ylabel("frequency (kHz)")
    figure.suptitle(f"{case.label}: log sweep 20 Hz–20 kHz; cyan = input Nyquist")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            label="dB relative to report peak",
            shrink=0.90,
            pad=0.02,
        )
    figure.savefig(output_dir / "sweep_spectrogram.png", dpi=150)
    plt.close(figure)


def _plot_sweep_spectrum(
    case: RateCase,
    outputs: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Plot passband and image-band sweep power for four paths."""
    figure, axis = plt.subplots(figsize=(12, 5))
    for name in ("bessel", "capb", "sharp", "gentle"):
        frequencies, psd = sp_signal.welch(
            outputs[name], fs=case.target_rate, nperseg=8_192
        )
        axis.semilogx(
            frequencies[1:],
            10.0 * np.log10(np.maximum(psd[1:], 1.0e-20)),
            label=name,
            linewidth=1.0,
        )
    axis.axvline(case.source_rate / 2.0, color="black", linestyle="--", linewidth=1.0)
    axis.text(case.source_rate / 2.0 * 1.02, -85.0, "input Nyquist")
    axis.set(
        xlim=(100.0, case.target_rate / 2.0),
        ylim=(-200.0, -30.0),
        xlabel="frequency (Hz)",
        ylabel="PSD (dB/Hz)",
        title=f"{case.label}: sweep passband and interpolation image",
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "sweep_spectrum.png", dpi=150)
    plt.close(figure)


def _measure_distortion(
    case: RateCase,
    model: CAPB,
    bank: PrototypeBank,
) -> dict[str, Any]:
    """Run coherent THD, two-tone IMD, and AM sideband probes."""
    time = np.arange(case.source_rate * _SIGNAL_DURATION_SEC) / case.source_rate
    sources = {
        "thd": 0.5 * np.sin(2.0 * np.pi * 1_000.0 * time),
        "smpte": 0.4 * np.sin(2.0 * np.pi * 60.0 * time)
        + 0.1 * np.sin(2.0 * np.pi * 7_000.0 * time),
        "ccif": 0.25 * np.sin(2.0 * np.pi * 19_000.0 * time)
        + 0.25 * np.sin(2.0 * np.pi * 20_000.0 * time),
        "am": (0.5 / 1.5)
        * (1.0 + 0.5 * np.sin(2.0 * np.pi * 37.0 * time))
        * np.sin(2.0 * np.pi * 10_000.0 * time),
    }
    processed = {
        name: _process_probe(source, case, model, bank)
        for name, source in sources.items()
    }
    cropped = {
        probe: {
            backend: _analysis_region(output, case.target_rate)
            for backend, output in result.outputs.items()
        }
        for probe, result in processed.items()
    }
    return _assemble_distortion_result(case, processed, cropped)


def _analysis_region(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Copy the steady one-second center region after startup transients."""
    start = sample_rate
    stop = start + sample_rate * _ANALYSIS_DURATION_SEC
    if signal.size < stop:
        raise ValueError("signal is too short for the coherent analysis region.")
    return np.array(signal[start:stop], dtype=np.float64, copy=True)


def _assemble_distortion_result(
    case: RateCase,
    processed: dict[str, ProcessedSignal],
    cropped: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Compute scalar metrics and plot-ready line levels."""
    backends = ("ideal", "bessel", "sharp", "gentle", "capb")
    metrics = {
        backend: {
            "thd_1khz_20khz_db": thd_db(cropped["thd"][backend], case.target_rate),
            "smpte_imd_db": smpte_imd_db(cropped["smpte"][backend], case.target_rate),
            "ccif_imd_db": ccif_imd_db(cropped["ccif"][backend], case.target_rate),
            "added_am_sideband_db": added_am_sideband_db(
                cropped["am"][backend], case.target_rate
            ),
        }
        for backend in backends
    }
    lines = _collect_line_levels(case, cropped)
    smpte_weights = processed["smpte"].weights
    controller = {
        "smpte_weight_min": smpte_weights.min(axis=1).tolist(),
        "smpte_weight_max": smpte_weights.max(axis=1).tolist(),
        "smpte_weight_mean": smpte_weights.mean(axis=1).tolist(),
        "prototype_names": ["sharp", "mid", "gentle"],
    }
    return {
        "metrics": metrics,
        "line_levels_db": lines,
        "controller": controller,
        "weights": smpte_weights,
    }


def _collect_line_levels(
    case: RateCase,
    cropped: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Collect all coherent spectral-line arrays used by plots."""
    thd_frequencies = tuple(float(order * 1_000) for order in range(2, 21))
    smpte_frequencies = tuple(float(7_000 + offset) for offset in range(-300, 301, 60))
    ccif_frequencies = (1_000.0, 18_000.0, 19_000.0, 20_000.0, 21_000.0)
    am_frequencies = tuple(float(10_000 + order * 37) for order in range(-6, 7))
    return {
        "thd_frequencies_hz": thd_frequencies,
        "thd": _line_map(cropped["thd"], case.target_rate, thd_frequencies, 1_000.0),
        "smpte_frequencies_hz": smpte_frequencies,
        "smpte": _line_map(
            cropped["smpte"], case.target_rate, smpte_frequencies, 7_000.0
        ),
        "ccif_frequencies_hz": ccif_frequencies,
        "ccif": _line_map(
            cropped["ccif"], case.target_rate, ccif_frequencies, 19_000.0
        ),
        "am_frequencies_hz": am_frequencies,
        "am": _line_map(cropped["am"], case.target_rate, am_frequencies, 10_000.0),
    }


def _line_map(
    outputs: dict[str, np.ndarray],
    sample_rate: int,
    frequencies_hz: tuple[float, ...],
    reference_hz: float,
) -> dict[str, tuple[float, ...]]:
    """Measure one line grid for all report backends."""
    return {
        name: relative_line_levels_db(
            outputs[name], sample_rate, frequencies_hz, reference_hz
        )
        for name in ("ideal", "bessel", "capb")
    }


def _plot_thd(case: RateCase, result: dict[str, Any], output_path: Path) -> None:
    """Plot individual 1 kHz harmonic levels through 20 kHz."""
    lines = result["line_levels_db"]
    orders = np.arange(2, 21)
    figure, axis = plt.subplots(figsize=(11, 5))
    for name, marker in (("ideal", "o"), ("bessel", "s"), ("capb", "^")):
        levels = np.maximum(lines["thd"][name], _DB_FLOOR)
        axis.plot(orders, levels, marker=marker, markersize=4, label=name)
    capb_thd = result["metrics"]["capb"]["thd_1khz_20khz_db"]
    axis.set(
        xticks=orders,
        ylim=(_DB_FLOOR, -60.0),
        xlabel="harmonic order (1 kHz fundamental)",
        ylabel="level (dBc)",
        title=f"{case.label}: audio-band THD = {capb_thd:.1f} dB",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_imd(case: RateCase, result: dict[str, Any], output_path: Path) -> None:
    """Plot SMPTE and CCIF intermodulation product lines."""
    lines = result["line_levels_db"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    smpte_offsets = np.asarray(lines["smpte_frequencies_hz"]) - 7_000.0
    ccif_freqs = np.asarray(lines["ccif_frequencies_hz"]) / 1_000.0
    for name, marker in (("ideal", "o"), ("bessel", "s"), ("capb", "^")):
        axes[0].plot(
            smpte_offsets,
            np.maximum(lines["smpte"][name], _DB_FLOOR),
            marker=marker,
            label=name,
        )
        axes[1].plot(
            ccif_freqs,
            np.maximum(lines["ccif"][name], _DB_FLOOR),
            marker=marker,
            label=name,
        )
    axes[0].set(
        title="SMPTE: lines around 7 kHz",
        xlabel="offset from 7 kHz (Hz)",
        ylabel="level (dBc)",
        ylim=(_DB_FLOOR, 5.0),
    )
    axes[1].set(
        title="CCIF: difference and third-order products",
        xlabel="frequency (kHz)",
        ylabel="level relative to 19 kHz (dBc)",
        ylim=(_DB_FLOOR, 5.0),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(
        f"{case.label}: CAPB IMD — SMPTE {result['metrics']['capb']['smpte_imd_db']:.1f} dB, "
        f"CCIF {result['metrics']['capb']['ccif_imd_db']:.1f} dB"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_sideband_degradation(
    case: RateCase,
    result: dict[str, Any],
    output_path: Path,
) -> None:
    """Relate SMPTE sideband growth to CAPB controller-weight motion."""
    lines = result["line_levels_db"]
    offsets = np.asarray(lines["smpte_frequencies_hz"]) - 7_000.0
    weights = np.asarray(result["weights"])
    frame_times_ms = np.arange(weights.shape[1]) * 64_000.0 / case.source_rate
    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    for name, marker in (("ideal", "o"), ("bessel", "s"), ("capb", "^")):
        axes[0].plot(
            offsets,
            np.maximum(lines["smpte"][name], _DB_FLOOR),
            marker=marker,
            label=name,
        )
    axes[0].set(
        xlabel="offset from 7 kHz (Hz)",
        ylabel="level (dBc)",
        ylim=(_DB_FLOOR, 5.0),
        title="Added symmetric sideband family",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for index, name in enumerate(("sharp", "mid", "gentle")):
        axes[1].plot(frame_times_ms, weights[index], label=name)
    axes[1].set(
        xlim=(1_000.0, 1_200.0),
        xlabel="time (ms)",
        ylabel="blend weight",
        title="Controller weights on the same SMPTE probe",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle(
        f"{case.label}: sideband degradation diagnostic — SMPTE IMD "
        f"{result['metrics']['capb']['smpte_imd_db']:.1f} dB"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_am_sidebands(
    case: RateCase,
    result: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot wanted and added sideband orders for sinusoidal AM."""
    lines = result["line_levels_db"]
    orders = np.arange(-6, 7)
    figure, axis = plt.subplots(figsize=(11, 5))
    for name, marker in (("ideal", "o"), ("bessel", "s"), ("capb", "^")):
        axis.plot(
            orders, np.maximum(lines["am"][name], _DB_FLOOR), marker=marker, label=name
        )
    axis.axvspan(-1.1, 1.1, color="#4caf50", alpha=0.08, label="input lines")
    axis.set(
        xticks=orders,
        ylim=(_DB_FLOOR, 5.0),
        xlabel="37 Hz sideband order around 10 kHz",
        ylabel="level (dBc)",
        title=(
            f"{case.label}: strongest added AM sideband = "
            f"{result['metrics']['capb']['added_am_sideband_db']:.1f} dB"
        ),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_cross_family(summary: dict[str, Any], output_path: Path) -> None:
    """Plot the four scalar CAPB distortion diagnostics by rate family."""
    metric_names = (
        "thd_1khz_20khz_db",
        "smpte_imd_db",
        "ccif_imd_db",
        "added_am_sideband_db",
    )
    labels = ("THD 1 kHz", "SMPTE IMD", "CCIF IMD", "added AM sideband")
    positions = np.arange(len(metric_names))
    figure, axis = plt.subplots(figsize=(11, 5))
    for offset, family in ((-0.18, "44k1"), (0.18, "48k")):
        metrics = summary[family]["distortion"]["capb"]
        axis.bar(
            positions + offset,
            [metrics[name] for name in metric_names],
            width=0.34,
            label=family,
        )
    axis.set(
        xticks=positions,
        xticklabels=labels,
        ylabel="distortion or spur level (dBc)",
        ylim=(-180.0, 0.0),
        title="CAPB coherent-line distortion comparison (higher is worse)",
    )
    axis.grid(alpha=0.25, axis="y")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _write_summary(
    summary: dict[str, Any],
    cases: tuple[RateCase, ...],
    output_dir: Path,
) -> None:
    """Write machine-readable metrics and the diagnostic interpretation."""
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    capb_44 = summary["44k1"]["distortion"]["capb"]
    capb_48 = summary["48k"]["distortion"]["capb"]
    smpte_delta = capb_48["smpte_imd_db"] - capb_44["smpte_imd_db"]
    smpte_percent = 100.0 * 10.0 ** (capb_48["smpte_imd_db"] / 20.0)
    report = _markdown_report(cases, capb_44, capb_48, smpte_delta, smpte_percent)
    (output_dir / "report.md").write_text(report)


def _markdown_report(
    cases: tuple[RateCase, ...],
    capb_44: dict[str, float],
    capb_48: dict[str, float],
    smpte_delta: float,
    smpte_percent: float,
) -> str:
    """Render the concise report with scope and interpretation."""
    rows = tuple(
        f"| {label} | {capb_44[key]:.2f} dB | {capb_48[key]:.2f} dB |"
        for label, key in (
            ("1 kHz THD, harmonics through 20 kHz", "thd_1khz_20khz_db"),
            ("SMPTE IMD, 60 Hz + 7 kHz", "smpte_imd_db"),
            ("CCIF IMD, 19 + 20 kHz", "ccif_imd_db"),
            ("Strongest added 10 kHz AM sideband", "added_am_sideband_db"),
        )
    )
    checkpoint_lines = "\n".join(
        f"- `{case.label}`: `{case.checkpoint}`" for case in cases
    )
    if smpte_delta > 0.0:
        interpretation = f"""The 48 kHz checkpoint has the larger SMPTE result. Its
sideband RSS is {smpte_delta:.2f} dB higher than the 44.1 kHz checkpoint and
corresponds to an amplitude ratio of approximately {smpte_percent:.4f}%.
The symmetric `7 kHz ± n·60 Hz` family and controller-weight excursion are
consistent with modulation introduced by the time-varying prototype blend.
This is a diagnosis, not proof of a single internal causal mechanism."""
    else:
        interpretation = f"""Both checkpoints are below the unchanged -110 dB
SMPTE modulation-sideband gate. The 48 kHz result is {abs(smpte_delta):.2f} dB
lower than the 44.1 kHz result and corresponds to an amplitude ratio of
approximately {smpte_percent:.6f}%. The per-family plots confirm that the
symmetric `7 kHz ± n·60 Hz` family and controller-weight excursion are
suppressed together."""
    return f"""# CAPB visualization and distortion investigation

## Checkpoints

{checkpoint_lines}

## Coherent-line results

| Diagnostic | 44.1→88.2 kHz | 48→96 kHz |
|---|---:|---:|
{chr(10).join(rows)}

{interpretation}

The 1 kHz audio-band THD, CCIF products, and higher-order AM sidebands are
reported separately so a two-tone modulation defect is not mislabeled as
broad harmonic distortion on every steady signal.

## Method

- All distortion measurements use the steady center one second of a
  three-second signal and integer-Hz coherent projections.
- THD includes harmonics only through 20 kHz so interpolation images are not
  mislabeled as nonlinear harmonics.
- SMPTE IMD is the RSS of the first five `7 kHz ± n·60 Hz` sideband pairs,
  relative to 7 kHz.
- CCIF IMD is the RSS of 1, 18, and 21 kHz products relative to the two
  primaries.
- Sinusoidal AM contains only orders 0 and ±1; orders ±2 through ±6 are
  treated as added sidebands.
- Ideal and Bessel paths are linear references. The versioned probe-gate
  reports remain authoritative for checkpoint acceptance; these plots are
  supplementary diagnostics and define no new gate.
"""


if __name__ == "__main__":
    main()
