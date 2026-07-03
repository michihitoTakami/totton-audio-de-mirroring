"""CAPB Phase 0: structural proof of the fixed prototype bank.

Runs the fixed prototype bank as constant-weight dummy models against the
Bessel reference SRC and reports, per prototype: square-probe ringing gates,
image-band (mirror) suppression, and low-band gain accuracy. No training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.degradation import upsample_bessel_reference
from totton_audio_de_mirroring.evaluation.time_domain_visualization import (
    compare_edge_aligned_ringing,
)
from totton_audio_de_mirroring.models.proto_bank import (
    DEFAULT_PROTOTYPE_SPECS,
    PrototypeBank,
    blend_modulation_bounds,
    build_prototype_bank,
    summarize_bank,
    upsample_with_kernel,
    validate_bank,
)

SOURCE_SR = 44_100
TARGET_SR = 88_200
SQUARE_FREQUENCIES_HZ = (50.0, 100.0, 500.0, 1_000.0, 2_000.0, 5_000.0)
SQUARE_AMPLITUDE = 0.5
PROBE_DURATION_SEC = 1.0
IMAGE_BAND_LOW_HZ = 22_550.0
LB_BAND_HIGH_HZ = 19_000.0
MAX_PLATEAU_RIPPLE_RATIO = 1.10
MAX_OVERSHOOT_INCREASE = 5.0e-3
MULTITONE_SEED = 20260704


def main() -> None:
    """Run Phase 0 structural validation and write the gate table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/capb_phase0"),
    )
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    bank = build_prototype_bank()
    validation = validate_bank(bank)
    modulation_bounds = blend_modulation_bounds(bank)
    bank_summary = summarize_bank(bank)

    probes = _build_probes()
    results = _evaluate_bank(bank, probes)

    payload: dict[str, Any] = {
        "validation": validation,
        "blend_modulation_bounds_db": modulation_bounds,
        "prototype_specs": [asdict(spec) for spec in DEFAULT_PROTOTYPE_SPECS],
        "bank_summary": bank_summary,
        "results": results,
    }
    json_path = args.report_dir / "phase0_results.json"
    json_path.write_text(json.dumps(payload, indent=2))

    markdown = _render_markdown(validation, modulation_bounds, bank_summary, results)
    md_path = args.report_dir / "phase0_gate_table.md"
    md_path.write_text(markdown)
    print(markdown)
    print(f"Wrote {json_path} and {md_path}")


def _build_probes() -> dict[str, np.ndarray]:
    """Build deterministic 44.1 kHz probe signals.

    Returns:
        Mapping of probe id to source-rate waveform.

    Physical Basis:
        Square waves expose plateau ripple/overshoot; a log sweep and a dense
        multitone exercise the full band so image leakage above the input
        Nyquist is measurable.
    """
    num_samples = int(SOURCE_SR * PROBE_DURATION_SEC)
    time_axis = np.arange(num_samples, dtype=np.float64) / SOURCE_SR

    probes: dict[str, np.ndarray] = {}
    for freq in SQUARE_FREQUENCIES_HZ:
        probes[f"square_{int(freq)}hz"] = SQUARE_AMPLITUDE * np.asarray(
            sp_signal.square(2.0 * np.pi * freq * time_axis), dtype=np.float64
        )

    probes["sweep_log_20_20k"] = 0.5 * sp_signal.chirp(
        time_axis,
        f0=20.0,
        f1=20_000.0,
        t1=PROBE_DURATION_SEC,
        method="logarithmic",
    ).astype(np.float64)

    rng = np.random.default_rng(MULTITONE_SEED)
    tones = np.zeros(num_samples, dtype=np.float64)
    for freq in np.geomspace(100.0, 20_000.0, 60):
        tones += np.sin(2.0 * np.pi * freq * time_axis + rng.uniform(0, 2 * np.pi))
    probes["multitone_60"] = 0.5 * tones / np.max(np.abs(tones))

    return probes


def _evaluate_bank(
    bank: PrototypeBank, probes: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Evaluate each fixed prototype against the Bessel reference SRC."""
    results: dict[str, Any] = {}
    for index, name in enumerate(bank.names):
        kernel = bank.kernels[index]
        per_probe: dict[str, Any] = {}
        for probe_id, source in probes.items():
            before = upsample_bessel_reference(
                signal=source,
                source_sr=SOURCE_SR,
                target_sr=TARGET_SR,
                cutoff_hz=20_000.0,
                order=6,
            )
            after = upsample_with_kernel(source, kernel, bank.upsample_ratio)
            per_probe[probe_id] = _probe_metrics(probe_id, before, after)
        results[name] = per_probe
    return results


def _probe_metrics(
    probe_id: str, before: np.ndarray, after: np.ndarray
) -> dict[str, Any]:
    """Compute gate-relevant metrics for one probe pair."""
    metrics: dict[str, Any] = {
        "image_band_before_db": _band_level_db(
            before, IMAGE_BAND_LOW_HZ, TARGET_SR / 2
        ),
        "image_band_after_db": _band_level_db(after, IMAGE_BAND_LOW_HZ, TARGET_SR / 2),
        "lb_gain_error_db": _lb_gain_error_db(before, after),
    }
    metrics["image_band_reduction_db"] = (
        metrics["image_band_before_db"] - metrics["image_band_after_db"]
    )

    if probe_id.startswith("square_"):
        comparison = compare_edge_aligned_ringing(
            before_signal=before,
            after_signal=after,
            sample_rate=TARGET_SR,
        )
        metrics.update(
            {
                "plateau_ripple_rms_ratio": comparison.plateau_ripple_rms_ratio,
                "plateau_ripple_p2p_ratio": comparison.plateau_ripple_p2p_ratio,
                "overshoot_abs_delta": comparison.overshoot_abs_delta,
                "ringing_ratio_delta": comparison.ringing_ratio_delta,
                "plateau_ripple_rms_after": comparison.after.plateau_ripple_rms,
                "plateau_ripple_rms_before": comparison.before.plateau_ripple_rms,
                "passes_ringing_gate": bool(
                    comparison.plateau_ripple_rms_ratio <= MAX_PLATEAU_RIPPLE_RATIO
                    and comparison.plateau_ripple_p2p_ratio <= MAX_PLATEAU_RIPPLE_RATIO
                    and comparison.overshoot_abs_delta <= MAX_OVERSHOOT_INCREASE
                    and comparison.ringing_ratio_delta <= 0.0
                ),
            }
        )
    return metrics


def _band_level_db(signal: np.ndarray, low_hz: float, high_hz: float) -> float:
    """Return mean spectral level of a band in dB relative to full scale."""
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / TARGET_SR)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    level = np.sqrt(np.mean(spectrum[band] ** 2)) / signal.size
    return float(20.0 * np.log10(max(level, 1e-300)))


def _lb_gain_error_db(before: np.ndarray, after: np.ndarray) -> float:
    """Return low-band RMS gain error of `after` relative to `before`."""
    taps = sp_signal.firwin(1025, LB_BAND_HIGH_HZ, fs=TARGET_SR)
    lb_before = sp_signal.fftconvolve(before, taps, mode="same")
    lb_after = sp_signal.fftconvolve(after, taps, mode="same")
    rms_before = float(np.sqrt(np.mean(lb_before**2)))
    rms_after = float(np.sqrt(np.mean(lb_after**2)))
    return float(20.0 * np.log10(max(rms_after, 1e-300) / max(rms_before, 1e-300)))


def _render_markdown(
    validation: dict[str, float],
    modulation_bounds: dict[str, float],
    bank_summary: dict[str, dict[str, float]],
    results: dict[str, Any],
) -> str:
    """Render the Phase 0 gate table as markdown."""
    lines = [
        "# CAPB Phase 0: prototype bank structural proof",
        "",
        "## Structural validation",
        "",
    ]
    for key, value in validation.items():
        lines.append(f"- {key}: {value:.6g}")
    lines += [
        "",
        "## Blend modulation bounds (worst pairwise response spread)",
        "",
    ]
    for key, value in modulation_bounds.items():
        lines.append(f"- {key} Hz: {value:.1f} dB")
    lines += [
        "",
        "## Prototype frequency responses",
        "",
        "| prototype | passband dev (dB) | image band max (dB) |"
        " >=24k max (dB) | response @20k (dB) |",
        "|---|---|---|---|---|",
    ]
    for name, stats in bank_summary.items():
        lines.append(
            f"| {name} | {stats['passband_dev_db']:.1f} |"
            f" {stats['image_band_max_db']:.1f} |"
            f" {stats['deep_image_max_db']:.1f} |"
            f" {stats['response_20k_db']:.2f} |"
        )

    lines += [
        "",
        "## Square-probe ringing gates (vs Bessel reference SRC)",
        "",
        "| prototype | probe | rms ratio | p2p ratio | overshoot Δ |"
        " ringing Δ | abs rms after | gate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, per_probe in results.items():
        for probe_id, metrics in per_probe.items():
            if "plateau_ripple_rms_ratio" not in metrics:
                continue
            gate = "PASS" if metrics["passes_ringing_gate"] else "FAIL"
            lines.append(
                f"| {name} | {probe_id} |"
                f" {metrics['plateau_ripple_rms_ratio']:.3f} |"
                f" {metrics['plateau_ripple_p2p_ratio']:.3f} |"
                f" {metrics['overshoot_abs_delta']:+.2e} |"
                f" {metrics['ringing_ratio_delta']:+.3f} |"
                f" {metrics['plateau_ripple_rms_after']:.2e} |"
                f" {gate} |"
            )

    lines += [
        "",
        "## Image-band suppression and LB gain (all probes)",
        "",
        "| prototype | probe | image before (dB) | image after (dB) |"
        " reduction (dB) | LB gain err (dB) |",
        "|---|---|---|---|---|---|",
    ]
    for name, per_probe in results.items():
        for probe_id, metrics in per_probe.items():
            lines.append(
                f"| {name} | {probe_id} |"
                f" {metrics['image_band_before_db']:.1f} |"
                f" {metrics['image_band_after_db']:.1f} |"
                f" {metrics['image_band_reduction_db']:+.1f} |"
                f" {metrics['lb_gain_error_db']:+.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
