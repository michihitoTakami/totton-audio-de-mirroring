"""CAPB Phase 0b: prototype parameter sweep without sharp-split projection.

Phase 0 showed that any sharp spectral cut (including the shared-passband
projection's 20 kHz split) injects a Gibbs plateau-ripple floor on square
probes. This sweep evaluates matched-passband prototypes whose only degree of
freedom is the transition band (edge frequencies, attenuation): plateau
ripple vs Bessel reference, overshoot, image-band suppression, and
inter-prototype passband matching (the structural bound on what a convex
blend can do to the low band).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.reference import upsample_bessel_reference
from totton_audio_de_mirroring.evaluation.time_domain_visualization import (
    compare_edge_aligned_ringing,
)

SOURCE_SR = 44_100
TARGET_SR = 88_200
RATIO = 2
SQUARE_FREQUENCIES_HZ = (50.0, 100.0, 500.0)
SQUARE_AMPLITUDE = 0.5
PROBE_DURATION_SEC = 1.0
IMAGE_BAND_LOW_HZ = 22_550.0
MATCH_BAND_HIGH_HZ = 19_000.0
NORMALIZATION_FREQ_HZ = 1_000.0
_RESPONSE_FFT_SIZE = 1 << 17

# (name, passband_edge_hz, stopband_edge_hz, attenuation_db)
CANDIDATES: tuple[tuple[str, float, float, float], ...] = (
    ("sharp_20p5_22p05_a90", 20_500.0, 22_050.0, 90.0),
    ("sharp_21_22p05_a90", 21_000.0, 22_050.0, 90.0),
    ("mid_20_23_a70", 20_000.0, 23_000.0, 70.0),
    ("mid_19p5_23_a80", 19_500.0, 23_000.0, 80.0),
    ("smooth_19_23_a80", 19_000.0, 23_000.0, 80.0),
    ("smooth_19_24_a80", 19_000.0, 24_000.0, 80.0),
    ("smooth_19_25_a80", 19_000.0, 25_000.0, 80.0),
    ("smooth_18_26_a80", 18_000.0, 26_000.0, 80.0),
    ("smooth_19_23_a60", 19_000.0, 23_000.0, 60.0),
    ("smooth_17_27_a90", 17_000.0, 27_000.0, 90.0),
)


def main() -> None:
    """Run the prototype transition-band sweep and print the result table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/capb_phase0"))
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    kernels = {
        name: _design_transition_prototype(pb, sb, atten)
        for name, pb, sb, atten in CANDIDATES
    }
    probes = _build_probes()
    befores = {
        probe_id: upsample_bessel_reference(
            signal=source,
            source_sr=SOURCE_SR,
            target_sr=TARGET_SR,
            cutoff_hz=20_000.0,
            order=6,
        )
        for probe_id, source in probes.items()
    }

    rows: list[dict[str, Any]] = []
    for name, kernel in kernels.items():
        row: dict[str, Any] = {
            "name": name,
            "num_taps": int(kernel.size),
            "support_ms": 1_000.0 * kernel.size / TARGET_SR,
        }
        for probe_id, source in probes.items():
            after = _upsample(source, kernel)
            before = befores[probe_id]
            if probe_id.startswith("square_"):
                cmp_ = compare_edge_aligned_ringing(
                    before_signal=before, after_signal=after, sample_rate=TARGET_SR
                )
                row[f"{probe_id}_rms_after"] = cmp_.after.plateau_ripple_rms
                row[f"{probe_id}_rms_ratio"] = cmp_.plateau_ripple_rms_ratio
                row[f"{probe_id}_overshoot_delta"] = cmp_.overshoot_abs_delta
                row[f"{probe_id}_ringing_delta"] = cmp_.ringing_ratio_delta
            else:
                row[f"{probe_id}_image_after_db"] = _band_level_db(after)
                row[f"{probe_id}_image_before_db"] = _band_level_db(before)
        rows.append(row)

    match_matrix = _passband_match_matrix(kernels)

    payload = {"rows": rows, "passband_match_db": match_matrix}
    json_path = args.report_dir / "phase0b_sweep.json"
    json_path.write_text(json.dumps(payload, indent=2))

    print(_render_table(rows))
    print("\n## Inter-prototype passband match (<=19 kHz, dB rel. gain)\n")
    for pair, dev_db in sorted(match_matrix.items()):
        print(f"- {pair}: {dev_db:.1f} dB")
    print(f"\nWrote {json_path}")


def _design_transition_prototype(
    passband_edge_hz: float, stopband_edge_hz: float, attenuation_db: float
) -> np.ndarray:
    """Design a Kaiser interpolation FIR from transition-band edges.

    Physical Basis:
        The Gibbs plateau-ripple tail after a step decays on a timescale of
        1/transition_width, so widening the transition is the direct control
        for low-frequency square-plateau ripple.
    """
    width_hz = stopband_edge_hz - passband_edge_hz
    num_taps, beta = sp_signal.kaiserord(attenuation_db, width_hz / (TARGET_SR / 2))
    if num_taps % 2 == 0:
        num_taps += 1
    cutoff_hz = 0.5 * (passband_edge_hz + stopband_edge_hz)
    taps = sp_signal.firwin(
        num_taps, cutoff_hz, window=("kaiser", beta), fs=TARGET_SR
    ).astype(np.float64)
    _, response = sp_signal.freqz(taps, worN=[NORMALIZATION_FREQ_HZ], fs=TARGET_SR)
    return taps * (float(RATIO) / float(np.abs(response[0])))


def _upsample(source: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    stuffed = np.zeros(source.size * RATIO, dtype=np.float64)
    stuffed[::RATIO] = source
    delay = (kernel.size - 1) // 2
    full = sp_signal.fftconvolve(stuffed, kernel)
    return full[delay : delay + stuffed.size]


def _build_probes() -> dict[str, np.ndarray]:
    num_samples = int(SOURCE_SR * PROBE_DURATION_SEC)
    time_axis = np.arange(num_samples, dtype=np.float64) / SOURCE_SR
    probes: dict[str, np.ndarray] = {}
    for freq in SQUARE_FREQUENCIES_HZ:
        probes[f"square_{int(freq)}hz"] = SQUARE_AMPLITUDE * np.asarray(
            sp_signal.square(2.0 * np.pi * freq * time_axis), dtype=np.float64
        )
    rng = np.random.default_rng(20260704)
    tones = np.zeros(num_samples, dtype=np.float64)
    for freq in np.geomspace(100.0, 20_000.0, 60):
        tones += np.sin(2.0 * np.pi * freq * time_axis + rng.uniform(0, 2 * np.pi))
    probes["multitone_60"] = 0.5 * tones / np.max(np.abs(tones))
    return probes


def _band_level_db(signal: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / TARGET_SR)
    band = freqs >= IMAGE_BAND_LOW_HZ
    level = np.sqrt(np.mean(spectrum[band] ** 2)) / signal.size
    return float(20.0 * np.log10(max(level, 1e-300)))


def _passband_match_matrix(kernels: dict[str, np.ndarray]) -> dict[str, float]:
    """Worst pairwise response deviation below MATCH_BAND_HIGH_HZ in dB."""
    freqs = np.fft.rfftfreq(_RESPONSE_FFT_SIZE, d=1.0 / TARGET_SR)
    in_band = freqs <= MATCH_BAND_HIGH_HZ
    responses = {
        name: np.abs(np.fft.rfft(kernel, n=_RESPONSE_FFT_SIZE))[in_band]
        for name, kernel in kernels.items()
    }
    names = sorted(responses)
    matrix: dict[str, float] = {}
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            dev = float(np.max(np.abs(responses[name_a] - responses[name_b]))) / RATIO
            matrix[f"{name_a} vs {name_b}"] = float(20.0 * np.log10(max(dev, 1e-300)))
    return matrix


def _render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# CAPB Phase 0b: transition-band sweep (no projection)",
        "",
        "| candidate | taps | support ms | sq50 rms abs | sq50 ratio |"
        " sq50 ovsh Δ | sq500 rms abs | sq500 ratio | sq500 ring Δ |"
        " image after (multitone, dB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['num_taps']} | {row['support_ms']:.2f} |"
            f" {row['square_50hz_rms_after']:.2e} |"
            f" {row['square_50hz_rms_ratio']:.2f} |"
            f" {row['square_50hz_overshoot_delta']:+.2e} |"
            f" {row['square_500hz_rms_after']:.2e} |"
            f" {row['square_500hz_rms_ratio']:.2f} |"
            f" {row['square_500hz_ringing_delta']:+.3f} |"
            f" {row['multitone_60_image_after_db']:.1f} |"
        )
    ref = rows[0]
    lines.append("")
    lines.append(
        f"(multitone image band before, Bessel reference: "
        f"{ref['multitone_60_image_before_db']:.1f} dB)"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
