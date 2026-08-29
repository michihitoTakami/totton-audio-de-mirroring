"""Worst-case per-probe gate evaluation for Stage 1 candidates.

Design principles (fixing the flaws that let a 15.8x plateau-ripple pass):

1. Every gate binds on the WORST probe; means are reported, never gated on.
2. Every relative threshold is paired with an absolute floor so degenerate
   references (e.g. a near-zero Bessel plateau ripple) cannot explode ratios.
3. Ringing metrics are computed only on probes that actually have plateaus.
4. Dual references: ringing gates compare against the Bessel reference SRC
   (hard requirement: no ringing regression vs reference), while fidelity
   gates (flatness, gain, low-band preservation) compare against an ideal
   linear-phase polyphase resampler of the 44.1 kHz source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import signal as sp_signal

from totton_audio_de_mirroring.evaluation.distortion import smpte_imd_db
from totton_audio_de_mirroring.evaluation.lb_preservation import (
    evaluate_lowband_preservation,
)
from totton_audio_de_mirroring.evaluation.passband_flatness import (
    compute_flatness,
    compute_lowband_gain_error_db,
)
from totton_audio_de_mirroring.evaluation.probe_suite import (
    KIND_DC_STEP,
    KIND_IMD_TWO_TONE,
    KIND_IMPULSE,
    KIND_IMPULSE_TRAIN,
    KIND_MULTITONE,
    KIND_PINK_NOISE,
    KIND_SQUARE,
    KIND_SWEEP_LOG,
    KIND_TONE_BURST,
    ProbeSpec,
)
from totton_audio_de_mirroring.evaluation.time_domain_visualization import (
    compare_edge_aligned_ringing,
)

GATE_SPEC_VERSION = 4
_EPSILON = 1e-300

LF_RINGING_MAX_FREQ_HZ = 2_000.0
# Image-band guard offset above the input Nyquist; the actual band edge is
# rate-dependent (22.55 kHz at 88.2k target, 24.5 kHz at 96k target).
IMAGE_BAND_NYQUIST_OFFSET_HZ = 500.0
IMAGE_BAND_LOW_HZ = 22_550.0  # 44.1k-family value, kept for reference.
MAIN_BAND_HIGH_HZ = 20_000.0
GAIN_BAND_HIGH_HZ = 10_000.0
GAIN_APPLICABILITY_HIGH_HZ = 8_000.0
GAIN_MIN_BAND_FRACTION = 0.05
_ECHO_GUARD_MS = 0.5
_ECHO_WINDOW_MS = 3.5


@dataclass(frozen=True)
class Stage1GateConfig:
    """Thresholds for the probe-based Stage 1 gates.

    Args:
        plateau_ripple_ratio_max: Relative plateau-ripple bound (after/before).
        plateau_rms_floor_rel: Absolute RMS floor relative to probe amplitude.
        plateau_p2p_floor_rel: Absolute P2P floor relative to probe amplitude.
        overshoot_delta_max: Allowed overshoot increase over the reference.
        overshoot_floor_rel: Absolute overshoot floor relative to amplitude.
        pre_echo_energy_ratio_max: Allowed pre-echo energy growth (energy).
        pre_echo_floor_rel: Amplitude-relative pre-echo absolute floor.
        image_rel_max_db: Max image-band level relative to main band (steady).
        image_peak_rel_max_db: Max swept-ridge image peak relative to main peak.
        added_hf_max_db: Max image-band increase over the Bessel reference.
        flatness_dip_max_db: Max smoothed response dip 100 Hz-18 kHz.
        flatness_boost_max_db: Max smoothed response boost 100 Hz-18 kHz.
        flatness_hf_dip_max_db: Max smoothed response dip 18-20 kHz.
        gain_error_max_db: Max low-band RMS gain error vs ideal reference.
        lb_phase_error_max_deg: Max low-band phase error vs ideal reference.
        lb_group_delay_error_max_samples: Max low-band group-delay error.
        lb_waveform_error_max_db: Max low-band waveform error in dB.
        modulation_sideband_max_db: Maximum two-tone modulation products in dBc.

    Physical Basis:
        Relative terms guard regressions against the reference SRC; absolute
        floors encode audibility-scaled limits so that comparisons between
        two inaudibly small quantities can never fail or pass a gate.
    """

    plateau_ripple_ratio_max: float = 1.10
    plateau_rms_floor_rel: float = 1.0e-3
    plateau_p2p_floor_rel: float = 3.16e-3
    overshoot_delta_max: float = 5.0e-3
    overshoot_floor_rel: float = 0.02
    pre_echo_energy_ratio_max: float = 1.44
    pre_echo_floor_rel: float = 1.0e-3
    image_rel_max_db: float = -65.0
    image_peak_rel_max_db: float = -65.0
    added_hf_max_db: float = 3.0
    image_negligible_rel_db: float = -70.0
    flatness_dip_max_db: float = 1.0
    flatness_boost_max_db: float = 1.0
    flatness_hf_dip_max_db: float = 3.0
    gain_error_max_db: float = 0.5
    lb_phase_error_max_deg: float = 15.0
    lb_group_delay_error_max_samples: float = 600.0
    lb_waveform_error_max_db: float = -20.0
    modulation_sideband_max_db: float = -110.0


@dataclass(frozen=True)
class ProbeEvaluation:
    """Raw metrics computed for one probe.

    Args:
        spec: The probe specification.
        metrics: Flat metric mapping computed by evaluate_probe.

    Physical Basis:
        Separating metric computation from gating keeps a single metric
        source of truth that multiple gates (and reports) can consume.
    """

    spec: ProbeSpec
    metrics: dict[str, float]


@dataclass(frozen=True)
class GateRow:
    """One gated criterion on one probe."""

    probe_id: str
    tier: str
    metric: str
    value: float
    threshold: float
    passed: bool
    binding: str


@dataclass(frozen=True)
class GateResult:
    """Result of one gate over all applicable probes."""

    gate_id: str
    passed: bool
    worst_probe_id: str | None
    rows: tuple[GateRow, ...]


@dataclass(frozen=True)
class GateReport:
    """Full gate report for one candidate.

    Args:
        all_passed: True when every gate passed on every applicable probe.
        gates: Per-gate results.
        spec_version: Gate specification version.
        manifest_hash: Hash of the probe manifest used.
        config: Threshold configuration snapshot.

    Physical Basis:
        A pass is only meaningful relative to the probe suite and threshold
        set it was earned against, so both are embedded in the report.
    """

    all_passed: bool
    gates: tuple[GateResult, ...]
    spec_version: int = GATE_SPEC_VERSION
    manifest_hash: str = ""
    config: dict[str, float] = field(default_factory=dict)


def evaluate_probe(
    spec: ProbeSpec,
    source: np.ndarray,
    source_sample_rate: int,
    bessel_reference: np.ndarray,
    ideal_reference: np.ndarray,
    output: np.ndarray,
    target_sample_rate: int,
) -> ProbeEvaluation:
    """Compute all gate-relevant metrics for one probe.

    Args:
        spec: Probe specification.
        source: Source-rate probe waveform.
        source_sample_rate: Source sample rate in Hz.
        bessel_reference: Bessel reference SRC output at the target rate.
        ideal_reference: Ideal linear-phase resampler output at target rate.
        output: Candidate system output at the target rate.
        target_sample_rate: Target sample rate in Hz.

    Returns:
        ProbeEvaluation with the flat metric mapping.

    Raises:
        ValueError: If signal lengths are inconsistent.

    Physical Basis:
        Ringing metrics are edge-aligned per signal, so exact time alignment
        between references and output is not required; spectral metrics are
        alignment-free by construction.
    """
    if output.shape != bessel_reference.shape:
        raise ValueError(
            "output and bessel_reference must share a shape, got "
            f"{output.shape} vs {bessel_reference.shape}."
        )

    image_low_hz = image_band_low_hz(target_sample_rate)
    metrics: dict[str, float] = {}
    metrics["image_rel_db"] = _image_minus_main_db(output, target_sample_rate)
    if spec.kind == KIND_SWEEP_LOG:
        metrics["image_peak_rel_db"] = _peak_image_minus_main_db(
            output, target_sample_rate
        )
    metrics["image_abs_db"] = _band_level_db(
        output, target_sample_rate, image_low_hz, target_sample_rate / 2
    )
    metrics["image_before_abs_db"] = _band_level_db(
        bessel_reference, target_sample_rate, image_low_hz, target_sample_rate / 2
    )
    metrics["gain_error_db"] = compute_lowband_gain_error_db(
        source,
        source_sample_rate,
        output,
        target_sample_rate,
        cutoff_hz=GAIN_BAND_HIGH_HZ,
    )
    metrics["gain_band_fraction"] = _band_energy_fraction(
        source, source_sample_rate, GAIN_APPLICABILITY_HIGH_HZ
    )

    if spec.kind in {KIND_SQUARE, KIND_DC_STEP}:
        comparison = compare_edge_aligned_ringing(
            before_signal=bessel_reference,
            after_signal=output,
            sample_rate=target_sample_rate,
        )
        metrics.update(
            {
                "plateau_rms_before": comparison.before.plateau_ripple_rms,
                "plateau_rms_after": comparison.after.plateau_ripple_rms,
                "plateau_p2p_before": comparison.before.plateau_ripple_p2p,
                "plateau_p2p_after": comparison.after.plateau_ripple_p2p,
                "overshoot_before": comparison.before.overshoot_abs,
                "overshoot_after": comparison.after.overshoot_abs,
            }
        )

    if spec.kind in {KIND_IMPULSE, KIND_IMPULSE_TRAIN, KIND_TONE_BURST}:
        center = _event_center_index(spec, output.size, target_sample_rate)
        pre_before, _ = _event_echo_energies(
            bessel_reference, center, target_sample_rate
        )
        pre_after, post_after = _event_echo_energies(output, center, target_sample_rate)
        metrics.update(
            {
                "pre_echo_energy_before": pre_before,
                "pre_echo_energy_after": pre_after,
                "post_echo_energy_after": post_after,
            }
        )

    if spec.kind in {KIND_PINK_NOISE, KIND_MULTITONE}:
        flatness = compute_flatness(
            source, source_sample_rate, output, target_sample_rate
        )
        metrics.update(
            {
                "flatness_dip_db": flatness.max_dip_db,
                "flatness_boost_db": flatness.max_boost_db,
                "flatness_hf_dip_db": flatness.hf_max_dip_db,
            }
        )
    if spec.kind == KIND_IMD_TWO_TONE:
        low_hz = spec.frequency_hz
        high_hz = spec.secondary_frequency_hz
        if low_hz is None or high_hz is None:
            raise ValueError("IMD probes require both primary frequencies.")
        center = output.size // 2
        half_window = target_sample_rate // 2
        steady = output[center - half_window : center + half_window]
        metrics["modulation_sideband_db"] = smpte_imd_db(
            steady,
            target_sample_rate,
            low_tone_hz=low_hz,
            high_tone_hz=high_hz,
        )

    if spec.kind in {KIND_PINK_NOISE, KIND_MULTITONE}:
        lb = evaluate_lowband_preservation(
            input_signal=ideal_reference,
            output_signal=output,
            sample_rate=target_sample_rate,
        )
        metrics.update(
            {
                "lb_phase_error_deg": lb.phase_error_deg,
                "lb_group_delay_error_samples": lb.group_delay_error_samples,
                "lb_waveform_error_db": lb.waveform_error_db,
            }
        )
    return ProbeEvaluation(spec=spec, metrics=metrics)


def evaluate_gates(
    evaluations: list[ProbeEvaluation],
    config: Stage1GateConfig | None = None,
    manifest_hash: str = "",
) -> GateReport:
    """Evaluate all gates over the probe evaluations (worst-case pass rule).

    Args:
        evaluations: Per-probe metric evaluations.
        config: Gate thresholds (defaults to Stage1GateConfig()).
        manifest_hash: Probe-manifest hash to embed in the report.

    Returns:
        GateReport with per-gate, per-probe rows.

    Raises:
        ValueError: If evaluations is empty.

    Physical Basis:
        A single catastrophic probe (e.g. a 500 Hz square with 15.8x plateau
        ripple) must fail the candidate outright; averaging across benign
        probes is how the previous gate design masked exactly that failure.
    """
    if not evaluations:
        raise ValueError("evaluations must not be empty.")
    cfg = config or Stage1GateConfig()

    gates = (
        _gate_ringing("G1_lf_ringing", evaluations, cfg, low_frequency=True),
        _gate_ringing("G2_hf_ringing", evaluations, cfg, low_frequency=False),
        _gate_pre_echo(evaluations, cfg),
        _gate_mirror(evaluations, cfg),
        _gate_flatness(evaluations, cfg),
        _gate_gain(evaluations, cfg),
        _gate_added_hf(evaluations, cfg),
        _gate_lb_preservation(evaluations, cfg),
        _gate_modulation_sidebands(evaluations, cfg),
    )
    return GateReport(
        all_passed=all(gate.passed for gate in gates),
        gates=gates,
        manifest_hash=manifest_hash,
        config={key: float(value) for key, value in asdict(cfg).items()},
    )


def report_to_dict(report: GateReport) -> dict[str, Any]:
    """Serialize a GateReport into a JSON-compatible dictionary."""
    return {
        "all_passed": report.all_passed,
        "spec_version": report.spec_version,
        "manifest_hash": report.manifest_hash,
        "config": report.config,
        "gates": [
            {
                "gate_id": gate.gate_id,
                "passed": gate.passed,
                "worst_probe_id": gate.worst_probe_id,
                "rows": [asdict(row) for row in gate.rows],
            }
            for gate in report.gates
        ],
    }


def render_markdown_report(report: GateReport) -> str:
    """Render a GateReport as a per-probe markdown table (worst-first)."""
    lines = [
        "# Stage 1 probe gate report",
        "",
        f"- all_passed: **{report.all_passed}**",
        f"- spec_version: {report.spec_version}",
        f"- manifest_hash: {report.manifest_hash}",
        "",
    ]
    for gate in report.gates:
        status = "PASS" if gate.passed else "FAIL"
        lines += [
            f"## {gate.gate_id}: {status}"
            + (f" (worst: {gate.worst_probe_id})" if gate.worst_probe_id else ""),
            "",
            "| probe | tier | metric | value | threshold | binding | pass |",
            "|---|---|---|---|---|---|---|",
        ]
        rows = sorted(gate.rows, key=lambda row: row.passed)
        for row in rows:
            lines.append(
                f"| {row.probe_id} | {row.tier} | {row.metric} |"
                f" {row.value:.4g} | {row.threshold:.4g} | {row.binding} |"
                f" {'PASS' if row.passed else 'FAIL'} |"
            )
        lines.append("")
    return "\n".join(lines)


def _gate_ringing(
    gate_id: str,
    evaluations: list[ProbeEvaluation],
    cfg: Stage1GateConfig,
    low_frequency: bool,
) -> GateResult:
    rows: list[GateRow] = []
    for evaluation in evaluations:
        spec = evaluation.spec
        if "plateau_rms_after" not in evaluation.metrics:
            continue
        freq = spec.frequency_hz or 0.0
        is_low = spec.kind == KIND_DC_STEP or freq <= LF_RINGING_MAX_FREQ_HZ
        if is_low != low_frequency:
            continue
        metrics = evaluation.metrics
        amp = spec.amplitude
        rows += _threshold_rows(
            spec,
            (
                (
                    "plateau_rms_after",
                    metrics["plateau_rms_after"],
                    cfg.plateau_ripple_ratio_max * metrics["plateau_rms_before"],
                    cfg.plateau_rms_floor_rel * amp,
                ),
                (
                    "plateau_p2p_after",
                    metrics["plateau_p2p_after"],
                    cfg.plateau_ripple_ratio_max * metrics["plateau_p2p_before"],
                    cfg.plateau_p2p_floor_rel * amp,
                ),
                (
                    "overshoot_after",
                    metrics["overshoot_after"],
                    metrics["overshoot_before"] + cfg.overshoot_delta_max,
                    cfg.overshoot_floor_rel * amp,
                ),
            ),
        )
    return _finalize(gate_id, rows)


def _gate_pre_echo(
    evaluations: list[ProbeEvaluation], cfg: Stage1GateConfig
) -> GateResult:
    rows: list[GateRow] = []
    for evaluation in evaluations:
        metrics = evaluation.metrics
        if "pre_echo_energy_after" not in metrics:
            continue
        amp = evaluation.spec.amplitude
        rows += _threshold_rows(
            evaluation.spec,
            (
                (
                    "pre_echo_energy_after",
                    metrics["pre_echo_energy_after"],
                    cfg.pre_echo_energy_ratio_max * metrics["pre_echo_energy_before"],
                    (cfg.pre_echo_floor_rel * amp) ** 2,
                ),
            ),
        )
    return _finalize("G2b_pre_echo", rows)


def _gate_mirror(
    evaluations: list[ProbeEvaluation], cfg: Stage1GateConfig
) -> GateResult:
    rows: list[GateRow] = []
    steady = {KIND_SWEEP_LOG, KIND_PINK_NOISE, KIND_MULTITONE}
    for evaluation in evaluations:
        if evaluation.spec.kind not in steady:
            continue
        value = evaluation.metrics["image_rel_db"]
        rows.append(
            GateRow(
                probe_id=evaluation.spec.probe_id,
                tier=evaluation.spec.tier,
                metric="image_rel_db",
                value=value,
                threshold=cfg.image_rel_max_db,
                passed=value <= cfg.image_rel_max_db,
                binding="absolute",
            )
        )
        if evaluation.spec.kind == KIND_SWEEP_LOG:
            peak_value = evaluation.metrics["image_peak_rel_db"]
            rows.append(
                GateRow(
                    probe_id=evaluation.spec.probe_id,
                    tier=evaluation.spec.tier,
                    metric="image_peak_rel_db",
                    value=peak_value,
                    threshold=cfg.image_peak_rel_max_db,
                    passed=peak_value <= cfg.image_peak_rel_max_db,
                    binding="absolute",
                )
            )
    return _finalize("G3_mirror", rows)


def _gate_flatness(
    evaluations: list[ProbeEvaluation], cfg: Stage1GateConfig
) -> GateResult:
    rows: list[GateRow] = []
    for evaluation in evaluations:
        metrics = evaluation.metrics
        if "flatness_dip_db" not in metrics:
            continue
        spec = evaluation.spec
        checks = (
            ("flatness_dip_db", -metrics["flatness_dip_db"], cfg.flatness_dip_max_db),
            (
                "flatness_boost_db",
                metrics["flatness_boost_db"],
                cfg.flatness_boost_max_db,
            ),
            (
                "flatness_hf_dip_db",
                -metrics["flatness_hf_dip_db"],
                cfg.flatness_hf_dip_max_db,
            ),
        )
        for metric, value, threshold in checks:
            rows.append(
                GateRow(
                    probe_id=spec.probe_id,
                    tier=spec.tier,
                    metric=metric,
                    value=float(value),
                    threshold=float(threshold),
                    passed=value <= threshold,
                    binding="absolute",
                )
            )
    return _finalize("G4_flatness", rows)


def _gate_gain(evaluations: list[ProbeEvaluation], cfg: Stage1GateConfig) -> GateResult:
    rows: list[GateRow] = []
    for evaluation in evaluations:
        if evaluation.metrics["gain_band_fraction"] < GAIN_MIN_BAND_FRACTION:
            continue
        value = abs(evaluation.metrics["gain_error_db"])
        rows.append(
            GateRow(
                probe_id=evaluation.spec.probe_id,
                tier=evaluation.spec.tier,
                metric="abs_gain_error_db",
                value=value,
                threshold=cfg.gain_error_max_db,
                passed=value <= cfg.gain_error_max_db,
                binding="absolute",
            )
        )
    return _finalize("G5_gain", rows)


def _gate_added_hf(
    evaluations: list[ProbeEvaluation], cfg: Stage1GateConfig
) -> GateResult:
    rows: list[GateRow] = []
    for evaluation in evaluations:
        metrics = evaluation.metrics
        added = metrics["image_abs_db"] - metrics["image_before_abs_db"]
        # Absolute floor: image levels already far below the main band are
        # irrelevant, so comparing two such tiny levels must not fail a gate.
        negligible = metrics["image_rel_db"] <= cfg.image_negligible_rel_db
        rows.append(
            GateRow(
                probe_id=evaluation.spec.probe_id,
                tier=evaluation.spec.tier,
                metric="added_hf_db",
                value=added,
                threshold=cfg.added_hf_max_db,
                passed=negligible or added <= cfg.added_hf_max_db,
                binding="absolute" if negligible else "relative",
            )
        )
    return _finalize("G7_no_added_hf", rows)


def _gate_lb_preservation(
    evaluations: list[ProbeEvaluation], cfg: Stage1GateConfig
) -> GateResult:
    rows: list[GateRow] = []
    for evaluation in evaluations:
        metrics = evaluation.metrics
        if "lb_phase_error_deg" not in metrics:
            continue
        spec = evaluation.spec
        checks = (
            (
                "lb_phase_error_deg",
                metrics["lb_phase_error_deg"],
                cfg.lb_phase_error_max_deg,
            ),
            (
                "lb_group_delay_error_samples",
                metrics["lb_group_delay_error_samples"],
                cfg.lb_group_delay_error_max_samples,
            ),
            (
                "lb_waveform_error_db",
                metrics["lb_waveform_error_db"],
                cfg.lb_waveform_error_max_db,
            ),
        )
        for metric, value, threshold in checks:
            rows.append(
                GateRow(
                    probe_id=spec.probe_id,
                    tier=spec.tier,
                    metric=metric,
                    value=float(value),
                    threshold=float(threshold),
                    passed=value <= threshold,
                    binding="absolute",
                )
            )
    return _finalize("G8_lb_preservation", rows)


def _gate_modulation_sidebands(
    evaluations: list[ProbeEvaluation], cfg: Stage1GateConfig
) -> GateResult:
    """Gate signal-dependent two-tone sidebands on canonical and held-out probes."""
    rows = [
        GateRow(
            probe_id=evaluation.spec.probe_id,
            tier=evaluation.spec.tier,
            metric="modulation_sideband_db",
            value=evaluation.metrics["modulation_sideband_db"],
            threshold=cfg.modulation_sideband_max_db,
            passed=(
                evaluation.metrics["modulation_sideband_db"]
                <= cfg.modulation_sideband_max_db
            ),
            binding="absolute",
        )
        for evaluation in evaluations
        if "modulation_sideband_db" in evaluation.metrics
    ]
    worst = max(rows, key=lambda row: row.value).probe_id if rows else None
    return GateResult(
        gate_id="G9_no_modulation_sidebands",
        passed=all(row.passed for row in rows),
        worst_probe_id=worst,
        rows=tuple(rows),
    )


def _threshold_rows(
    spec: ProbeSpec,
    checks: tuple[tuple[str, float, float, float], ...],
) -> list[GateRow]:
    """Build rows where threshold = max(relative_term, absolute_floor)."""
    rows: list[GateRow] = []
    for metric, value, relative_term, absolute_floor in checks:
        threshold = max(relative_term, absolute_floor)
        binding = "relative" if relative_term >= absolute_floor else "absolute"
        rows.append(
            GateRow(
                probe_id=spec.probe_id,
                tier=spec.tier,
                metric=metric,
                value=float(value),
                threshold=float(threshold),
                passed=value <= threshold,
                binding=binding,
            )
        )
    return rows


def _finalize(gate_id: str, rows: list[GateRow]) -> GateResult:
    failing = [row for row in rows if not row.passed]
    worst: str | None = None
    if failing:
        worst = max(
            failing, key=lambda row: row.value / max(row.threshold, 1e-12)
        ).probe_id
    elif rows:
        worst = max(
            rows, key=lambda row: row.value / max(row.threshold, 1e-12)
        ).probe_id
    return GateResult(
        gate_id=gate_id,
        passed=not failing,
        worst_probe_id=worst,
        rows=tuple(rows),
    )


def _event_center_index(spec: ProbeSpec, length: int, sample_rate: int) -> int:
    """Return the sample index of the event onset for echo analysis.

    Physical Basis:
        Pre-echo is energy BEFORE the first causal excitation, so tone
        bursts are referenced to their onset (start of the Hann gate), not
        their center, which would place the analysis window inside the burst.
    """
    if spec.kind == KIND_IMPULSE_TRAIN and spec.period_ms:
        period = max(1, int(round(spec.period_ms * sample_rate / 1_000.0)))
        return (length // 2 // period) * period
    if spec.kind == KIND_TONE_BURST and spec.burst_ms:
        burst_len = max(3, int(round(spec.burst_ms * sample_rate / 1_000.0)))
        return (length - burst_len) // 2
    return length // 2


def _event_echo_energies(
    signal: np.ndarray, center: int, sample_rate: int
) -> tuple[float, float]:
    guard = int(round(_ECHO_GUARD_MS * sample_rate / 1_000.0))
    window = int(round(_ECHO_WINDOW_MS * sample_rate / 1_000.0))
    pre = signal[max(0, center - guard - window) : max(0, center - guard)]
    post = signal[center + guard : center + guard + window]
    return _mean_square(pre), _mean_square(post)


def _mean_square(window: np.ndarray) -> float:
    if window.size == 0:
        return 0.0
    return float(np.mean(np.square(window)))


def _band_level_db(
    signal: np.ndarray, sample_rate: int, low_hz: float, high_hz: float
) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band):
        return -300.0
    level = np.sqrt(np.mean(np.square(spectrum[band]))) / signal.size
    return float(20.0 * np.log10(max(level, _EPSILON)))


def _band_energy_fraction(
    signal: np.ndarray, sample_rate: int, cutoff_hz: float
) -> float:
    spectrum = np.square(np.abs(np.fft.rfft(signal)))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    total = float(np.sum(spectrum))
    if total <= 0.0:
        return 0.0
    return float(np.sum(spectrum[freqs <= cutoff_hz]) / total)


def image_band_low_hz(target_sample_rate: int) -> float:
    """Return the image-band lower edge for a 2x-upsampled target rate.

    Physical Basis:
        Mirror images of a 2x upsampler start at the input Nyquist
        (target_rate / 4); the fixed offset keeps the measurement clear of
        brickwall transition skirts. Evaluates to 22 550 Hz at 88.2 kHz and
        24 500 Hz at 96 kHz.
    """
    if target_sample_rate <= 0:
        raise ValueError(
            f"target_sample_rate must be positive, got {target_sample_rate}."
        )
    return target_sample_rate / 4.0 + IMAGE_BAND_NYQUIST_OFFSET_HZ


def _image_minus_main_db(signal: np.ndarray, sample_rate: int) -> float:
    image = _band_level_db(
        signal, sample_rate, image_band_low_hz(sample_rate), sample_rate / 2
    )
    main = _band_level_db(signal, sample_rate, 20.0, MAIN_BAND_HIGH_HZ)
    return image - main


def _peak_image_minus_main_db(signal: np.ndarray, sample_rate: int) -> float:
    """Return peak swept-image ridge relative to the main swept ridge.

    Physical Basis:
        An integrated image-band average can hide a narrow residual confined
        to the end of a sweep. The maximum Hann-STFT magnitude over time
        follows the swept ridge at each frequency and makes that worst-case
        residual binding.
    """
    nperseg = min(2_048, signal.size)
    if nperseg < 16:
        raise ValueError("signal is too short for peak image measurement.")
    frequencies, _, spectrum = sp_signal.stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg * 7 // 8,
        nfft=nperseg,
        boundary=None,
        padded=False,
    )
    envelope = np.max(np.abs(spectrum), axis=1)
    main = (frequencies >= 20.0) & (frequencies <= MAIN_BAND_HIGH_HZ)
    image = frequencies >= image_band_low_hz(sample_rate)
    main_peak = float(np.max(envelope[main]))
    image_peak = float(np.max(envelope[image]))
    return float(20.0 * np.log10(max(image_peak, _EPSILON) / max(main_peak, _EPSILON)))
