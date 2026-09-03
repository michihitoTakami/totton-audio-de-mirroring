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

from dataclasses import asdict

import numpy as np

from totton_audio_de_mirroring.evaluation.distortion import smpte_imd_db
from totton_audio_de_mirroring.evaluation.gate_reporting import (
    render_markdown_report as render_markdown_report,
)
from totton_audio_de_mirroring.evaluation.gate_reporting import (
    report_to_dict as report_to_dict,
)
from totton_audio_de_mirroring.evaluation.gate_spectral import (
    GAIN_APPLICABILITY_HIGH_HZ,
    GAIN_BAND_HIGH_HZ,
    _band_energy_fraction,
    _band_level_db,
    _image_minus_main_db,
    _peak_image_minus_main_db,
    image_band_low_hz,
)
from totton_audio_de_mirroring.evaluation.gate_types import (
    GateReport,
    GateResult,
    GateRow,
    ProbeEvaluation,
    SkippedProbe,
    Stage1GateConfig,
)
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

# Spec 6: square/DC-step ringing metrics are measured on 8x sinc-oversampled
# signals so edge alignment and plateau windows are sub-sample. Thresholds and
# their Bessel-relative definitions are unchanged; the change removes the
# output-grid phase dependence that made 48 kHz square rows (edges exactly
# between 96 kHz samples) jump by a whole sample and gave the two rate
# families different effective thresholds for the same filter behavior.
#
# Spec 7: the plateau window is bounded by the probe's own half period, and a
# probe whose half period cannot hold a settled plateau emits no plateau or
# overshoot row at all. Spec 6 used a fixed 0.1-0.8 ms window for every square,
# so from 625 Hz upward the window straddled the next transition and the
# "plateau ripple" was the square wave itself: at 2 kHz three prototypes whose
# post-edge ringing differs by 48x collapsed into a 3% spread against a 10%
# allowance, and at 5 kHz the ordering inverted so a flatter FIR scored worse.
# Thresholds, their Bessel-relative definitions and the probe manifest are
# unchanged; only which rows are emitted, over which window, changes.
SQUARE_EDGE_OVERSAMPLE = 8
_EPSILON = 1e-300

# Plateau window geometry, in milliseconds after the detected edge. The start
# is a filter-settling allowance and does not scale with the probe period. The
# guard keeps the window clear of the *next* transition by the same margin, and
# a window shorter than the minimum span cannot carry a meaningful ripple RMS.
PLATEAU_START_MS = 0.1
PLATEAU_END_MS_DEFAULT = 0.8
PLATEAU_EDGE_GUARD_MS = 0.1
PLATEAU_MIN_SPAN_MS = 0.1

# The G1/G2 split is derived from the window rule, not chosen: G1 holds probes
# whose plateau reaches the full settled span, G2 those whose plateau is cut
# short by the next transition. Above PLATEAU_MAX_FREQ_HZ no plateau exists at
# all and the probe carries no plateau row in either gate.
LF_RINGING_MAX_FREQ_HZ = 500.0 / (PLATEAU_END_MS_DEFAULT + PLATEAU_EDGE_GUARD_MS)
PLATEAU_MIN_HALF_PERIOD_MS = (
    PLATEAU_START_MS + PLATEAU_MIN_SPAN_MS + PLATEAU_EDGE_GUARD_MS
)
PLATEAU_MAX_FREQ_HZ = 500.0 / PLATEAU_MIN_HALF_PERIOD_MS
GAIN_MIN_BAND_FRACTION = 0.05
_ECHO_GUARD_MS = 0.5
_ECHO_WINDOW_MS = 3.5


def probe_half_period_ms(spec: ProbeSpec) -> float | None:
    """Return the probe's half period in milliseconds, or None if aperiodic.

    Args:
        spec: Probe specification.

    Returns:
        Half period in milliseconds, or None for a DC step or an unspecified
        frequency.

    Raises:
        ValueError: If the probe declares a non-positive frequency.

    Physical Basis:
        A square wave settles for at most one half period before the opposite
        transition arrives, so the half period is the hard ceiling on any
        post-edge measurement window.
    """
    if spec.kind == KIND_DC_STEP:
        return None
    frequency = spec.frequency_hz
    if frequency is None:
        return None
    if frequency <= 0.0:
        raise ValueError(f"Probe {spec.probe_id} declares frequency {frequency}.")
    return 1_000.0 / frequency / 2.0


def plateau_window_for_half_period(
    half_period_ms: float | None,
) -> tuple[float, float] | None:
    """Return the post-edge plateau window in ms, or None if unresolvable.

    Args:
        half_period_ms: Half period of the probe waveform, or None when the
            waveform has no following transition (a DC step).

    Returns:
        ``(start_ms, end_ms)`` measured from the detected edge, or None when
        the half period cannot hold a settled plateau.

    Raises:
        ValueError: If the half period is not positive.

    Physical Basis:
        The window must start after the interpolation filter has settled and
        end before the next transition, so it is clamped to
        ``half_period - guard``. When that leaves less than the minimum span,
        no ringing-free plateau exists at any window, and a ripple statistic
        computed there would describe the waveform rather than the ringing.
    """
    if half_period_ms is None:
        return (PLATEAU_START_MS, PLATEAU_END_MS_DEFAULT)
    if half_period_ms <= 0.0:
        raise ValueError(f"half_period_ms must be positive, got {half_period_ms}")
    end = min(PLATEAU_END_MS_DEFAULT, half_period_ms - PLATEAU_EDGE_GUARD_MS)
    if end - PLATEAU_START_MS < PLATEAU_MIN_SPAN_MS:
        return None
    return (PLATEAU_START_MS, end)


def plateau_window_for_frequency(
    frequency_hz: float | None,
) -> tuple[float, float] | None:
    """Return the plateau window for a square of the given frequency.

    Args:
        frequency_hz: Square-wave frequency in Hz, or None for a DC step.

    Returns:
        ``(start_ms, end_ms)``, or None when no settled plateau exists.

    Raises:
        ValueError: If the frequency is not positive.
    """
    if frequency_hz is None:
        return plateau_window_for_half_period(None)
    if frequency_hz <= 0.0:
        raise ValueError(f"frequency_hz must be positive, got {frequency_hz}")
    return plateau_window_for_half_period(500.0 / frequency_hz)


def resolve_plateau_window(spec: ProbeSpec) -> tuple[float, float] | None:
    """Return the plateau window for one probe, or None if unresolvable."""
    return plateau_window_for_half_period(probe_half_period_ms(spec))


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

    plateau_window: tuple[float, float] | None = None
    if spec.kind in {KIND_SQUARE, KIND_DC_STEP}:
        plateau_window = resolve_plateau_window(spec)
    if plateau_window is not None:
        comparison = compare_edge_aligned_ringing(
            before_signal=bessel_reference,
            after_signal=output,
            sample_rate=target_sample_rate,
            plateau_start_ms=plateau_window[0],
            plateau_end_ms=plateau_window[1],
            oversample=SQUARE_EDGE_OVERSAMPLE,
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
        onset = _event_center_index(spec, output.size, target_sample_rate)
        event_stop = _event_stop_index(spec, onset, target_sample_rate)
        pre_before, _ = _event_echo_energies(
            bessel_reference, onset, target_sample_rate
        )
        _, post_before = _event_echo_energies(
            bessel_reference, event_stop, target_sample_rate
        )
        pre_after, _ = _event_echo_energies(output, onset, target_sample_rate)
        _, post_after = _event_echo_energies(output, event_stop, target_sample_rate)
        metrics.update(
            {
                "pre_echo_energy_before": pre_before,
                "pre_echo_energy_after": pre_after,
                "post_echo_energy_before": post_before,
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
    return ProbeEvaluation(spec=spec, metrics=metrics, plateau_window_ms=plateau_window)


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
        _gate_post_echo(evaluations, cfg),
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


def _gate_ringing(
    gate_id: str,
    evaluations: list[ProbeEvaluation],
    cfg: Stage1GateConfig,
    low_frequency: bool,
) -> GateResult:
    rows: list[GateRow] = []
    skipped: list[SkippedProbe] = []
    for evaluation in evaluations:
        spec = evaluation.spec
        if spec.kind not in {KIND_SQUARE, KIND_DC_STEP}:
            continue
        freq = spec.frequency_hz or 0.0
        is_low = spec.kind == KIND_DC_STEP or freq <= LF_RINGING_MAX_FREQ_HZ
        if is_low != low_frequency:
            continue
        if "plateau_rms_after" not in evaluation.metrics:
            half_period = probe_half_period_ms(spec)
            skipped.append(
                SkippedProbe(
                    probe_id=spec.probe_id,
                    tier=spec.tier,
                    metric_group="plateau_ripple_and_overshoot",
                    reason=(
                        "no settled plateau exists: half period "
                        f"{half_period:.4g} ms leaves less than "
                        f"{PLATEAU_MIN_SPAN_MS:.4g} ms between the "
                        f"{PLATEAU_START_MS:.4g} ms settling start and the "
                        f"{PLATEAU_EDGE_GUARD_MS:.4g} ms next-edge guard"
                    )
                    if half_period is not None
                    else "plateau metrics unavailable",
                    half_period_ms=half_period,
                )
            )
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
    return _finalize(gate_id, rows, skipped)


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


def _gate_post_echo(
    evaluations: list[ProbeEvaluation], cfg: Stage1GateConfig
) -> GateResult:
    """Gate post-event energy against the same Bessel-relative policy as G2b."""
    rows: list[GateRow] = []
    for evaluation in evaluations:
        metrics = evaluation.metrics
        if "post_echo_energy_after" not in metrics:
            continue
        amp = evaluation.spec.amplitude
        rows += _threshold_rows(
            evaluation.spec,
            (
                (
                    "post_echo_energy_after",
                    metrics["post_echo_energy_after"],
                    cfg.post_echo_energy_ratio_max * metrics["post_echo_energy_before"],
                    (cfg.post_echo_floor_rel * amp) ** 2,
                ),
            ),
        )
    return _finalize("G2c_post_echo", rows)


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


def _worst_row_key(row: GateRow) -> tuple[float, str]:
    """Return the ordering key that selects a gate's worst row.

    Physical Basis:
        A gate binds on the largest fraction of its own threshold. Several
        probes routinely reach the same fraction (the low-frequency overshoot
        rows all sit on one absolute floor), so the probe id breaks the tie and
        keeps the reported worst probe independent of suite ordering.
    """
    return (row.value / max(row.threshold, 1e-12), row.probe_id)


def _finalize(
    gate_id: str,
    rows: list[GateRow],
    skipped: list[SkippedProbe] | None = None,
) -> GateResult:
    failing = [row for row in rows if not row.passed]
    worst: str | None = None
    ranked = failing or rows
    if ranked:
        # Ties are common on the low-frequency overshoot rows, so break them on
        # the probe id to keep the reported worst probe reproducible.
        worst = max(ranked, key=_worst_row_key).probe_id
    return GateResult(
        gate_id=gate_id,
        passed=not failing,
        worst_probe_id=worst,
        rows=tuple(rows),
        skipped=tuple(skipped or ()),
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


def _event_stop_index(spec: ProbeSpec, onset: int, sample_rate: int) -> int:
    """Return the exclusive event end used to start post-echo measurement."""
    if spec.kind == KIND_TONE_BURST and spec.burst_ms:
        burst_len = max(3, int(round(spec.burst_ms * sample_rate / 1_000.0)))
        return onset + burst_len
    return onset + 1


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
