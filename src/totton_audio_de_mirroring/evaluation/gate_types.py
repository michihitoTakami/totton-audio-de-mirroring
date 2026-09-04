"""Typed records for Stage 1 probe gating.

Physical Basis:
    Separating metric computation from gating keeps a single metric source of
    truth that several gates and both report serializers can consume, and it
    lets the thresholds travel with the report that was earned against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from totton_audio_de_mirroring.evaluation.probe_suite import ProbeSpec

# Spec 6 measured square/DC-step ringing on 8x sinc-oversampled signals; spec 7
# bounds the plateau window by the probe's own half period. See gates.py for
# the full rationale of each bump.
GATE_SPEC_VERSION = 7


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
        post_echo_energy_ratio_max: Allowed post-echo energy growth.
        post_echo_floor_rel: Amplitude-relative post-echo absolute floor.
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
    post_echo_energy_ratio_max: float = 1.44
    post_echo_floor_rel: float = 1.0e-3
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
    plateau_window_ms: tuple[float, float] | None = None


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
class SkippedProbe:
    """A probe that carries no gate row, with the reason it carries none.

    Physical Basis:
        An absent row must never read as a passing row. Recording the probe
        alongside its half period and the window that could not be resolved
        keeps the omission auditable in the report.
    """

    probe_id: str
    tier: str
    metric_group: str
    reason: str
    half_period_ms: float | None


@dataclass(frozen=True)
class GateResult:
    """Result of one gate over all applicable probes."""

    gate_id: str
    passed: bool
    worst_probe_id: str | None
    rows: tuple[GateRow, ...]
    skipped: tuple[SkippedProbe, ...] = ()


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
