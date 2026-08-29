"""Tests for worst-case per-probe gate evaluation."""

import numpy as np
import pytest
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.reference import upsample_bessel_reference
from totton_audio_de_mirroring.evaluation.gates import (
    ProbeEvaluation,
    Stage1GateConfig,
    evaluate_gates,
    evaluate_probe,
    render_markdown_report,
    report_to_dict,
)
from totton_audio_de_mirroring.evaluation.probe_suite import (
    TIER_CANONICAL,
    ProbeSpec,
    generate_probe,
)
from totton_audio_de_mirroring.models.proto_bank import (
    build_prototype_bank,
    upsample_with_kernel,
)

SOURCE_SR = 44_100
TARGET_SR = 88_200


def _square_evaluation(
    probe_id: str,
    rms_after: float,
    rms_before: float = 1.0e-4,
    amplitude: float = 0.5,
) -> ProbeEvaluation:
    spec = ProbeSpec(
        probe_id=probe_id,
        kind="square",
        tier=TIER_CANONICAL,
        frequency_hz=500.0,
        amplitude=amplitude,
    )
    return ProbeEvaluation(
        spec=spec,
        metrics={
            "plateau_rms_before": rms_before,
            "plateau_rms_after": rms_after,
            "plateau_p2p_before": rms_before,
            "plateau_p2p_after": rms_after,
            "overshoot_before": 0.05,
            "overshoot_after": 0.05,
            "image_rel_db": -90.0,
            "image_abs_db": -150.0,
            "image_before_abs_db": -150.0,
            "gain_error_db": 0.0,
            "gain_band_fraction": 1.0,
        },
    )


def test_absolute_floor_prevents_ratio_explosion() -> None:
    """A tiny ripple far below audibility must pass despite a huge ratio."""
    evaluation = _square_evaluation("sq", rms_after=4.0e-4, rms_before=1.0e-6)
    report = evaluate_gates([evaluation])
    gate = next(g for g in report.gates if g.gate_id == "G1_lf_ringing")
    assert gate.passed


def test_catastrophic_probe_fails_worst_case() -> None:
    """One 15.8x-style probe among benign ones must fail the gate."""
    benign = [_square_evaluation(f"sq{i}", rms_after=1.0e-4) for i in range(5)]
    bad = _square_evaluation("sq_bad", rms_after=1.6e-3)
    report = evaluate_gates(benign + [bad])
    gate = next(g for g in report.gates if g.gate_id == "G1_lf_ringing")
    assert not gate.passed
    assert gate.worst_probe_id == "sq_bad"
    assert not report.all_passed


def test_relative_binding_when_reference_is_large() -> None:
    evaluation = _square_evaluation("sq", rms_after=1.05e-2, rms_before=1.0e-2)
    report = evaluate_gates([evaluation])
    gate = next(g for g in report.gates if g.gate_id == "G1_lf_ringing")
    row = next(r for r in gate.rows if r.metric == "plateau_rms_after")
    assert row.binding == "relative"
    assert row.passed


def test_empty_evaluations_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate_gates([])


def test_report_serialization_roundtrip() -> None:
    report = evaluate_gates([_square_evaluation("sq", rms_after=1.0e-4)])
    payload = report_to_dict(report)
    assert payload["all_passed"] is True
    assert payload["manifest_hash"] == ""
    markdown = render_markdown_report(report)
    assert "G1_lf_ringing" in markdown


def test_config_override_tightens_gate() -> None:
    evaluation = _square_evaluation("sq", rms_after=4.0e-4)
    strict = Stage1GateConfig(plateau_rms_floor_rel=1.0e-5)
    report = evaluate_gates([evaluation], config=strict)
    gate = next(g for g in report.gates if g.gate_id == "G1_lf_ringing")
    assert not gate.passed


def test_sweep_peak_image_metric_is_worst_case_binding() -> None:
    evaluation = ProbeEvaluation(
        spec=ProbeSpec(
            probe_id="sweep",
            kind="sweep_log",
            tier=TIER_CANONICAL,
        ),
        metrics={
            "image_rel_db": -100.0,
            "image_peak_rel_db": -45.0,
            "image_abs_db": -150.0,
            "image_before_abs_db": -40.0,
            "gain_error_db": 0.0,
            "gain_band_fraction": 1.0,
        },
    )
    report = evaluate_gates([evaluation])
    gate = next(g for g in report.gates if g.gate_id == "G3_mirror")
    peak_row = next(r for r in gate.rows if r.metric == "image_peak_rel_db")
    assert not peak_row.passed
    assert not gate.passed


@pytest.mark.slow
def test_gentle_prototype_passes_lf_ringing_end_to_end() -> None:
    """Integration: the gentle prototype passes G1 on a real 500 Hz square."""
    spec = ProbeSpec(
        probe_id="square_500hz",
        kind="square",
        tier=TIER_CANONICAL,
        frequency_hz=500.0,
    )
    source = generate_probe(spec, SOURCE_SR)
    bessel_ref = upsample_bessel_reference(
        signal=source,
        source_sr=SOURCE_SR,
        target_sr=TARGET_SR,
        cutoff_hz=20_000.0,
        order=6,
    )
    ideal_ref = np.asarray(sp_signal.resample_poly(source, 2, 1), dtype=np.float64)
    bank = build_prototype_bank()
    output = upsample_with_kernel(
        source, bank.kernels[bank.names.index("gentle")], bank.upsample_ratio
    )
    evaluation = evaluate_probe(
        spec=spec,
        source=source,
        source_sample_rate=SOURCE_SR,
        bessel_reference=bessel_ref,
        ideal_reference=ideal_ref,
        output=output,
        target_sample_rate=TARGET_SR,
    )
    report = evaluate_gates([evaluation])
    gate = next(g for g in report.gates if g.gate_id == "G1_lf_ringing")
    assert gate.passed


def test_image_band_low_hz_by_rate_family() -> None:
    """Image band starts 500 Hz above the input Nyquist (target rate / 4)."""
    from totton_audio_de_mirroring.evaluation.gates import image_band_low_hz

    assert image_band_low_hz(88_200) == 22_550.0
    assert image_band_low_hz(96_000) == 24_500.0
    with pytest.raises(ValueError, match="positive"):
        image_band_low_hz(0)


def test_modulation_sideband_gate_binds_on_worst_two_tone() -> None:
    base_metrics = {
        "image_rel_db": -100.0,
        "image_abs_db": -150.0,
        "image_before_abs_db": -150.0,
        "gain_error_db": 0.0,
        "gain_band_fraction": 1.0,
    }
    evaluations = []
    for probe_id, tier, value in (
        ("canonical", "canonical", -120.0),
        ("held", "held_out", -95.0),
    ):
        spec = ProbeSpec(
            probe_id=probe_id,
            kind="imd_two_tone",
            tier=tier,
            frequency_hz=60.0,
            secondary_frequency_hz=7_000.0,
            amplitude_ratio=4.0,
            duration_sec=3.0,
        )
        evaluations.append(
            ProbeEvaluation(
                spec=spec,
                metrics={**base_metrics, "modulation_sideband_db": value},
            )
        )
    gate = next(
        result
        for result in evaluate_gates(evaluations).gates
        if result.gate_id == "G9_no_modulation_sidebands"
    )
    assert not gate.passed
    assert gate.worst_probe_id == "held"
