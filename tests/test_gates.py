"""Tests for worst-case per-probe gate evaluation."""

import numpy as np
import pytest
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.reference import upsample_bessel_reference
from totton_audio_de_mirroring.evaluation.gates import (
    LF_RINGING_MAX_FREQ_HZ,
    PLATEAU_EDGE_GUARD_MS,
    PLATEAU_END_MS_DEFAULT,
    PLATEAU_MAX_FREQ_HZ,
    PLATEAU_MIN_HALF_PERIOD_MS,
    PLATEAU_MIN_SPAN_MS,
    PLATEAU_START_MS,
    ProbeEvaluation,
    Stage1GateConfig,
    evaluate_gates,
    evaluate_probe,
    probe_half_period_ms,
    render_markdown_report,
    report_to_dict,
    resolve_plateau_window,
)
from totton_audio_de_mirroring.evaluation.probe_suite import (
    KIND_DC_STEP,
    KIND_SQUARE,
    TIER_CANONICAL,
    ProbeSpec,
    build_default_probe_suite,
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


def test_post_echo_gate_binds_against_bessel_reference() -> None:
    evaluation = ProbeEvaluation(
        spec=ProbeSpec(
            probe_id="impulse",
            kind="impulse",
            tier=TIER_CANONICAL,
            amplitude=0.5,
        ),
        metrics={
            "pre_echo_energy_before": 1.0e-6,
            "pre_echo_energy_after": 1.0e-6,
            "post_echo_energy_before": 1.0e-4,
            "post_echo_energy_after": 2.0e-4,
            "image_rel_db": -100.0,
            "image_abs_db": -150.0,
            "image_before_abs_db": -150.0,
            "gain_error_db": 0.0,
            "gain_band_fraction": 1.0,
        },
    )

    gate = next(
        result
        for result in evaluate_gates([evaluation]).gates
        if result.gate_id == "G2c_post_echo"
    )

    assert not gate.passed
    assert gate.worst_probe_id == "impulse"


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


def test_oversampled_edge_metrics_are_insensitive_to_half_sample_edge_phase() -> None:
    """Spec 6: shifting a square edge by half an output sample must not move the metric."""
    from scipy import signal as sp_signal

    from totton_audio_de_mirroring.evaluation.time_domain_visualization import (
        compare_edge_aligned_ringing,
    )

    rate = 96_000
    duration = int(0.05 * rate)
    time = np.arange(duration) / rate
    # Band-limited step via a gentle Butterworth so the edge has a finite slope.
    sos = sp_signal.butter(4, 20_000.0, fs=rate, output="sos")

    def step(offset_samples: float) -> np.ndarray:
        raw = np.where(time * rate >= duration / 2 + offset_samples, 1.0, -1.0)
        return np.asarray(sp_signal.sosfiltfilt(sos, raw), dtype=np.float64)

    reference = step(0.0)
    native = [
        compare_edge_aligned_ringing(reference, step(shift), rate, oversample=1)
        for shift in (0.0, 0.5)
    ]
    oversampled = [
        compare_edge_aligned_ringing(reference, step(shift), rate, oversample=8)
        for shift in (0.0, 0.5)
    ]
    native_spread = abs(
        native[0].after.plateau_ripple_rms - native[1].after.plateau_ripple_rms
    )
    oversampled_spread = abs(
        oversampled[0].after.plateau_ripple_rms
        - oversampled[1].after.plateau_ripple_rms
    )
    assert oversampled_spread <= native_spread
    assert oversampled[0].after.edge_time_ms == pytest.approx(
        native[0].after.edge_time_ms, abs=1.0 / rate * 1000.0
    )
    with pytest.raises(ValueError, match="oversample"):
        compare_edge_aligned_ringing(reference, reference, rate, oversample=0)


# --- Spec 7: probe-frequency-aware plateau window -------------------------

_EXPECTED_PLATEAU_WINDOWS: dict[str, tuple[float, float] | None] = {
    "square_50hz": (0.1, 0.8),
    "square_73hz_held": (0.1, 0.8),
    "square_100hz": (0.1, 0.8),
    "square_331hz_held": (0.1, 0.8),
    "square_500hz": (0.1, 0.8),
    "square_500hz_a005": (0.1, 0.8),
    "dc_step_up": (0.1, 0.8),
    "dc_step_down": (0.1, 0.8),
    "square_1000hz": (0.1, 0.4),
    "square_1730hz_held": None,
    "square_2000hz": None,
    "square_4400hz_held": None,
    "square_5000hz": None,
}


def _edge_probe_specs() -> list[ProbeSpec]:
    return [
        spec
        for spec in build_default_probe_suite()
        if spec.kind in {KIND_SQUARE, KIND_DC_STEP}
    ]


def test_plateau_window_constants_are_self_consistent() -> None:
    """The G1/G2 split and the no-plateau ceiling are derived, not chosen."""
    assert (
        pytest.approx(PLATEAU_START_MS + PLATEAU_MIN_SPAN_MS + PLATEAU_EDGE_GUARD_MS)
        == PLATEAU_MIN_HALF_PERIOD_MS
    )
    assert pytest.approx(500.0 / PLATEAU_MIN_HALF_PERIOD_MS) == PLATEAU_MAX_FREQ_HZ
    assert (
        pytest.approx(500.0 / (PLATEAU_END_MS_DEFAULT + PLATEAU_EDGE_GUARD_MS))
        == LF_RINGING_MAX_FREQ_HZ
    )


def test_derived_plateau_window_table_is_frozen() -> None:
    """Every square/step probe resolves to the recorded spec 7 window."""
    resolved = {
        spec.probe_id: resolve_plateau_window(spec) for spec in _edge_probe_specs()
    }
    assert set(resolved) == set(_EXPECTED_PLATEAU_WINDOWS)
    for probe_id, expected in _EXPECTED_PLATEAU_WINDOWS.items():
        actual = resolved[probe_id]
        if expected is None:
            assert actual is None, probe_id
            continue
        assert actual is not None, probe_id
        assert actual[0] == pytest.approx(expected[0]), probe_id
        assert actual[1] == pytest.approx(expected[1]), probe_id


def test_derived_plateau_window_never_spans_a_transition() -> None:
    """A resolved window stays one guard clear of the next edge.

    The spec 6 fixed 0.1-0.8 ms window violates this for every probe at or
    above 1000 Hz, which is the defect spec 7 removes.
    """
    for spec in _edge_probe_specs():
        window = resolve_plateau_window(spec)
        if window is None:
            continue
        assert window[1] - window[0] >= PLATEAU_MIN_SPAN_MS - 1e-12, spec.probe_id
        half_period = probe_half_period_ms(spec)
        if half_period is None:
            continue
        assert window[1] + PLATEAU_EDGE_GUARD_MS <= half_period + 1e-12, spec.probe_id


@pytest.mark.parametrize(
    ("frequency_hz", "resolvable"),
    [(1600.0, True), (1700.0, False)],
)
def test_plateau_window_resolvability_boundary(
    frequency_hz: float, resolvable: bool
) -> None:
    """The minimum-span rule decides resolvability either side of the ceiling."""
    spec = ProbeSpec(
        probe_id=f"square_{int(frequency_hz)}hz",
        kind=KIND_SQUARE,
        tier=TIER_CANONICAL,
        frequency_hz=frequency_hz,
        amplitude=0.5,
    )
    assert (resolve_plateau_window(spec) is not None) is resolvable


def _prototype_evaluation(spec: ProbeSpec, prototype: str) -> ProbeEvaluation:
    """Evaluate one fixed prototype on one probe through the real metric path."""
    bank = build_prototype_bank()
    kernel = bank.kernels[bank.names.index(prototype)]
    source = generate_probe(spec, SOURCE_SR)
    bessel = upsample_bessel_reference(
        signal=source,
        source_sr=SOURCE_SR,
        target_sr=TARGET_SR,
        cutoff_hz=20_000.0,
        order=6,
    )
    ideal = np.asarray(sp_signal.resample_poly(source, 2, 1), dtype=np.float64)
    return evaluate_probe(
        spec=spec,
        source=source,
        source_sample_rate=SOURCE_SR,
        bessel_reference=bessel,
        ideal_reference=ideal,
        output=upsample_with_kernel(source, kernel, bank.upsample_ratio),
        target_sample_rate=TARGET_SR,
    )


def test_high_frequency_square_emits_no_plateau_rows() -> None:
    """A 5 kHz square has no settled plateau, so it carries no plateau row.

    Exercises the real metric path rather than a fabricated metric mapping.
    """
    spec = next(
        item for item in _edge_probe_specs() if item.probe_id == "square_5000hz"
    )
    evaluation = _prototype_evaluation(spec, "gentle")
    assert evaluation.plateau_window_ms is None
    for metric in ("plateau_rms_after", "plateau_p2p_after", "overshoot_after"):
        assert metric not in evaluation.metrics

    report = evaluate_gates([evaluation])
    ringing = [gate for gate in report.gates if gate.gate_id.startswith(("G1_", "G2_"))]
    skipped = [item for gate in ringing for item in gate.skipped]
    assert [item.probe_id for item in skipped] == ["square_5000hz"]
    assert skipped[0].metric_group == "plateau_ripple_and_overshoot"
    assert skipped[0].half_period_ms == pytest.approx(0.1)
    assert not any(gate.rows for gate in ringing)


def test_spec7_plateau_metric_discriminates_ringing_at_1khz() -> None:
    """The 1 kHz row separates the prototypes instead of measuring the square.

    Under the spec 6 fixed window the same row scored sharp below gentle
    (0.98 vs 1.00), so this assertion fails on a revert.
    """
    spec = next(
        item for item in _edge_probe_specs() if item.probe_id == "square_1000hz"
    )
    ratios = {}
    for prototype in ("gentle", "sharp"):
        metrics = _prototype_evaluation(spec, prototype).metrics
        ratios[prototype] = metrics["plateau_rms_after"] / metrics["plateau_rms_before"]
    assert ratios["gentle"] < 1.0
    assert ratios["sharp"] > 10.0
    assert ratios["sharp"] > ratios["gentle"]


def test_skipped_probes_are_never_reported_as_passing() -> None:
    """A skipped probe must not appear on a PASS line in either report form."""
    evaluations = [
        _prototype_evaluation(spec, "gentle")
        for spec in _edge_probe_specs()
        if spec.probe_id in {"square_500hz", "square_5000hz"}
    ]
    report = evaluate_gates(evaluations)
    markdown = render_markdown_report(report)
    assert "NOT a pass" in markdown
    # Scope the check to the ringing sections: square_5000hz legitimately
    # appears as a passing G5_gain row, which spec 7 does not touch.
    section = ""
    for line in markdown.splitlines():
        if line.startswith("## "):
            section = line.removeprefix("## ").split(":", 1)[0]
        if section.startswith(("G1_", "G2_")) and "square_5000hz" in line:
            assert "PASS" not in line

    payload = report_to_dict(report)
    skipped = [
        item["probe_id"] for gate in payload["gates"] for item in gate["skipped"]
    ]
    assert skipped == ["square_5000hz"]


def test_no_ringing_gate_is_silently_empty() -> None:
    """Both ringing gates keep at least one measured row on the frozen suite."""
    evaluations = [
        _prototype_evaluation(spec, "gentle") for spec in _edge_probe_specs()
    ]
    report = evaluate_gates(evaluations)
    for gate in report.gates:
        if not gate.gate_id.startswith(("G1_", "G2_")):
            continue
        assert gate.rows, f"{gate.gate_id} has no measured row"


def test_worst_probe_id_is_independent_of_row_order() -> None:
    """Tied ratios must not let suite ordering decide the reported worst probe."""
    forward = [
        _square_evaluation("square_a", 1.0e-4),
        _square_evaluation("square_b", 1.0e-4),
    ]
    reverse = list(reversed(forward))
    worst = [
        next(
            gate.worst_probe_id
            for gate in evaluate_gates(order).gates
            if gate.gate_id.startswith("G1_")
        )
        for order in (forward, reverse)
    ]
    assert worst[0] == worst[1]
