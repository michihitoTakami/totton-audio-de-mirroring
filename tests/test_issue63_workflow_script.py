"""Tests for Issue #63 Stage 1 workflow script helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.run_issue63_stage1_workflow import (
    CandidateEvaluation,
    GateConfig,
    _passes_hard_gate,
    _passes_imd_gate,
    _select_best_candidate,
)


def _gate_config() -> GateConfig:
    return GateConfig(
        max_lb_phase_error_deg=15.0,
        max_lb_group_delay_error_samples=600.0,
        max_lb_amplitude_error_db=-20.0,
        require_zero_energy_cap_violations=True,
        require_positive_thdn_improvement=True,
    )


def _candidate(
    *,
    name: str,
    score: float,
    pass_hard: bool = True,
    pass_imd: bool = True,
    thdn: float = 1.0,
    symmetry: float = 0.8,
    touch: float = 0.4,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        checkpoint_path=Path(f"/tmp/{name}.pt"),
        output_dir=Path(f"/tmp/{name}"),
        hard_summary={
            "hb_energy_cap_violation_rate": 0.0,
            "lb_phase_error_deg": 5.0,
            "lb_group_delay_error_samples": 120.0,
            "lb_amplitude_error_db": -35.0,
            "touch_metric": touch,
        },
        mirror_summary={"symmetry_reduction_ratio": symmetry},
        imd_summary={
            "mean_thdn_improvement_db": thdn,
            "all_nmse_has_lower_imd": True,
        },
        passes_hard_gate=pass_hard,
        passes_imd_gate=pass_imd,
        composite_score=score,
    )


def test_passes_hard_gate_rejects_cap_violation() -> None:
    gate = _gate_config()
    hard_summary = {
        "hb_energy_cap_violation_rate": 0.1,
        "lb_phase_error_deg": 1.0,
        "lb_group_delay_error_samples": 10.0,
        "lb_amplitude_error_db": -40.0,
    }
    assert not _passes_hard_gate(hard_summary=hard_summary, gate_config=gate)


def test_passes_hard_gate_accepts_within_thresholds() -> None:
    gate = _gate_config()
    hard_summary = {
        "hb_energy_cap_violation_rate": 0.0,
        "lb_phase_error_deg": 10.0,
        "lb_group_delay_error_samples": 250.0,
        "lb_amplitude_error_db": -30.0,
    }
    assert _passes_hard_gate(hard_summary=hard_summary, gate_config=gate)


def test_passes_imd_gate_requires_positive_improvement_when_strict() -> None:
    gate = _gate_config()
    imd_summary = {
        "mean_thdn_improvement_db": 0.0,
        "all_nmse_has_lower_imd": True,
    }
    assert not _passes_imd_gate(imd_summary=imd_summary, gate_config=gate)


def test_select_best_candidate_uses_highest_score_among_passing() -> None:
    c1 = _candidate(name="a", score=1.0)
    c2 = _candidate(name="b", score=2.0)
    c3 = _candidate(name="c", score=3.0, pass_hard=False)

    selected = _select_best_candidate([c1, c2, c3])
    assert selected.checkpoint_path.name == "b.pt"


def test_select_best_candidate_raises_when_nothing_passes() -> None:
    c1 = _candidate(name="a", score=1.0, pass_hard=False)
    c2 = _candidate(name="b", score=2.0, pass_imd=False)

    with pytest.raises(RuntimeError, match="No checkpoint passed"):
        _ = _select_best_candidate([c1, c2])
