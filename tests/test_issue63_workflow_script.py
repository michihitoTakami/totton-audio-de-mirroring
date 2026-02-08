"""Tests for Issue #63 Stage 1 workflow script helpers."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.run_issue63_stage1_workflow import (
    CandidateEvaluation,
    GateConfig,
    _build_gate_details,
    _evaluate_square_probe_ringing,
    _generate_square_probe_signal,
    _load_ringing_summary,
    _passes_hard_gate,
    _passes_imd_gate,
    _passes_mirror_gate,
    _passes_ringing_gate,
    _run_command_with_live_log,
    _select_best_candidate,
    _summarize_square_probe_ringing,
)


def _gate_config() -> GateConfig:
    return GateConfig(
        max_lb_phase_error_deg=15.0,
        max_lb_group_delay_error_samples=600.0,
        max_lb_amplitude_error_db=-20.0,
        require_zero_energy_cap_violations=True,
        min_mirror_symmetry_reduction_ratio=0.70,
        require_positive_thdn_improvement=True,
        max_plateau_ripple_rms_ratio=1.10,
        max_plateau_ripple_p2p_ratio=1.10,
        max_overshoot_abs_increase=0.005,
        require_nonpositive_ringing_ratio_delta=True,
    )


def _candidate(
    *,
    name: str,
    score: float,
    pass_hard: bool = True,
    pass_mirror: bool = True,
    pass_imd: bool = True,
    pass_ringing: bool = True,
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
        ringing_summary={
            "mean_plateau_ripple_rms_ratio": 1.0,
            "mean_plateau_ripple_p2p_ratio": 1.0,
            "mean_overshoot_abs_delta": 0.0,
            "mean_ringing_ratio_delta": 0.0,
        },
        gate_details={},
        passes_hard_gate=pass_hard,
        passes_mirror_gate=pass_mirror,
        passes_imd_gate=pass_imd,
        passes_ringing_gate=pass_ringing,
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


def test_passes_mirror_gate_requires_symmetry_threshold() -> None:
    gate = _gate_config()
    mirror_summary = {"symmetry_reduction_ratio": 0.69}
    assert not _passes_mirror_gate(mirror_summary=mirror_summary, gate_config=gate)


def test_passes_mirror_gate_accepts_on_threshold() -> None:
    gate = _gate_config()
    mirror_summary = {"symmetry_reduction_ratio": 0.70}
    assert _passes_mirror_gate(mirror_summary=mirror_summary, gate_config=gate)


def test_passes_ringing_gate_rejects_ripple_regression() -> None:
    gate = _gate_config()
    ringing_summary = {
        "mean_plateau_ripple_rms_ratio": 1.20,
        "mean_plateau_ripple_p2p_ratio": 1.05,
        "mean_overshoot_abs_delta": 0.0,
        "mean_ringing_ratio_delta": 0.0,
    }
    assert not _passes_ringing_gate(ringing_summary=ringing_summary, gate_config=gate)


def test_passes_ringing_gate_accepts_within_thresholds() -> None:
    gate = _gate_config()
    ringing_summary = {
        "mean_plateau_ripple_rms_ratio": 1.05,
        "mean_plateau_ripple_p2p_ratio": 1.03,
        "mean_overshoot_abs_delta": 0.001,
        "mean_ringing_ratio_delta": 0.0,
    }
    assert _passes_ringing_gate(ringing_summary=ringing_summary, gate_config=gate)


def test_passes_ringing_gate_accepts_on_threshold() -> None:
    gate = _gate_config()
    ringing_summary = {
        "mean_plateau_ripple_rms_ratio": 1.10,
        "mean_plateau_ripple_p2p_ratio": 1.10,
        "mean_overshoot_abs_delta": 0.005,
        "mean_ringing_ratio_delta": 0.0,
    }
    assert _passes_ringing_gate(ringing_summary=ringing_summary, gate_config=gate)


def test_passes_ringing_gate_rejects_positive_ratio_delta_when_strict() -> None:
    gate = _gate_config()
    ringing_summary = {
        "mean_plateau_ripple_rms_ratio": 1.0,
        "mean_plateau_ripple_p2p_ratio": 1.0,
        "mean_overshoot_abs_delta": 0.0,
        "mean_ringing_ratio_delta": 1.0e-6,
    }
    assert not _passes_ringing_gate(ringing_summary=ringing_summary, gate_config=gate)


def test_select_best_candidate_uses_highest_score_among_passing() -> None:
    c1 = _candidate(name="a", score=1.0)
    c2 = _candidate(name="b", score=2.0)
    c3 = _candidate(name="c", score=3.0, pass_hard=False)

    selected = _select_best_candidate([c1, c2, c3])
    assert selected.checkpoint_path.name == "b.pt"


def test_select_best_candidate_raises_when_nothing_passes() -> None:
    c1 = _candidate(name="a", score=1.0, pass_hard=False)
    c2 = _candidate(name="b", score=2.0, pass_imd=False, pass_ringing=False)

    with pytest.raises(RuntimeError, match="No checkpoint passed"):
        _ = _select_best_candidate([c1, c2])


def test_select_best_candidate_rejects_mirror_gate_failure() -> None:
    c1 = _candidate(name="a", score=1.0, pass_mirror=False)
    c2 = _candidate(name="b", score=2.0, pass_hard=False)

    with pytest.raises(RuntimeError, match="No checkpoint passed"):
        _ = _select_best_candidate([c1, c2])


def test_run_command_with_live_log_streams_output(tmp_path: Path) -> None:
    log_path = tmp_path / "stream.log"
    command = [
        sys.executable,
        "-c",
        (
            "import sys,time;"
            "print('line1', flush=True);"
            "time.sleep(0.01);"
            "print('line2', flush=True)"
        ),
    ]

    exit_code = _run_command_with_live_log(
        command,
        log_path=log_path,
        section_label="test",
    )

    assert exit_code == 0
    text = log_path.read_text(encoding="utf-8")
    assert "line1" in text
    assert "line2" in text
    assert "exit_code=0" in text


def test_run_command_with_live_log_updates_during_execution(tmp_path: Path) -> None:
    log_path = tmp_path / "stream_live.log"
    command = [
        sys.executable,
        "-c",
        (
            "import time;"
            "print('live-start', flush=True);"
            "time.sleep(0.3);"
            "print('live-end', flush=True)"
        ),
    ]

    result: dict[str, int] = {}

    def _runner() -> None:
        result["exit_code"] = _run_command_with_live_log(
            command,
            log_path=log_path,
            section_label="live-test",
        )

    thread = threading.Thread(target=_runner)
    thread.start()

    found_live_start = False
    for _ in range(20):
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8")
            if "live-start" in text:
                found_live_start = True
                break
        time.sleep(0.05)

    thread.join(timeout=2.0)
    assert found_live_start
    assert result["exit_code"] == 0
    final_text = log_path.read_text(encoding="utf-8")
    assert "live-end" in final_text


def test_generate_square_probe_signal_has_expected_amplitude() -> None:
    signal = _generate_square_probe_signal(
        sample_rate=44_100,
        frequency_hz=1_000.0,
        duration_sec=0.01,
        amplitude=0.5,
    )
    assert np.max(signal) == pytest.approx(0.5, abs=1e-6)
    assert np.min(signal) == pytest.approx(-0.5, abs=1e-6)


def test_summarize_square_probe_ringing_computes_means() -> None:
    summary = _summarize_square_probe_ringing(
        [
            {
                "plateau_ripple_rms_before": 1.0,
                "plateau_ripple_rms_after": 1.2,
                "plateau_ripple_rms_ratio": 1.2,
                "plateau_ripple_p2p_before": 2.0,
                "plateau_ripple_p2p_after": 2.4,
                "plateau_ripple_p2p_ratio": 1.2,
                "overshoot_abs_before": 0.1,
                "overshoot_abs_after": 0.11,
                "overshoot_abs_delta": 0.01,
                "ringing_ratio_before": 1.0,
                "ringing_ratio_after": 1.1,
                "ringing_ratio_delta": 0.1,
            },
            {
                "plateau_ripple_rms_before": 2.0,
                "plateau_ripple_rms_after": 2.2,
                "plateau_ripple_rms_ratio": 1.1,
                "plateau_ripple_p2p_before": 3.0,
                "plateau_ripple_p2p_after": 3.3,
                "plateau_ripple_p2p_ratio": 1.1,
                "overshoot_abs_before": 0.2,
                "overshoot_abs_after": 0.205,
                "overshoot_abs_delta": 0.005,
                "ringing_ratio_before": 1.2,
                "ringing_ratio_after": 1.25,
                "ringing_ratio_delta": 0.05,
            },
        ]
    )
    assert summary["num_samples"] == 2
    assert summary["mean_plateau_ripple_rms_ratio"] == pytest.approx(1.15)
    assert summary["mean_overshoot_abs_delta"] == pytest.approx(0.0075)


def test_load_ringing_summary_prefers_ringing_metrics_over_top_summary() -> None:
    payload = {
        "summary": {"lb_phase_error_deg": 1.0},
        "ringing_metrics": {
            "summary": {
                "mean_plateau_ripple_rms_ratio": 1.02,
                "mean_plateau_ripple_p2p_ratio": 1.01,
                "mean_overshoot_abs_delta": 0.0,
                "mean_ringing_ratio_delta": -0.1,
            }
        },
    }
    summary = _load_ringing_summary(payload)
    assert summary["mean_plateau_ripple_rms_ratio"] == pytest.approx(1.02)


def test_build_gate_details_contains_traceable_thresholds() -> None:
    gate = _gate_config()
    details = _build_gate_details(
        hard_summary={
            "hb_energy_cap_violation_rate": 0.0,
            "lb_phase_error_deg": 5.0,
            "lb_group_delay_error_samples": 100.0,
            "lb_amplitude_error_db": -30.0,
        },
        mirror_summary={"symmetry_reduction_ratio": 0.75},
        imd_summary={
            "all_nmse_has_lower_imd": True,
            "mean_thdn_improvement_db": 0.2,
        },
        ringing_summary={
            "mean_plateau_ripple_rms_ratio": 1.0,
            "mean_plateau_ripple_p2p_ratio": 1.0,
            "mean_overshoot_abs_delta": 0.0,
            "mean_ringing_ratio_delta": 0.0,
        },
        gate_config=gate,
    )
    assert details["mirror_gate"]["passed"] is True
    assert "min_symmetry_reduction_ratio" in details["mirror_gate"]["threshold"]
    assert details["ringing_gate"]["observed"][
        "mean_plateau_ripple_rms_ratio"
    ] == pytest.approx(1.0)


def test_evaluate_square_probe_ringing_writes_json_and_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _IdentityModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return x

    def _mock_build_model(
        checkpoint_path: Path,
        data_config_path: Path,
        device: str,
    ) -> torch.nn.Module:
        _ = checkpoint_path
        _ = data_config_path
        return _IdentityModel().to(torch.device(device))

    monkeypatch.setattr(
        "scripts.run_issue63_stage1_workflow._build_stage1_model_from_checkpoint",
        _mock_build_model,
    )

    payload = _evaluate_square_probe_ringing(
        checkpoint_path=tmp_path / "dummy.pt",
        data_config_path=tmp_path / "dummy.yaml",
        device="cpu",
        report_dir=tmp_path / "ringing",
        source_sample_rate=44_100,
        target_sample_rate=88_200,
        frequencies_hz=(500.0, 1_000.0),
        duration_sec=0.05,
        amplitude=0.5,
        plateau_start_ms=0.1,
        plateau_end_ms=0.8,
        ringing_window_ms=0.8,
    )

    assert payload["summary"]["num_samples"] == 2
    assert len(payload["samples"]) == 2
    assert (tmp_path / "ringing" / "ringing_square_metrics.json").exists()
    assert (tmp_path / "ringing" / "ringing_square_metrics.csv").exists()
