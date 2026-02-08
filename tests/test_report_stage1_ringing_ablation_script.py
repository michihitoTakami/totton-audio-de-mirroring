"""Tests for Issue #81 ringing ablation report script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from scripts.report_stage1_ringing_ablation import main


def test_report_script_writes_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_json = tmp_path / "baseline.json"
    ringing_json = tmp_path / "ringing.json"
    output_md = tmp_path / "report.md"
    baseline_ckpt = tmp_path / "baseline.pt"
    ringing_ckpt = tmp_path / "ringing.pt"

    baseline_json.write_text(
        json.dumps(_make_eval_payload(symmetry=0.95, ringing_delta=0.02)),
        encoding="utf-8",
    )
    ringing_json.write_text(
        json.dumps(_make_eval_payload(symmetry=0.96, ringing_delta=-0.01)),
        encoding="utf-8",
    )
    torch.save({"train_history": [_make_history(0.01, 0.02)]}, baseline_ckpt)
    torch.save({"train_history": [_make_history(0.05, 0.06)]}, ringing_ckpt)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_stage1_ringing_ablation.py",
            "--baseline-eval-json",
            str(baseline_json),
            "--ringing-eval-json",
            str(ringing_json),
            "--baseline-checkpoint",
            str(baseline_ckpt),
            "--ringing-checkpoint",
            str(ringing_ckpt),
            "--output-md",
            str(output_md),
        ],
    )
    main()

    content = output_md.read_text(encoding="utf-8")
    assert "Issue #81 Stage1 Ringing-Loss Ablation Report" in content
    assert "Loss Contribution Comparison" in content
    assert "ringing improved: PASS" in content
    assert "mirror maintained: PASS" in content


def _make_eval_payload(symmetry: float, ringing_delta: float) -> dict[str, object]:
    return {
        "summary": {
            "hb_energy_cap_violation_rate": 0.0,
            "lb_phase_error_deg": 10.0,
            "lb_group_delay_error_samples": 200.0,
        },
        "mirror_metrics": {
            "summary": {
                "symmetry_reduction_ratio": symmetry,
            }
        },
        "ringing_metrics": {
            "summary": {
                "mean_ringing_ratio_delta": ringing_delta,
                "mean_overshoot_abs_delta": 0.001,
                "mean_plateau_ripple_rms_ratio": 1.02,
                "mean_plateau_ripple_p2p_ratio": 1.01,
            }
        },
    }


def _make_history(contrib_edge: float, contrib_step: float) -> dict[str, float]:
    return {
        "contrib_mask": 0.3,
        "contrib_stft": 0.3,
        "contrib_preserve": 0.2,
        "contrib_energy": 0.1,
        "contrib_edge": contrib_edge,
        "contrib_step": contrib_step,
    }
