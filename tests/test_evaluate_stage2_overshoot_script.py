"""Tests for Stage 2 overshoot evaluation CLI script."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from scripts.evaluate_stage2_overshoot import main


def test_cli_json_output_contains_expected_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI emits parseable JSON metrics for default Stage 2 taps."""
    monkeypatch.setattr(sys, "argv", ["evaluate_stage2_overshoot.py", "--json"])

    main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["output_sample_rate"] == 705_600
    assert np.isfinite(payload["step"]["ratio"])
    assert np.isfinite(payload["square"]["ratio"])
    assert payload["step"]["ratio"] >= 0.0
    assert payload["square"]["ratio"] >= 0.0
