"""Tests for ABX protocol assets frozen for Issue #59."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

_ROOT = Path(".")
_GOLDEN_ROOT = _ROOT / "tests" / "fixtures" / "golden_samples"
_ABX_PAIRS_PATH = _GOLDEN_ROOT / "abx_pairs.json"
_MODEL_SELECTION_PATH = _GOLDEN_ROOT / "issue64_model_selection.json"
_PROTOCOL_PATH = _ROOT / "docs" / "abx_listening_protocol.md"
_TRIAL_TEMPLATE_PATH = _ROOT / "docs" / "templates" / "abx_trial_log_template.csv"
_SUMMARY_TEMPLATE_PATH = (
    _ROOT / "docs" / "templates" / "abx_session_summary_template.md"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_abx_pairs_define_fixed_minimum_sample_set() -> None:
    """ABX pairs should keep a deterministic minimum sample set.

    Physical Basis:
        Listening comparisons must use fixed naive/NMSE pairs to isolate
        perceived mirror-harshness reduction from sample-selection variance.
    """
    payload = _load_json(_ABX_PAIRS_PATH)
    pairs = payload["pairs"]
    assert isinstance(pairs, list)
    assert len(pairs) >= 2

    sample_ids = {str(pair["sample_id"]) for pair in pairs}
    assert {"sample_a", "sample_b"}.issubset(sample_ids)


def test_abx_pair_paths_exist_and_are_npy() -> None:
    """Every ABX pair path should exist for reproducible listening sessions.

    Physical Basis:
        Missing pair assets invalidate A/B/X trial reproducibility and make
        hard-metric and listening-metric comparisons inconsistent.
    """
    payload = _load_json(_ABX_PAIRS_PATH)
    for pair in payload["pairs"]:
        for key in ("input", "naive", "nmse"):
            path = Path(str(pair[key]))
            assert path.suffix == ".npy"
            assert path.exists(), f"Missing ABX asset: {path}"


def test_protocol_templates_exist_and_include_required_fields() -> None:
    """Templates must contain quantitative and free-text recording fields.

    Physical Basis:
        ABX conclusions are valid only when correctness metrics and qualitative
        transient/harshness observations are documented together.
    """
    assert _PROTOCOL_PATH.exists()
    assert _TRIAL_TEMPLATE_PATH.exists()
    assert _SUMMARY_TEMPLATE_PATH.exists()

    with _TRIAL_TEMPLATE_PATH.open("r", encoding="utf-8", newline="") as file_obj:
        header = next(csv.reader(file_obj))

    required_columns = {
        "sample_id",
        "listener_answer",
        "is_correct",
        "harshness_note",
        "attack_note",
    }
    assert required_columns.issubset(set(header))

    summary_text = _SUMMARY_TEMPLATE_PATH.read_text(encoding="utf-8")
    assert "hit_rate_percent" in summary_text
    assert "binomial_test_p_value" in summary_text
    assert "Harshness / Graininess" in summary_text
    assert "Transient Attack" in summary_text


def test_protocol_references_frozen_assets_and_report_path() -> None:
    """Protocol document should reference frozen pair assets and save path.

    Physical Basis:
        Fixed asset references and deterministic report locations are needed
        to compare listening outcomes against frozen regression baselines.
    """
    protocol_text = _PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "tests/fixtures/golden_samples/abx_pairs.json" in protocol_text
    assert "tests/fixtures/golden_samples/issue64_model_selection.json" in protocol_text
    assert "reports/abx/<session_id>/" in protocol_text

    selection_payload = _load_json(_MODEL_SELECTION_PATH)
    assert "selected_checkpoint_sha256" in selection_payload
