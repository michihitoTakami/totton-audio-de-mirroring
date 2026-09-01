"""Tests for fail-closed long-FIR release selection."""

from typing import Any

from scripts.summarize_long_fir_selection import select_profile


def _candidate(*, eligible: bool, image_db: float) -> dict[str, Any]:
    row = {
        "passed": True,
        "cpu": {
            "g3_image_peak_db": image_db,
            "g2b_pre_echo": 1.0e-7,
            "g9_sideband_db": -120.0,
        },
    }
    return {
        "eligible": eligible,
        "families": {
            "44k1": {"seeds": [row]},
            "48k": {"seeds": [row]},
        },
    }


def test_selection_retains_release_when_no_profile_is_eligible() -> None:
    release = {
        "44k1": {"g3_image_peak_db": -100.0},
        "48k": {"g3_image_peak_db": -100.0},
    }

    selected = select_profile(
        release,
        {"long_sharp_1535_a120": _candidate(eligible=False, image_db=-120.0)},
    )

    assert selected == "release_v4"


def test_selection_accepts_eligible_meaningful_image_improvement() -> None:
    release = {
        "44k1": {"g3_image_peak_db": -100.0},
        "48k": {"g3_image_peak_db": -100.0},
    }

    selected = select_profile(
        release,
        {"long_sharp_1535_a120": _candidate(eligible=True, image_db=-101.0)},
    )

    assert selected == "long_sharp_1535_a120"
