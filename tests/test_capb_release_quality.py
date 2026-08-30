"""Tests for CAPB cross-rate release-quality checks."""

from typing import Any

import pytest

from totton_audio_de_mirroring.evaluation.release_quality import (
    evaluate_release_quality,
    normalized_prototype_position,
)


def _gate(gain: float) -> dict[str, Any]:
    return {
        "all_passed": True,
        "execution": {"precision_mode": "strict_fp32"},
        "gates": [
            {
                "gate_id": "G5_gain",
                "rows": [{"probe_id": "impulse_train_10ms", "value": gain}],
            }
        ],
    }


def _impulse(capb_44: float = 0.30, capb_48: float = 0.29) -> dict[str, Any]:
    def family(capb: float, pre: float) -> dict[str, Any]:
        return {
            "metrics": {
                "capb": {
                    "local_energy": capb,
                    "peak": capb,
                    "pre_echo_mean_square": pre,
                },
                "gentle": {"local_energy": 0.20, "peak": 0.20},
                "sharp": {"local_energy": 0.60, "peak": 0.60},
            }
        }

    return {"44k1": family(capb_44, 1.0e-7), "48k": family(capb_48, 5.0e-8)}


def _distortion(value_48: float = -139.0) -> dict[str, Any]:
    metric = {
        "thd_1khz_20khz_db": -145.0,
        "smpte_imd_db": -125.0,
        "ccif_imd_db": -150.0,
        "added_am_sideband_db": -130.0,
    }
    candidate = dict(metric)
    candidate["thd_1khz_20khz_db"] = value_48
    floor = dict.fromkeys(metric, -140.0)
    return {
        "execution": {"precision_mode": "strict_fp32"},
        "44k1": {"distortion": {"capb": metric}},
        "48k": {"distortion": {"capb": candidate, "torch_sharp": floor}},
    }


def test_normalized_prototype_position_validates_span() -> None:
    assert normalized_prototype_position(0.3, 0.2, 0.6) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="sharp"):
        normalized_prototype_position(0.3, 0.6, 0.2)


def test_release_quality_accepts_rate_local_floor() -> None:
    result = evaluate_release_quality(
        _distortion(), _impulse(), _gate(0.44), _gate(0.39)
    )
    assert result["all_passed"]


def test_release_quality_rejects_non_strict_report() -> None:
    distortion = _distortion()
    distortion["execution"]["precision_mode"] = "tf32"
    with pytest.raises(ValueError, match="strict_fp32"):
        evaluate_release_quality(distortion, _impulse(), _gate(0.44), _gate(0.39))
