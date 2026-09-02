"""Tests for strict synthetic CAPB routing gates."""

import numpy as np
import pytest
import torch
from scripts.evaluate_capb_routing_gates import _gate_rows, _masked_weight


def test_masked_weight_maps_target_samples_to_controller_frames() -> None:
    weight = np.asarray([0.1, 0.9])
    mask = torch.tensor([0.0, 0.0, 1.0, 1.0])

    assert _masked_weight(weight, mask) == pytest.approx(0.9)


def test_gate_rows_bind_on_held_out_worst_case() -> None:
    signal_types = {
        "flowing_noise": {"safe_active": 0.95},
        "multitone": {"safe_active": 0.94},
        "damped_string": {
            "safe_active": 0.93,
            "pre_echo": 0.95,
            "post_echo": 0.95,
            "edge": 0.95,
        },
        "isolated_click": {"pre_echo": 0.95, "post_echo": 0.95, "edge": 0.95},
        "tone_burst": {"pre_echo": 0.95, "post_echo": 0.95, "edge": 0.95},
        "clustered_impacts": {
            "pre_echo": 0.95,
            "post_echo": 0.95,
            "edge": 0.95,
        },
        "square_wave": {"edge": 0.95},
        "step_plateau": {"edge": 0.95},
        "string_riff": {"edge": 0.95},
        "impact_stream": {"edge": 0.95},
    }
    held = {key: dict(value) for key, value in signal_types.items()}
    held["clustered_impacts"]["post_echo"] = 0.85

    rows = _gate_rows(
        {"canonical": signal_types, "held_out": held},
        stationary_threshold=0.90,
        transient_threshold=0.90,
    )
    post = next(row for row in rows if row["gate_id"] == "R3_post_echo_protective")

    assert not post["passed"]
    assert post["worst_tier"] == "held_out"
    assert post["worst_signal_type"] == "clustered_impacts"
