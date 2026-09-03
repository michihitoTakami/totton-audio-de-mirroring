"""Tests for strict synthetic CAPB routing gates."""

import numpy as np
import pytest
import torch
from scripts.evaluate_capb_routing_gates import (
    _gate_rows,
    _masked_weight,
    _target_prototypes,
)

from totton_audio_de_mirroring.models.capb import RoutingPriorConfig


def test_masked_weight_maps_target_samples_to_controller_frames() -> None:
    weight = np.asarray([0.1, 0.9])
    mask = torch.tensor([0.0, 0.0, 1.0, 1.0])

    assert _masked_weight(weight, mask) == pytest.approx(0.9)


def test_gate_rows_bind_on_held_out_worst_case() -> None:
    signal_types = {
        "flowing_noise": {"safe_active": 0.95},
        "multitone": {"safe_active": 0.94},
        "pink_noise": {"safe_active": 0.96},
        "band_limited_noise": {"safe_active": 0.96},
        "near_nyquist_noise": {"safe_active": 0.96},
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


class _StubModel:
    def __init__(self, names: tuple[str, ...], fraction: float) -> None:
        self.prototype_names = names
        self.routing_prior = RoutingPriorConfig(focused_gentle_fraction=fraction)


def test_target_prototypes_follow_checkpoint_gentle_fraction() -> None:
    three = ("sharp", "mid", "gentle")
    legacy = _StubModel(three, 0.0)
    gentle_only = _StubModel(three, 1.0)
    two = _StubModel(("sharp", "gentle"), 0.0)

    assert _target_prototypes(legacy, "isolated_click", "pre_echo") == ("mid", "gentle")  # type: ignore[arg-type]
    assert _target_prototypes(gentle_only, "isolated_click", "pre_echo") == ("gentle",)  # type: ignore[arg-type]
    assert _target_prototypes(_StubModel(three, 0.85), "square_wave", "edge") == (
        "gentle",
    )  # type: ignore[arg-type]
    assert _target_prototypes(two, "isolated_click", "edge") == ("gentle",)  # type: ignore[arg-type]
    assert _target_prototypes(legacy, "pink_noise", "safe_active") == ("sharp",)  # type: ignore[arg-type]
