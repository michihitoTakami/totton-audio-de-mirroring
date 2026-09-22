"""Tests for the transient dwell projection."""

from pathlib import Path

import numpy as np
import pytest
import torch

from totton_audio_de_mirroring.models.capb import (
    CAPB,
    DEFAULT_CONTROL_STRIDE,
    RoutingPriorConfig,
    capb_candidate_from_checkpoint,
    capb_from_checkpoint,
)
from totton_audio_de_mirroring.models.proto_bank import (
    build_prototype_bank_for_profile,
)
from totton_audio_de_mirroring.models.transient_dwell import (
    TransientDwellConfig,
    apply_transient_dwell,
    detect_transient_frames,
)
from totton_audio_de_mirroring.training.capb_trainer import load_capb_training_config

NAMES = ("sharp", "mid", "gentle")
ENABLED = TransientDwellConfig(enabled=True)
SOURCE_RATE = 44_100


def _frames(length: int) -> int:
    return -(-length // DEFAULT_CONTROL_STRIDE)


def _all_gentle(frames: int) -> torch.Tensor:
    weights = torch.zeros(1, 3, frames)
    weights[:, 2, :] = 1.0
    return weights


def _dc_step(length: int = 44_100) -> tuple[torch.Tensor, int]:
    """Return a unit step whose edge sits on a controller frame centre."""
    edge_frame = (length // 2) // DEFAULT_CONTROL_STRIDE
    source = torch.zeros(1, length)
    source[:, edge_frame * DEFAULT_CONTROL_STRIDE :] = 1.0
    return source, edge_frame


def _square(freq_hz: float, length: int = 44_100) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32) / SOURCE_RATE
    return torch.sign(torch.sin(2 * torch.pi * freq_hz * t)).unsqueeze(0)


def _pink_noise(length: int = 44_100, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(length)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length)
    spectrum[1:] /= np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum, n=length)
    pink /= np.abs(pink).max()
    return torch.from_numpy(pink.astype(np.float32)).unsqueeze(0)


def test_config_validates_ranges() -> None:
    with pytest.raises(ValueError, match="crest_threshold"):
        TransientDwellConfig(crest_threshold=1.0)
    with pytest.raises(ValueError, match="sharp_zone_frames"):
        TransientDwellConfig(gentle_zone_frames=4, sharp_zone_frames=4)
    with pytest.raises(ValueError, match="ramp_frames"):
        TransientDwellConfig(ramp_frames=-1)
    with pytest.raises(ValueError, match="mapping"):
        TransientDwellConfig.from_mapping(1.0)  # type: ignore[arg-type]


def test_config_roundtrips_and_defaults_to_disabled() -> None:
    config = TransientDwellConfig(enabled=True, sharp_zone_frames=5, ramp_frames=2)
    assert TransientDwellConfig.from_mapping(config.to_dict()) == config
    assert TransientDwellConfig.from_mapping(None) == TransientDwellConfig()
    assert not TransientDwellConfig.from_mapping(None).enabled
    assert TransientDwellConfig.from_mapping({"sharp_zone_frames": 6}).enabled


def test_disabled_projection_is_identity() -> None:
    source, _ = _dc_step()
    weights = _all_gentle(_frames(source.shape[-1]))
    projected = apply_transient_dwell(
        weights, source, NAMES, DEFAULT_CONTROL_STRIDE, TransientDwellConfig()
    )
    assert torch.equal(projected, weights)


def test_detector_marks_step_frame_and_square_edges() -> None:
    source, edge_frame = _dc_step()
    frames = _frames(source.shape[-1])
    events = detect_transient_frames(source, frames, DEFAULT_CONTROL_STRIDE, ENABLED)
    assert events.shape == (1, 1, frames)
    assert events[0, 0, edge_frame] == 1.0
    assert events.sum() == 1.0

    square_events = detect_transient_frames(
        _square(1_000.0), frames, DEFAULT_CONTROL_STRIDE, ENABLED
    )
    assert square_events.mean() > 0.99


def test_detector_ignores_stationary_noise_and_tone() -> None:
    frames = _frames(44_100)
    noise_events = detect_transient_frames(
        _pink_noise(), frames, DEFAULT_CONTROL_STRIDE, ENABLED
    )
    assert noise_events.sum() == 0.0
    t = torch.arange(44_100, dtype=torch.float32) / SOURCE_RATE
    tone = torch.sin(2 * torch.pi * 3_000.0 * t).unsqueeze(0)
    tone_events = detect_transient_frames(tone, frames, DEFAULT_CONTROL_STRIDE, ENABLED)
    assert tone_events[0, 0, 2:-2].sum() == 0.0


def test_step_keeps_gentle_core_mid_ring_and_sharp_beyond() -> None:
    source, edge = _dc_step()
    weights = _all_gentle(_frames(source.shape[-1]))
    projected = apply_transient_dwell(
        weights, source, NAMES, DEFAULT_CONTROL_STRIDE, ENABLED
    )
    sharp, mid, gentle = projected[0]
    assert torch.allclose(projected.sum(dim=1), torch.ones(1, projected.shape[-1]))
    assert torch.all(projected >= 0.0)
    # Gentle core: the controller mix is untouched within one frame.
    for offset in (-1, 0, 1):
        assert gentle[edge + offset] == pytest.approx(1.0)
    # Mid ring: fully middle from two to four frames (sharp support), no sharp.
    for offset in (-4, -2, 2, 4):
        assert mid[edge + offset] == pytest.approx(1.0)
        assert sharp[edge + offset] == pytest.approx(0.0)
    # Beyond sharp's support everything has returned to sharp.
    assert torch.all(sharp[: edge - 4] == 1.0)
    assert torch.all(sharp[edge + 5 :] == 1.0)


def test_default_zone_edges_are_frame_steps() -> None:
    source, edge = _dc_step()
    weights = _all_gentle(_frames(source.shape[-1]))
    gentle = apply_transient_dwell(
        weights, source, NAMES, DEFAULT_CONTROL_STRIDE, ENABLED
    )[0, 2]
    assert gentle[edge + 1] == pytest.approx(1.0)
    assert gentle[edge + 2] == pytest.approx(0.0)


def test_optional_ramp_is_linear_over_two_frames() -> None:
    source, edge = _dc_step()
    weights = _all_gentle(_frames(source.shape[-1]))
    gentle = apply_transient_dwell(
        weights,
        source,
        NAMES,
        DEFAULT_CONTROL_STRIDE,
        TransientDwellConfig(enabled=True, ramp_frames=1),
    )[0, 2]
    assert gentle[edge + 1] == pytest.approx(1.0)
    assert gentle[edge + 2] == pytest.approx(2.0 / 3.0)
    assert gentle[edge + 3] == pytest.approx(1.0 / 3.0)
    assert gentle[edge + 4] == pytest.approx(0.0)


def test_fast_square_is_left_untouched() -> None:
    source = _square(1_000.0)
    weights = _all_gentle(_frames(source.shape[-1]))
    projected = apply_transient_dwell(
        weights, source, NAMES, DEFAULT_CONTROL_STRIDE, ENABLED
    )
    assert torch.allclose(projected, weights)


def test_stationary_noise_releases_to_sharp_or_mid() -> None:
    source = _pink_noise()
    weights = _all_gentle(_frames(source.shape[-1]))
    released = apply_transient_dwell(
        weights, source, NAMES, DEFAULT_CONTROL_STRIDE, ENABLED
    )
    assert torch.all(released[0, 0] == 1.0)
    held = apply_transient_dwell(
        weights,
        source,
        NAMES,
        DEFAULT_CONTROL_STRIDE,
        TransientDwellConfig(enabled=True, release_to_sharp=False),
    )
    assert torch.all(held[0, 1] == 1.0)
    assert torch.all(held[0, 0] == 0.0)


def test_two_prototype_bank_moves_gentle_to_sharp_only_beyond_support() -> None:
    source, edge = _dc_step()
    frames = _frames(source.shape[-1])
    weights = torch.zeros(1, 2, frames)
    weights[:, 1, :] = 1.0
    projected = apply_transient_dwell(
        weights, source, ("sharp", "gentle"), DEFAULT_CONTROL_STRIDE, ENABLED
    )
    assert projected[0, 1, edge + 4] == pytest.approx(1.0)
    assert projected[0, 0, edge + 8] == pytest.approx(1.0)


def test_controller_mix_inside_core_is_preserved() -> None:
    source, edge = _dc_step()
    frames = _frames(source.shape[-1])
    weights = torch.full((1, 3, frames), 1.0 / 3.0)
    projected = apply_transient_dwell(
        weights, source, NAMES, DEFAULT_CONTROL_STRIDE, ENABLED
    )
    assert torch.allclose(projected[0, :, edge], weights[0, :, edge])
    assert projected[0, 1, edge + 4] == pytest.approx(2.0 / 3.0)
    assert projected[0, 0, edge + 4] == pytest.approx(1.0 / 3.0)


def test_frame_count_mismatch_falls_back_to_adaptive_pooling() -> None:
    source, _ = _dc_step(10_000)
    events = detect_transient_frames(source, 100, DEFAULT_CONTROL_STRIDE, ENABLED)
    assert events.shape == (1, 1, 100)
    assert events.sum() >= 1.0


def test_apply_rejects_invalid_inputs() -> None:
    source, _ = _dc_step(4_096)
    with pytest.raises(ValueError, match="prototypes"):
        apply_transient_dwell(
            torch.ones(1, 2, 64), source, NAMES, DEFAULT_CONTROL_STRIDE, ENABLED
        )
    with pytest.raises(ValueError, match="sharp and gentle"):
        apply_transient_dwell(
            torch.ones(1, 2, 64), source, ("a", "b"), DEFAULT_CONTROL_STRIDE, ENABLED
        )
    with pytest.raises(ValueError, match="non-empty"):
        detect_transient_frames(torch.zeros(1, 0), 4, DEFAULT_CONTROL_STRIDE, ENABLED)


def _physics_model(config: TransientDwellConfig) -> CAPB:
    bank = build_prototype_bank_for_profile(88_200, "v5b_sharp1023_midflat70")
    return CAPB(
        bank=bank,
        controller_dilation=2,
        controller_feature_mode="physics_routing",
        routing_prior=RoutingPriorConfig(
            focused_gentle_fraction=0.3, level_change_threshold=0.30
        ),
        transient_dwell=config,
    )


def test_capb_weights_stay_convex_with_dwell_enabled() -> None:
    model = _physics_model(ENABLED)
    source, _ = _dc_step(8_192)
    with torch.no_grad():
        weights = model.controller_weights(source)
    assert weights.shape[1:] == (3, _frames(8_192))
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-5)
    assert torch.all(weights >= 0.0)


def test_capb_dwell_returns_sharp_far_from_a_step() -> None:
    model = _physics_model(ENABLED)
    source, edge = _dc_step(44_100)
    with torch.no_grad():
        weights = model.controller_weights(source)[0]
    assert weights[0, edge + 12] == pytest.approx(1.0)
    assert weights[0, edge - 12] == pytest.approx(1.0)


def test_dwell_roundtrips_through_checkpoint() -> None:
    config = TransientDwellConfig(enabled=True, sharp_zone_frames=5)
    model = _physics_model(config)
    checkpoint = {
        "model_state": model.state_dict(),
        "prototype_profile": model.prototype_profile,
        "prototype_hash": model.prototype_hash,
        "controller_dilation": 2,
        "controller_feature_mode": "physics_routing",
        "routing_prior": model.routing_prior.to_dict(),
        "transient_dwell": config.to_dict(),
        "target_sample_rate": 88_200,
    }
    assert capb_from_checkpoint(checkpoint).transient_dwell == config
    candidate = capb_candidate_from_checkpoint(
        checkpoint, prototype_profile="long_sharp_1023_a120"
    )
    assert candidate.transient_dwell == config
    override = TransientDwellConfig()
    assert (
        capb_candidate_from_checkpoint(
            checkpoint,
            prototype_profile="long_sharp_1023_a120",
            transient_dwell=override,
        ).transient_dwell
        == override
    )


def test_legacy_checkpoint_has_dwell_disabled() -> None:
    model = CAPB(controller_dilation=2, controller_feature_mode="physics_routing")
    checkpoint = {
        "model_state": model.state_dict(),
        "prototype_profile": model.prototype_profile,
        "prototype_hash": model.prototype_hash,
        "controller_dilation": 2,
        "controller_feature_mode": "physics_routing",
        "target_sample_rate": 88_200,
    }
    assert not capb_from_checkpoint(checkpoint).transient_dwell.enabled


@pytest.mark.parametrize(
    "config_name",
    [
        "training_stage1_capb_transient_dwell_3p.yaml",
        "training_stage1_capb_48k_transient_dwell_3p.yaml",
    ],
)
def test_training_configs_enable_dwell(config_name: str) -> None:
    config = load_capb_training_config(Path("configs") / config_name)
    assert config.transient_dwell is not None
    assert config.transient_dwell.enabled
    assert config.transient_dwell.sharp_zone_frames == 4
    assert config.epochs == 20


def test_routing_v2_config_leaves_dwell_unset() -> None:
    config = load_capb_training_config(
        Path("configs/training_stage1_capb_routing_v2_3p.yaml")
    )
    assert config.transient_dwell is None
