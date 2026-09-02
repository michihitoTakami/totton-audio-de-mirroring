"""Tests for the CAPB model and losses."""

from pathlib import Path

import numpy as np
import pytest
import torch

from totton_audio_de_mirroring.data.capb_dataset import CAPBDataConfig
from totton_audio_de_mirroring.models.capb import (
    CAPB,
    DEFAULT_INIT_WEIGHTS,
    TWO_PROTOTYPE_INIT_WEIGHTS,
    capb_candidate_from_checkpoint,
    capb_from_checkpoint,
)
from totton_audio_de_mirroring.models.proto_bank import (
    TWO_PROTOTYPE_PROFILE,
    build_prototype_bank,
    build_prototype_bank_for_profile,
    prototype_specs_for_target_rate,
    upsample_with_kernel,
)
from totton_audio_de_mirroring.training.capb_losses import (
    CAPBLossWeights,
    compute_capb_losses,
    edge_ring_loss,
    plateau_ripple_loss,
    pre_echo_excess_loss,
    prototype_routing_loss,
    quiet_energy_loss,
    stationary_modulation_loss,
    weight_tv_loss,
)
from totton_audio_de_mirroring.training.capb_trainer import (
    CAPBTrainingConfig,
    _load_initial_checkpoint,
    _scale_controller_head,
    load_capb_training_config,
)
from totton_audio_de_mirroring.training.stft_loss import STFTLossConfig

STFT_CONFIGS = [STFTLossConfig(n_fft=512, hop_length=128, win_length=512)]


@pytest.fixture(scope="module")
def model() -> CAPB:
    torch.manual_seed(0)
    return CAPB()


def test_forward_shape(model: CAPB) -> None:
    source = torch.randn(2, 4_096)
    output = model(source)
    assert output.shape == (2, 8_192)
    assert torch.all(torch.isfinite(output))


def test_initial_weights_match_bias_init(model: CAPB) -> None:
    """Untrained controller must emit the configured blend distribution."""
    source = torch.randn(2, 8_192)
    mean_weights = model.mean_weights(source)
    np.testing.assert_allclose(mean_weights.numpy(), DEFAULT_INIT_WEIGHTS, atol=1e-5)


def test_two_prototype_profile_uses_endpoint_prior() -> None:
    bank = build_prototype_bank_for_profile(88_200, TWO_PROTOTYPE_PROFILE)
    model = CAPB(bank=bank)

    assert bank.names == ("sharp", "gentle")
    np.testing.assert_allclose(
        model.mean_weights(torch.randn(2, 4_096)).numpy(),
        TWO_PROTOTYPE_INIT_WEIGHTS,
        atol=1.0e-5,
    )


def test_controller_dilation_roundtrips_through_checkpoint() -> None:
    model = CAPB(controller_dilation=2, controller_feature_mode="envelope_flux")
    checkpoint = {
        "model_state": model.state_dict(),
        "prototype_profile": model.prototype_profile,
        "prototype_hash": model.prototype_hash,
        "controller_dilation": 2,
        "controller_feature_mode": "envelope_flux",
        "target_sample_rate": 88_200,
    }

    loaded = capb_from_checkpoint(checkpoint)

    assert loaded.controller_dilation == 2
    assert loaded.controller_feature_mode == "envelope_flux"
    assert loaded.controller.encoder[0].dilation == (2,)
    assert loaded.controller.encoder[0].in_channels == 4


def test_controller_dilation_must_be_positive() -> None:
    with pytest.raises(ValueError, match="dilation"):
        CAPB(controller_dilation=0)


def test_physics_routing_prior_is_sharp_on_stationary_tone() -> None:
    bank = build_prototype_bank_for_profile(88_200, TWO_PROTOTYPE_PROFILE)
    model = CAPB(
        bank=bank,
        controller_dilation=2,
        controller_feature_mode="physics_routing",
    )
    time = torch.arange(44_100, dtype=torch.float32) / 44_100.0
    source = torch.sin(2.0 * torch.pi * 1_000.0 * time).unsqueeze(0)

    _, weights = model(source, return_weights=True)

    core = weights[0, 0, 20:-20]
    assert float(torch.quantile(core, 0.05).detach()) > 0.99


def test_physics_routing_prior_is_sharp_on_stationary_noise() -> None:
    bank = build_prototype_bank_for_profile(88_200, TWO_PROTOTYPE_PROFILE)
    model = CAPB(
        bank=bank,
        controller_dilation=2,
        controller_feature_mode="physics_routing",
    )
    generator = torch.Generator().manual_seed(19)
    source = torch.randn(1, 44_100, generator=generator)

    _, weights = model(source, return_weights=True)

    core = weights[0, 0, 40:-40]
    assert float(torch.quantile(core, 0.05).detach()) > 0.99


def test_physics_routing_prior_selects_gentle_around_impulse() -> None:
    bank = build_prototype_bank_for_profile(88_200, TWO_PROTOTYPE_PROFILE)
    model = CAPB(
        bank=bank,
        controller_dilation=2,
        controller_feature_mode="physics_routing",
    )
    source = torch.zeros(1, 44_100)
    source[:, source.shape[1] // 2] = 1.0

    _, weights = model(source, return_weights=True)

    center = weights.shape[-1] // 2
    assert float(torch.mean(weights[0, 1, center - 3 : center + 4]).detach()) > 0.99


def test_physics_routing_prior_selects_gentle_on_sparse_square_edges() -> None:
    bank = build_prototype_bank_for_profile(88_200, TWO_PROTOTYPE_PROFILE)
    model = CAPB(
        bank=bank,
        controller_dilation=2,
        controller_feature_mode="physics_routing",
    )
    time = torch.arange(44_100, dtype=torch.float32) / 44_100.0
    source = torch.sign(torch.sin(2.0 * torch.pi * 50.0 * time)).unsqueeze(0)

    _, weights = model(source, return_weights=True)

    core = weights[0, 1, 20:-20]
    assert float(torch.quantile(core, 0.05).detach()) > 0.99


def test_three_prototype_prior_routes_sparse_impulse_to_middle() -> None:
    bank = build_prototype_bank_for_profile(88_200, "release_v4")
    model = CAPB(
        bank=bank,
        controller_dilation=2,
        controller_feature_mode="physics_routing",
    )
    source = torch.zeros(1, 44_100)
    source[:, source.shape[1] // 2] = 1.0

    _, weights = model(source, return_weights=True)

    center = weights.shape[-1] // 2
    assert float(weights[0, 1, center].detach()) > 0.99


def test_weights_are_convex(model: CAPB) -> None:
    source = torch.randn(1, 4_096)
    _, weights = model(source, return_weights=True)
    sums = weights.sum(dim=1)
    torch.testing.assert_close(sums, torch.ones_like(sums))
    assert torch.all(weights >= 0.0)


def test_only_controller_is_trainable(model: CAPB) -> None:
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert all(name.startswith("controller.") for name in trainable)
    assert "kernels" in dict(model.named_buffers())


def test_head_bias_is_frozen(model: CAPB) -> None:
    """The static blend component must stay fixed (anti-collapse guard)."""
    assert model.controller.head.bias is not None
    assert not model.controller.head.bias.requires_grad
    assert model.controller.head.weight.requires_grad


def test_epoch0_output_close_to_init_blend(model: CAPB) -> None:
    """Untrained output must equal the fixed init blend of prototypes."""
    rng = np.random.default_rng(5)
    source_np = rng.standard_normal(8_192)
    bank = build_prototype_bank()
    expected = sum(
        weight * upsample_with_kernel(source_np, bank.kernels[i], 2)
        for i, weight in enumerate(DEFAULT_INIT_WEIGHTS)
    )
    with torch.no_grad():
        output = model(torch.from_numpy(source_np).float().unsqueeze(0))
    core = slice(1_024, -1_024)
    np.testing.assert_allclose(
        output.squeeze(0).numpy()[core], expected[core], atol=2e-4
    )


def test_invalid_input_raises(model: CAPB) -> None:
    with pytest.raises(ValueError, match="batch, time"):
        model(torch.randn(16))


def test_plateau_loss_zero_for_flat_output() -> None:
    output = torch.full((1, 1_000), 0.5)
    mask = torch.ones_like(output)
    assert float(plateau_ripple_loss(output, mask)) == pytest.approx(0.0)


def test_plateau_loss_detects_ripple() -> None:
    time_axis = torch.arange(1_000, dtype=torch.float32)
    ripple = 0.01 * torch.sin(2 * np.pi * 0.25 * time_axis).unsqueeze(0)
    mask = torch.ones_like(ripple)
    assert float(plateau_ripple_loss(0.5 + ripple, mask)) > 1e-5


def test_quiet_loss_zero_for_silence() -> None:
    output = torch.zeros(1, 1_000)
    mask = torch.ones_like(output)
    assert float(quiet_energy_loss(output, mask)) == pytest.approx(0.0)


def test_dense_edge_loss_prefers_gentle_prototype() -> None:
    bank = build_prototype_bank()
    source_time = np.arange(4_410, dtype=np.float64) / 44_100.0
    source = 0.5 * np.sign(np.sin(2.0 * np.pi * 5_000.0 * source_time))
    sharp = torch.from_numpy(upsample_with_kernel(source, bank.kernels[0], 2)).float()
    gentle = torch.from_numpy(upsample_with_kernel(source, bank.kernels[2], 2)).float()
    edge_mask = torch.ones(1, gentle.numel())

    sharp_loss = edge_ring_loss(sharp.unsqueeze(0), gentle.unsqueeze(0), edge_mask)
    gentle_loss = edge_ring_loss(gentle.unsqueeze(0), gentle.unsqueeze(0), edge_mask)

    assert float(sharp_loss) > 0.0
    assert float(gentle_loss) == pytest.approx(0.0)


def test_pre_echo_excess_loss_uses_gentle_as_zero_baseline() -> None:
    gentle = torch.full((2, 64), 0.1)
    output = gentle.clone()
    output[0, 8:16] = 0.2
    mask = torch.zeros_like(output)
    mask[:, 8:16] = 1.0

    loss = pre_echo_excess_loss(output, gentle, mask)
    baseline = pre_echo_excess_loss(gentle, gentle, mask)

    assert float(loss) > 0.0
    assert float(baseline) == pytest.approx(0.0)


def test_prototype_routing_prefers_gentle_edges_and_sharp_stationary() -> None:
    correct = torch.tensor([[[0.9, 0.1], [0.05, 0.05], [0.05, 0.85]]])
    reversed_weights = torch.flip(correct, dims=(1,))
    edge_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    stationary = torch.tensor([True])

    correct_loss = prototype_routing_loss(
        correct,
        edge_mask,
        stationary,
        sharp_index=0,
        gentle_index=2,
    )
    reversed_loss = prototype_routing_loss(
        reversed_weights,
        edge_mask,
        stationary,
        sharp_index=0,
        gentle_index=2,
    )
    assert float(correct_loss) < float(reversed_loss)


def test_prototype_routing_balances_sparse_edge_against_long_safe_region() -> None:
    frames = 100
    edge_mask = torch.zeros(1, frames * 2)
    edge_mask[:, -2:] = 1.0
    safe_mask = 1.0 - edge_mask
    balanced = torch.full((1, 2, frames), 0.5)
    static_sharp = torch.empty(1, 2, frames)
    static_sharp[:, 0, :] = 0.99
    static_sharp[:, 1, :] = 0.01

    balanced_loss = prototype_routing_loss(
        balanced,
        edge_mask,
        torch.tensor([True]),
        safe_active_mask=safe_mask,
        sharp_index=0,
        gentle_index=1,
    )
    static_loss = prototype_routing_loss(
        static_sharp,
        edge_mask,
        torch.tensor([True]),
        safe_active_mask=safe_mask,
        sharp_index=0,
        gentle_index=1,
    )

    assert float(balanced_loss) < float(static_loss)


def test_prototype_routing_mines_worst_batch_quartile() -> None:
    weights = torch.full((4, 2, 2), 0.001)
    weights[:, 0, :] = 0.999
    weights[-1, 0, :] = 0.001
    weights[-1, 1, :] = 0.999
    safe = torch.ones(4, 4)

    loss = prototype_routing_loss(
        weights,
        torch.zeros_like(safe),
        torch.ones(4, dtype=torch.bool),
        safe_active_mask=safe,
        sharp_index=0,
        gentle_index=1,
    )

    assert float(loss) > 6.0


def test_prototype_routing_recovers_weight_below_generic_epsilon() -> None:
    """Routing labels must retain a gradient after a softmax nearly saturates."""
    logits = torch.tensor(
        [[[-30.0, -30.0], [0.0, 0.0], [0.0, 0.0]]], requires_grad=True
    )
    weights = torch.softmax(logits, dim=1)
    loss = prototype_routing_loss(
        weights,
        torch.zeros(1, 4),
        torch.tensor([True]),
        sharp_index=0,
        gentle_index=2,
    )

    loss.backward()

    assert logits.grad is not None
    assert torch.all(torch.abs(logits.grad[:, 0, :]) > 0.0)


def test_prototype_routing_defers_focused_transients_to_pre_echo_loss() -> None:
    weights = torch.tensor([[[0.2, 0.2], [0.1, 0.1], [0.7, 0.7]]])
    loss = prototype_routing_loss(
        weights,
        torch.ones(1, 4),
        torch.tensor([False]),
        focused_transient=torch.tensor([True]),
        sharp_index=0,
        gentle_index=2,
    )
    assert float(loss) == pytest.approx(0.0)


def test_prototype_routing_labels_only_focused_risk_window() -> None:
    weights = torch.tensor([[[0.05, 0.05], [0.90, 0.90], [0.05, 0.05]]])
    edge_mask = torch.ones(1, 4)
    risk_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    loss = prototype_routing_loss(
        weights,
        edge_mask,
        torch.tensor([False]),
        focused_transient=torch.tensor([True]),
        focused_risk_mask=risk_mask,
        sharp_index=0,
        gentle_index=2,
    )

    reversed_weights = torch.tensor([[[0.90, 0.90], [0.05, 0.05], [0.05, 0.05]]])
    reversed_loss = prototype_routing_loss(
        reversed_weights,
        edge_mask,
        torch.tensor([False]),
        focused_transient=torch.tensor([True]),
        focused_risk_mask=risk_mask,
        sharp_index=0,
        gentle_index=2,
    )
    assert float(loss) < float(reversed_loss)


def test_prototype_routing_prefers_sharp_on_safe_active_focused_body() -> None:
    correct = torch.tensor([[[0.90, 0.05], [0.05, 0.90], [0.05, 0.05]]])
    safe = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    risk = 1.0 - safe

    correct_loss = prototype_routing_loss(
        correct,
        risk,
        torch.tensor([False]),
        focused_transient=torch.tensor([True]),
        focused_risk_mask=risk,
        safe_active_mask=safe,
        sharp_index=0,
        gentle_index=2,
    )
    reversed_loss = prototype_routing_loss(
        torch.flip(correct, dims=(1,)),
        risk,
        torch.tensor([False]),
        focused_transient=torch.tensor([True]),
        focused_risk_mask=risk,
        safe_active_mask=safe,
        sharp_index=0,
        gentle_index=2,
    )

    assert float(correct_loss) < float(reversed_loss)


def test_tv_loss_zero_for_constant_weights() -> None:
    weights = torch.full((2, 3, 10), 1.0 / 3.0)
    assert float(weight_tv_loss(weights)) == pytest.approx(0.0)


def test_stationary_modulation_loss_is_zero_for_fixed_blend() -> None:
    prototypes = torch.randn(2, 3, 128)
    weights = torch.full((2, 3, 8), 1.0 / 3.0)
    output = torch.sum(weights[:, :, :1] * prototypes, dim=1)
    loss = stationary_modulation_loss(
        output, prototypes, weights, torch.tensor([True, True]), trim=0
    )
    assert float(loss) < 1.0e-6


def test_stationary_modulation_loss_detects_time_varying_blend() -> None:
    prototypes = torch.stack(
        (torch.ones(128), torch.zeros(128), -torch.ones(128)), dim=0
    ).unsqueeze(0)
    weights = torch.zeros(1, 3, 8)
    weights[:, 0, ::2] = 1.0
    weights[:, 2, 1::2] = 1.0
    weights_up = torch.nn.functional.interpolate(weights, size=128, mode="linear")
    output = torch.sum(weights_up * prototypes, dim=1)
    loss = stationary_modulation_loss(
        output, prototypes, weights, torch.tensor([True]), trim=0
    )
    ignored = stationary_modulation_loss(
        output, prototypes, weights, torch.tensor([False]), trim=0
    )
    assert float(loss) > 0.0
    assert float(ignored) == pytest.approx(0.0)


def test_stationary_modulation_loss_observes_weights_for_equal_prototypes() -> None:
    prototypes = torch.ones(1, 3, 128)
    weights = torch.zeros(1, 3, 8)
    weights[:, 0, ::2] = 1.0
    weights[:, 1, 1::2] = 1.0
    output = torch.ones(1, 128)
    loss = stationary_modulation_loss(
        output, prototypes, weights, torch.tensor([True]), trim=0
    )
    assert float(loss) > 0.0


def test_compute_capb_losses_total(model: CAPB) -> None:
    source = torch.randn(2, 4_096)
    target = torch.randn(2, 8_192)
    output, weights = model(source, return_weights=True)
    losses = compute_capb_losses(
        output=output,
        target=target,
        weights_frames=weights,
        flat_mask=torch.zeros_like(target),
        quiet_mask=torch.zeros_like(target),
        stft_configs=STFT_CONFIGS,
        loss_weights=CAPBLossWeights(),
        trim=256,
    )
    assert set(losses) == {
        "wave",
        "stft",
        "plateau",
        "quiet",
        "tv",
        "entropy_floor",
        "stationary_modulation",
        "edge_ring",
        "pre_echo_excess",
        "post_echo_excess",
        "prototype_routing",
        "total",
    }
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    grads = [p.grad for p in model.controller.parameters() if p.grad is not None]
    assert grads, "controller must receive gradients"


def test_compute_losses_routes_focused_onset_edge_to_gentle(model: CAPB) -> None:
    source = torch.randn(1, 2_048)
    output, weights, prototypes = model.forward_with_details(source)
    edge = torch.zeros_like(output)
    edge[:, output.shape[1] // 2 - 32 : output.shape[1] // 2 + 32] = 1.0
    losses = compute_capb_losses(
        output=output,
        target=output.detach(),
        weights_frames=weights,
        flat_mask=torch.zeros_like(output),
        quiet_mask=torch.zeros_like(output),
        stft_configs=STFT_CONFIGS,
        loss_weights=CAPBLossWeights(prototype_routing=1.0),
        trim=256,
        edge_mask=edge,
        gentle_output=prototypes[:, -1].detach(),
        stationary=torch.tensor([False]),
        focused_transient=torch.tensor([True]),
        sharp_index=0,
        gentle_index=2,
    )

    assert float(losses["prototype_routing"].detach()) > 0.0


def test_losses_shape_validation() -> None:
    output = torch.randn(1, 100)
    with pytest.raises(ValueError, match="share shape"):
        compute_capb_losses(
            output=output,
            target=torch.randn(1, 99),
            weights_frames=torch.rand(1, 3, 4),
            flat_mask=torch.zeros(1, 100),
            quiet_mask=torch.zeros(1, 100),
            stft_configs=STFT_CONFIGS,
            loss_weights=CAPBLossWeights(),
            trim=0,
        )


def _make_checkpoint(target_rate: int | None, input_rate: int | None) -> dict:
    rate = target_rate if target_rate is not None else 88_200
    bank = build_prototype_bank(prototype_specs_for_target_rate(rate), sample_rate=rate)
    model = CAPB(bank=bank)
    checkpoint: dict = {"model_state": model.state_dict()}
    if target_rate is not None:
        checkpoint["target_sample_rate"] = target_rate
    if input_rate is not None:
        checkpoint["expected_input_rate"] = input_rate
    return checkpoint


def test_capb_from_checkpoint_selects_48k_bank() -> None:
    """A 48k checkpoint must be paired with the 48k prototype kernels."""
    checkpoint = _make_checkpoint(target_rate=96_000, input_rate=48_000)
    model = capb_from_checkpoint(checkpoint)
    from totton_audio_de_mirroring.models.proto_bank import PROTOTYPE_SPECS_48K

    reference = build_prototype_bank(PROTOTYPE_SPECS_48K, sample_rate=96_000)
    assert model.kernel_size == reference.kernels.shape[1]


def test_capb_from_checkpoint_legacy_defaults_to_44k1() -> None:
    """run9-era checkpoints without rate keys load with the 44.1k bank."""
    checkpoint = _make_checkpoint(target_rate=None, input_rate=None)
    model = capb_from_checkpoint(checkpoint)
    reference = build_prototype_bank()
    assert model.kernel_size == reference.kernels.shape[1]


def test_capb_from_checkpoint_rejects_inconsistent_rates() -> None:
    checkpoint = _make_checkpoint(target_rate=96_000, input_rate=44_100)
    with pytest.raises(ValueError, match="inconsistent"):
        capb_from_checkpoint(checkpoint)


def test_capb_from_checkpoint_requires_model_state() -> None:
    with pytest.raises(RuntimeError, match="model_state"):
        capb_from_checkpoint({"target_sample_rate": 96_000})


def test_profiled_checkpoint_rebuilds_named_kernels() -> None:
    profile = "long_sharp_2047_a120"
    bank = build_prototype_bank_for_profile(88_200, profile)
    model = CAPB(bank=bank)
    state = model.state_dict()
    state["kernels"] = torch.zeros_like(state["kernels"])
    checkpoint = {
        "model_state": state,
        "target_sample_rate": 88_200,
        "expected_input_rate": 44_100,
        "prototype_profile": profile,
        "prototype_hash": bank.coefficient_hash,
        "fir_compute_dtype": "float32",
    }

    loaded = capb_from_checkpoint(checkpoint)

    assert loaded.kernel_size == 2047
    assert torch.count_nonzero(loaded.kernels) > 0
    assert loaded.prototype_hash == bank.coefficient_hash


def test_profiled_checkpoint_rejects_hash_mismatch() -> None:
    bank = build_prototype_bank_for_profile(88_200, "long_sharp_1023_a120")
    checkpoint = {
        "model_state": CAPB(bank=bank).state_dict(),
        "target_sample_rate": 88_200,
        "prototype_profile": "long_sharp_1023_a120",
        "prototype_hash": "wrong",
    }
    with pytest.raises(ValueError, match="prototype_hash"):
        capb_from_checkpoint(checkpoint)


def test_candidate_transfer_changes_bank_but_preserves_controller() -> None:
    checkpoint = _make_checkpoint(target_rate=88_200, input_rate=44_100)
    baseline = capb_from_checkpoint(checkpoint)

    candidate = capb_candidate_from_checkpoint(
        checkpoint,
        prototype_profile="long_sharp_2047_a120",
        fir_compute_dtype="float64",
    )

    assert candidate.kernel_size == 2047
    assert candidate.kernels.dtype == torch.float64
    for actual, expected in zip(
        candidate.controller.parameters(), baseline.controller.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_float64_fir_keeps_controller_float32() -> None:
    bank = build_prototype_bank_for_profile(88_200, "long_sharp_1023_a120")
    model = CAPB(bank=bank, fir_compute_dtype="float64")
    source = torch.randn(1, 512)

    output, weights = model(source, return_weights=True)

    assert output.dtype == torch.float64
    assert weights.dtype == torch.float32


def test_training_config_loads_initial_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "training.yaml"
    checkpoint_path = tmp_path / "initial.pt"
    config_path.write_text(f"initial_checkpoint: {checkpoint_path}\n")

    config = load_capb_training_config(config_path)

    assert config.initial_checkpoint == checkpoint_path


def test_training_config_loads_long_fir_candidate_options(tmp_path: Path) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        "prototype_profile: long_sharp_2047_a120\n"
        "fir_compute_dtype: float64\n"
        "initial_controller_only: true\n"
    )

    config = load_capb_training_config(config_path)

    assert config.prototype_profile == "long_sharp_2047_a120"
    assert config.fir_compute_dtype == "float64"
    assert config.initial_controller_only


def test_training_config_loads_checkpoint_interval(tmp_path: Path) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text("checkpoint_interval_epochs: 5\n")
    assert load_capb_training_config(config_path).checkpoint_interval_epochs == 5

    config_path.write_text("checkpoint_interval_epochs: -1\n")
    with pytest.raises(ValueError, match="checkpoint_interval_epochs"):
        load_capb_training_config(config_path)


def test_training_config_validates_initial_head_scale(tmp_path: Path) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text("initial_head_scale: 0.75\n")
    assert load_capb_training_config(config_path).initial_head_scale == 0.75

    config_path.write_text("initial_head_scale: 0\n")
    with pytest.raises(ValueError, match="initial_head_scale"):
        load_capb_training_config(config_path)


def test_scale_controller_head_preserves_bias() -> None:
    candidate = CAPB()
    with torch.no_grad():
        candidate.controller.head.weight.fill_(2.0)
    bias_before = candidate.controller.head.bias.detach().clone()

    _scale_controller_head(candidate, 0.75)

    torch.testing.assert_close(
        candidate.controller.head.weight,
        torch.full_like(candidate.controller.head.weight, 1.5),
    )
    torch.testing.assert_close(candidate.controller.head.bias, bias_before)


def test_initial_checkpoint_loader_validates_rate(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "initial.pt"
    torch.save(_make_checkpoint(target_rate=96_000, input_rate=48_000), checkpoint_path)
    training_config = CAPBTrainingConfig(initial_checkpoint=checkpoint_path)

    with pytest.raises(ValueError, match="target rate"):
        _load_initial_checkpoint(training_config, CAPBDataConfig())


def test_initial_checkpoint_loader_accepts_matching_rate(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "initial.pt"
    torch.save(_make_checkpoint(target_rate=96_000, input_rate=48_000), checkpoint_path)
    training_config = CAPBTrainingConfig(initial_checkpoint=checkpoint_path)
    data_config = CAPBDataConfig(
        source_sample_rate=48_000,
        target_sample_rate=96_000,
        near_nyquist_high_range_hz=(20_000.0, 23_700.0),
    )

    loaded = _load_initial_checkpoint(training_config, data_config)

    assert (
        loaded.kernel_size
        == build_prototype_bank(
            prototype_specs_for_target_rate(96_000), sample_rate=96_000
        ).kernels.shape[1]
    )


def test_initial_checkpoint_loader_transfers_controller_to_long_bank(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "initial.pt"
    torch.save(_make_checkpoint(target_rate=88_200, input_rate=44_100), checkpoint_path)
    training_config = CAPBTrainingConfig(
        initial_checkpoint=checkpoint_path,
        prototype_profile="long_sharp_2047_a120",
        initial_controller_only=True,
        border_trim=1_024,
    )

    loaded = _load_initial_checkpoint(training_config, CAPBDataConfig())

    assert loaded.kernel_size == 2_047
    assert loaded.prototype_profile == "long_sharp_2047_a120"
