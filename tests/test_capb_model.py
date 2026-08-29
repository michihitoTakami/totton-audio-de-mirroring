"""Tests for the CAPB model and losses."""

from pathlib import Path

import numpy as np
import pytest
import torch

from totton_audio_de_mirroring.data.capb_dataset import CAPBDataConfig
from totton_audio_de_mirroring.models.capb import CAPB, DEFAULT_INIT_WEIGHTS
from totton_audio_de_mirroring.models.proto_bank import (
    build_prototype_bank,
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
        "prototype_routing",
        "total",
    }
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    grads = [p.grad for p in model.controller.parameters() if p.grad is not None]
    assert grads, "controller must receive gradients"


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
    from totton_audio_de_mirroring.models.capb import capb_from_checkpoint

    checkpoint = _make_checkpoint(target_rate=96_000, input_rate=48_000)
    model = capb_from_checkpoint(checkpoint)
    from totton_audio_de_mirroring.models.proto_bank import PROTOTYPE_SPECS_48K

    reference = build_prototype_bank(PROTOTYPE_SPECS_48K, sample_rate=96_000)
    assert model.kernel_size == reference.kernels.shape[1]


def test_capb_from_checkpoint_legacy_defaults_to_44k1() -> None:
    """run9-era checkpoints without rate keys load with the 44.1k bank."""
    from totton_audio_de_mirroring.models.capb import capb_from_checkpoint

    checkpoint = _make_checkpoint(target_rate=None, input_rate=None)
    model = capb_from_checkpoint(checkpoint)
    reference = build_prototype_bank()
    assert model.kernel_size == reference.kernels.shape[1]


def test_capb_from_checkpoint_rejects_inconsistent_rates() -> None:
    from totton_audio_de_mirroring.models.capb import capb_from_checkpoint

    checkpoint = _make_checkpoint(target_rate=96_000, input_rate=44_100)
    with pytest.raises(ValueError, match="inconsistent"):
        capb_from_checkpoint(checkpoint)


def test_capb_from_checkpoint_requires_model_state() -> None:
    from totton_audio_de_mirroring.models.capb import capb_from_checkpoint

    with pytest.raises(RuntimeError, match="model_state"):
        capb_from_checkpoint({"target_sample_rate": 96_000})


def test_training_config_loads_initial_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "training.yaml"
    checkpoint_path = tmp_path / "initial.pt"
    config_path.write_text(f"initial_checkpoint: {checkpoint_path}\n")

    config = load_capb_training_config(config_path)

    assert config.initial_checkpoint == checkpoint_path


def test_training_config_loads_checkpoint_interval(tmp_path: Path) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text("checkpoint_interval_epochs: 5\n")
    assert load_capb_training_config(config_path).checkpoint_interval_epochs == 5

    config_path.write_text("checkpoint_interval_epochs: -1\n")
    with pytest.raises(ValueError, match="checkpoint_interval_epochs"):
        load_capb_training_config(config_path)


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
