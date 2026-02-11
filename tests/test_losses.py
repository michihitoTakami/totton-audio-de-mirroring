"""Tests for training loss functions."""

import pytest
import torch

from totton_audio_de_mirroring.training.losses import (
    LossWeights,
    RingingLossConfig,
    STFTLossConfig,
    compute_loss_contribution_ratios,
    compute_losses,
    energy_cap_loss,
    mask_loss,
    preserve_loss,
    ringing_edge_loss,
    ringing_step_loss,
    strict_energy_cap_loss,
    subtractive_suppression_loss,
)


def _make_stft_config() -> STFTLossConfig:
    return STFTLossConfig(n_fft=128, hop_length=32, win_length=128)


def _make_mirror_mask(signal: torch.Tensor, config: STFTLossConfig) -> torch.Tensor:
    window = torch.hann_window(config.win_length, periodic=True)
    stft = torch.stft(
        signal,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=config.center,
        return_complex=True,
    )
    return torch.zeros_like(torch.abs(stft))


def test_mask_loss_modes() -> None:
    pred = torch.tensor([[[0.2, 0.5]]])
    target = torch.tensor([[[0.0, 1.0]]])
    loss_l1 = mask_loss(pred, target, mode="l1")
    loss_l2 = mask_loss(pred, target, mode="l2")
    assert loss_l1 > 0.0
    assert loss_l2 > 0.0
    assert loss_l2 != loss_l1


def test_preserve_loss_respects_mirror_mask() -> None:
    pred_mag = torch.ones(1, 2, 3)
    input_mag = torch.zeros(1, 2, 3)
    mirror_mask = torch.ones(2, 3)
    loss = preserve_loss(pred_mag, input_mag, mirror_mask)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_preserve_loss_accepts_mismatched_mask_grid_by_resizing() -> None:
    pred_mag = torch.ones(2, 4, 5)
    input_mag = torch.zeros(2, 4, 5)
    mirror_mask = torch.ones(2, 3, 2)

    loss = preserve_loss(pred_mag, input_mag, mirror_mask)

    assert torch.isfinite(loss)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_preserve_loss_accepts_bool_mismatched_mask_grid() -> None:
    pred_mag = torch.ones(1, 4, 5)
    input_mag = torch.zeros(1, 4, 5)
    mirror_mask = torch.ones(1, 2, 3, dtype=torch.bool)

    loss = preserve_loss(pred_mag, input_mag, mirror_mask)

    assert torch.isfinite(loss)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_compute_losses_has_grad_flow() -> None:
    torch.manual_seed(0)
    batch = 2
    length = 256
    hb_in = torch.randn(batch, length)
    hb_target = hb_in * 0.8

    scale = torch.nn.Parameter(torch.tensor(0.9))
    hb_pred = hb_in * scale

    mask_config = _make_stft_config()
    stft_configs = (_make_stft_config(),)
    mirror_mask = _make_mirror_mask(hb_in, mask_config)

    terms = compute_losses(
        hb_in,
        hb_target,
        hb_pred,
        mirror_mask,
        mask_config=mask_config,
        stft_configs=stft_configs,
        weights=LossWeights(),
        energy_cap=1000.0,
    )
    terms.total.backward()

    assert scale.grad is not None
    assert torch.abs(scale.grad).item() > 0.0


def test_energy_cap_loss_penalizes_excess() -> None:
    pred_mag = torch.ones(1, 4, 5)
    loss = energy_cap_loss(pred_mag, energy_cap=0.5)
    assert loss > 0.0


def test_subtractive_suppression_loss_penalizes_only_additive_gain() -> None:
    input_mag = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
    pred_mag = torch.tensor([[[1.5, 1.0]]], dtype=torch.float32)
    loss = subtractive_suppression_loss(pred_mag, input_mag)
    assert loss == pytest.approx(0.25)


def test_strict_energy_cap_loss_is_stronger_than_linear_energy_loss() -> None:
    pred_mag = torch.ones(1, 4, 5) * 2.0
    linear = energy_cap_loss(pred_mag, energy_cap=1.0)
    strict = strict_energy_cap_loss(pred_mag, energy_cap=1.0)
    assert strict > linear


def test_ringing_aux_losses_are_zero_when_prediction_matches_target() -> None:
    signal = torch.tensor([[0.0, 0.2, 0.8, 1.0, 0.9, 0.5]], dtype=torch.float32)
    edge_loss = ringing_edge_loss(signal, signal)
    step_loss = ringing_step_loss(signal, signal)
    assert torch.isclose(edge_loss, torch.tensor(0.0))
    assert torch.isclose(step_loss, torch.tensor(0.0))


def test_ringing_aux_losses_increase_for_edge_distortion() -> None:
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    pred = torch.tensor([[0.0, 0.2, 0.6, 0.9, 1.1, 1.2]], dtype=torch.float32)

    config = RingingLossConfig(step_window_size=3)
    edge_loss = ringing_edge_loss(pred, target, config=config)
    step_loss = ringing_step_loss(pred, target, config=config)
    assert edge_loss > 0.0
    assert step_loss > 0.0


def test_ringing_aux_losses_stay_finite_for_fp16_silence() -> None:
    pred = torch.nn.Parameter(
        torch.tensor([[0.0, 1.0e-3, -1.0e-3, 0.0]], dtype=torch.float16)
    )
    target = torch.zeros_like(pred)
    config = RingingLossConfig(step_window_size=3, eps=1.0e-8)

    edge_loss = ringing_edge_loss(pred, target, config=config)
    step_loss = ringing_step_loss(pred, target, config=config)
    total = edge_loss + step_loss
    total.backward()

    assert torch.isfinite(edge_loss)
    assert torch.isfinite(step_loss)
    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))


def test_ringing_aux_losses_compute_in_fp32_for_fp16_inputs() -> None:
    pred = torch.tensor([[0.0, 1.0e-4, -1.0e-4, 0.0]], dtype=torch.float16)
    target = torch.zeros_like(pred)
    config = RingingLossConfig(step_window_size=3, eps=1.0e-8)

    edge_loss = ringing_edge_loss(pred, target, config=config)
    step_loss = ringing_step_loss(pred, target, config=config)

    assert edge_loss.dtype == torch.float32
    assert step_loss.dtype == torch.float32
    assert torch.isfinite(edge_loss)
    assert torch.isfinite(step_loss)


def test_ringing_aux_losses_preserve_fp64_precision() -> None:
    pred = torch.tensor([[0.0, 1.0e-6, -2.0e-6, 0.0]], dtype=torch.float64)
    target = torch.zeros_like(pred)
    config = RingingLossConfig(step_window_size=3, eps=1.0e-12)

    edge_loss = ringing_edge_loss(pred, target, config=config)
    step_loss = ringing_step_loss(pred, target, config=config)

    assert edge_loss.dtype == torch.float64
    assert step_loss.dtype == torch.float64
    assert torch.isfinite(edge_loss)
    assert torch.isfinite(step_loss)


@pytest.mark.parametrize(
    ("pred_dtype", "target_dtype"),
    (
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
    ),
)
def test_ringing_aux_losses_promote_to_higher_precision_for_mixed_inputs(
    pred_dtype: torch.dtype,
    target_dtype: torch.dtype,
) -> None:
    pred = torch.tensor([[0.0, 1.0e-4, -1.0e-4, 0.0]], dtype=pred_dtype)
    target = torch.tensor([[0.0, 1.0e-6, -2.0e-6, 0.0]], dtype=target_dtype)
    config = RingingLossConfig(step_window_size=3, eps=1.0e-12)

    edge_loss = ringing_edge_loss(pred, target, config=config)
    step_loss = ringing_step_loss(pred, target, config=config)

    assert edge_loss.dtype == torch.float64
    assert step_loss.dtype == torch.float64
    assert torch.isfinite(edge_loss)
    assert torch.isfinite(step_loss)


def test_ringing_step_loss_stays_finite_for_long_fp16_sequence() -> None:
    length = 22_050
    values = torch.zeros(1, length, dtype=torch.float16)
    values[:, 1::2] = 1.0e-3
    values[:, 2::2] = -1.0e-3
    pred = torch.nn.Parameter(values)
    target = torch.zeros_like(pred)
    config = RingingLossConfig(step_window_size=33, eps=1.0e-8)

    step_loss = ringing_step_loss(pred, target, config=config)
    step_loss.backward()

    assert torch.isfinite(step_loss)
    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))


def test_compute_loss_contribution_ratios_sums_to_one() -> None:
    hb_in = torch.randn(1, 128)
    hb_target = hb_in * 0.5
    hb_pred = hb_in * 0.8
    mask_config = _make_stft_config()
    mirror_mask = _make_mirror_mask(hb_in, mask_config)

    terms = compute_losses(
        hb_in,
        hb_target,
        hb_pred,
        mirror_mask,
        mask_config=mask_config,
        stft_configs=(mask_config,),
        weights=LossWeights(edge=0.1, step=0.1),
        energy_cap=10.0,
    )
    contrib = compute_loss_contribution_ratios(
        terms,
        LossWeights(edge=0.1, step=0.1),
    )
    total = (
        contrib.mask
        + contrib.stft
        + contrib.preserve
        + contrib.energy
        + contrib.subtract
        + contrib.cap_strict
        + contrib.edge
        + contrib.step
    )
    assert total == pytest.approx(1.0, rel=1.0e-6, abs=1.0e-6)
