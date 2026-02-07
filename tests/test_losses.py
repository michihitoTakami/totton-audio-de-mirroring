"""Tests for training loss functions."""

import torch

from totton_audio_de_mirroring.training.losses import (
    LossWeights,
    STFTLossConfig,
    compute_losses,
    energy_cap_loss,
    mask_loss,
    preserve_loss,
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
    loss = energy_cap_loss(pred_mag, energy_cap=1.0)
    assert loss > 0.0
