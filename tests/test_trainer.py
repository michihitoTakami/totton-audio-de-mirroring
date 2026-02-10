"""Tests for training utilities."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from totton_audio_de_mirroring.training.losses import LossWeights, STFTLossConfig
from totton_audio_de_mirroring.training.runtime import compute_lowband_metrics
from totton_audio_de_mirroring.training.trainer import (
    TrainingConfig,
    _compute_batch_metrics,
    load_training_config,
    select_device,
    train_stage1,
)


class _DummyNMSE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.9))
        self.sample_rate = 88_200.0
        self.cutoff_hz = 20_000.0

    def forward_highband(self, high_band: torch.Tensor) -> torch.Tensor:
        return high_band * self.gain

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return signal


class _StaticBatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._batches[index]


def test_select_device_override_cpu() -> None:
    device = select_device(device_override="cpu")
    assert device.type == "cpu"


def test_select_device_require_cuda_raises_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU training is required"):
        _ = select_device(require_cuda=True)


def test_training_config_from_dict_parses_weights() -> None:
    config = TrainingConfig.from_dict(
        {
            "epochs": 2,
            "learning_rate": 1.0e-3,
            "loss_weights": {
                "mask": 2.0,
                "stft": 0.5,
                "preserve": 1.5,
                "energy": 0.2,
                "edge": 0.05,
                "step": 0.07,
            },
            "ringing_loss_config": {
                "edge_weight_cap": 3.0,
                "step_window_size": 15,
                "eps": 1.0e-7,
            },
        }
    )
    assert config.epochs == 2
    assert config.loss_weights.mask == 2.0
    assert config.loss_weights.stft == 0.5
    assert config.loss_weights.edge == 0.05
    assert config.loss_weights.step == 0.07
    assert config.ringing_loss_config.edge_weight_cap == 3.0
    assert config.ringing_loss_config.step_window_size == 15


def test_training_config_defaults_follow_teacher_type() -> None:
    raw_config = TrainingConfig.from_dict({"teacher_type": "raw_88k2"})
    bessel_config = TrainingConfig.from_dict({"teacher_type": "bessel_88k2"})
    assert raw_config.energy_cap == pytest.approx(1.0e-3)
    assert bessel_config.energy_cap == pytest.approx(1.0)


def test_training_config_applies_hb_and_preserve_weight_aliases() -> None:
    config = TrainingConfig.from_dict(
        {
            "teacher_type": "raw_88k2",
            "hb_loss_weight": 1.25,
            "preserve_lb_weight": 1.6,
        }
    )
    assert config.loss_weights.mask == pytest.approx(1.25)
    assert config.loss_weights.stft == pytest.approx(1.25)
    assert config.loss_weights.preserve == pytest.approx(1.6)


def test_training_config_rejects_invalid_teacher_type() -> None:
    with pytest.raises(ValueError, match="teacher_type must be one of"):
        _ = TrainingConfig.from_dict({"teacher_type": "invalid"})


def test_training_config_from_dict_parses_use_amp_string_false() -> None:
    config = TrainingConfig.from_dict({"use_amp": "false"})
    assert config.use_amp is False


def test_load_training_config_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            epochs: 3
            learning_rate: 0.001
            require_cuda: false
            loss_weights:
              mask: 1.0
              stft: 1.0
              preserve: 1.0
              energy: 1.0
              edge: 0.05
              step: 0.05
            """
        ).strip(),
        encoding="utf-8",
    )
    config = load_training_config(config_path)
    assert config.epochs == 3
    assert config.learning_rate == pytest.approx(0.001)
    assert config.require_cuda is False


def test_load_training_config_uses_default_teacher_type_for_fallbacks(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text("require_cuda: false\n", encoding="utf-8")
    config = load_training_config(config_path, default_teacher_type="bessel_88k2")
    assert config.teacher_type == "bessel_88k2"
    assert config.energy_cap == pytest.approx(1.0)


def test_train_stage1_saves_best_and_last_checkpoints(tmp_path: Path) -> None:
    train_loader = _make_loader(num_steps=2)
    val_loader = _make_loader(num_steps=1)

    config = TrainingConfig(
        epochs=2,
        learning_rate=1.0e-3,
        use_amp=False,
        log_interval=100,
        require_cuda=False,
        loss_weights=LossWeights(edge=0.05, step=0.05),
        mask_config=STFTLossConfig(n_fft=64, hop_length=16, win_length=64),
        stft_configs=(STFTLossConfig(n_fft=64, hop_length=16, win_length=64),),
    )

    model = _DummyNMSE()
    result = train_stage1(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        checkpoint_dir=tmp_path,
    )

    assert result.last_checkpoint == tmp_path / "stage1_last.pt"
    assert result.best_checkpoint == tmp_path / "stage1_best.pt"
    assert result.last_checkpoint.exists()
    assert result.best_checkpoint.exists()
    assert len(result.train_history) == 2
    assert len(result.val_history) == 2
    assert result.train_history[0].edge >= 0.0
    assert result.train_history[0].step >= 0.0
    contrib_sum = (
        result.train_history[0].contrib_mask
        + result.train_history[0].contrib_stft
        + result.train_history[0].contrib_preserve
        + result.train_history[0].contrib_energy
        + result.train_history[0].contrib_edge
        + result.train_history[0].contrib_step
    )
    assert contrib_sum == pytest.approx(1.0, rel=1.0e-4, abs=1.0e-4)

    state = torch.load(result.last_checkpoint, map_location="cpu", weights_only=False)
    assert "model_state" in state
    assert "optimizer_state" in state
    assert "scheduler_state" in state
    assert "training_config" in state
    assert "device" in state
    assert state["training_config"]["loss_weights"]["edge"] == pytest.approx(0.05)
    assert state["training_config"]["loss_weights"]["step"] == pytest.approx(0.05)


def test_compute_lowband_metrics_normalizes_by_lowband_bins_only() -> None:
    stft_config = STFTLossConfig(n_fft=64, hop_length=16, win_length=64)
    sample_rate = 88_200.0
    cutoff_hz = 5_000.0
    length = 512
    time = torch.arange(length, dtype=torch.float32) / sample_rate
    x_full = torch.sin(2.0 * torch.pi * 1_000.0 * time).unsqueeze(0)
    y_full = (1.5 * x_full).clone()

    mag_mae, phase_mae = compute_lowband_metrics(
        x_full=x_full,
        y_full=y_full,
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        stft_config=stft_config,
    )

    window = torch.hann_window(64, periodic=True, dtype=x_full.dtype)
    x_spec = torch.stft(
        x_full,
        n_fft=64,
        hop_length=16,
        win_length=64,
        center=True,
        window=window,
        return_complex=True,
    )
    y_spec = torch.stft(
        y_full,
        n_fft=64,
        hop_length=16,
        win_length=64,
        center=True,
        window=window,
        return_complex=True,
    )
    freqs = torch.linspace(0.0, sample_rate / 2.0, x_spec.shape[-2])
    low_mask = (freqs <= cutoff_hz).view(1, -1, 1).expand_as(x_spec.real)
    expected_mag_mae = torch.mean(
        torch.abs(y_spec.abs() - x_spec.abs())[low_mask]
    ).item()
    expected_phase_mae = torch.mean(
        torch.abs(torch.angle(y_spec) - torch.angle(x_spec))[low_mask]
    ).item()

    assert mag_mae == pytest.approx(expected_mag_mae, rel=1.0e-5, abs=1.0e-7)
    assert phase_mae == pytest.approx(expected_phase_mae, rel=1.0e-5, abs=1.0e-7)


def test_compute_batch_metrics_clamps_extreme_energy_ratio_to_finite() -> None:
    model = _DummyNMSE()
    mask_config = STFTLossConfig(n_fft=64, hop_length=16, win_length=64)

    hb_in = torch.ones(2, 256, dtype=torch.float32)
    hb_pred = torch.zeros(2, 256, dtype=torch.float32)
    stft = torch.stft(
        hb_in,
        n_fft=64,
        hop_length=16,
        win_length=64,
        window=torch.hann_window(64),
        return_complex=True,
    )
    mirror_mask = torch.ones(2, stft.shape[-2], stft.shape[-1], dtype=torch.float32)

    metrics = _compute_batch_metrics(
        model=model,
        batch={},
        hb_in=hb_in,
        hb_pred=hb_pred,
        mirror_mask=mirror_mask,
        device=torch.device("cpu"),
        mask_config=mask_config,
        energy_cap=1.0,
        compute_low_band=False,
    )

    assert torch.isfinite(torch.tensor(metrics["mirror_reduction_db"]))
    assert torch.isfinite(torch.tensor(metrics["touch_l1"]))
    assert torch.isfinite(torch.tensor(metrics["energy_cap_violation"]))


def test_train_stage1_raises_when_non_finite_output_detected(tmp_path: Path) -> None:
    batch = _make_batch(batch_size=2, length=256)
    high_band = batch["high_band"].clone()
    high_band[0, 0] = torch.nan
    batch["high_band"] = high_band

    dataset = _StaticBatchDataset([batch])
    train_loader = DataLoader(dataset, batch_size=None)
    config = TrainingConfig(
        epochs=1,
        learning_rate=1.0e-3,
        use_amp=False,
        log_interval=100,
        require_cuda=False,
        mask_config=STFTLossConfig(n_fft=64, hop_length=16, win_length=64),
        stft_configs=(STFTLossConfig(n_fft=64, hop_length=16, win_length=64),),
    )

    with pytest.raises(RuntimeError, match="Non-finite model output detected"):
        _ = train_stage1(
            model=_DummyNMSE(),
            train_dataloader=train_loader,
            config=config,
            checkpoint_dir=tmp_path,
        )

    assert (tmp_path / "stage1_emergency.pt").exists()


def _make_loader(num_steps: int) -> DataLoader[dict[str, Any]]:
    batch = _make_batch(batch_size=2, length=256)
    dataset = _StaticBatchDataset([batch for _ in range(num_steps)])
    return DataLoader(dataset, batch_size=None)


def _make_batch(batch_size: int, length: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    high_band = torch.randn(batch_size, length, generator=generator)
    hb_target = high_band * 0.8

    stft = torch.stft(
        high_band,
        n_fft=64,
        hop_length=16,
        win_length=64,
        window=torch.hann_window(64),
        return_complex=True,
    )
    mirror_mask = torch.ones(batch_size, stft.shape[-2], stft.shape[-1])
    return {
        "high_band": high_band,
        "hb_target": hb_target,
        "mirror_mask": mirror_mask,
        "x_full": high_band,
    }
