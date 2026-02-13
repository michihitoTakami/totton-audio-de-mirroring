"""Tests for distillation training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from totton_audio_de_mirroring.training.distillation import (
    DistillationConfig,
    _distillation_loss,
    apply_global_magnitude_pruning,
    load_distillation_config,
    train_stage1_distillation,
)


class _IdentityTeacher(nn.Module):
    def forward_highband(self, high_band: torch.Tensor) -> torch.Tensor:
        return high_band

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return signal


class _TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.5))

    def forward_highband(self, high_band: torch.Tensor) -> torch.Tensor:
        return high_band * self.gain

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return signal


class _SingleBatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, batch: dict[str, torch.Tensor], repeats: int) -> None:
        self._batch = batch
        self._repeats = repeats

    def __len__(self) -> int:
        return self._repeats

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._batch


def test_distillation_config_from_yaml(tmp_path: Path) -> None:
    """YAML config should be parsed into DistillationConfig."""
    config_path = tmp_path / "distill.yaml"
    config_path.write_text(
        "epochs: 3\nlearning_rate: 0.0002\nrequire_cuda: false\n",
        encoding="utf-8",
    )
    config = load_distillation_config(config_path)
    assert isinstance(config, DistillationConfig)
    assert config.epochs == 3
    assert config.learning_rate == pytest.approx(0.0002)
    assert config.require_cuda is False


def test_distillation_config_defaults_follow_teacher_type() -> None:
    raw_config = DistillationConfig.from_dict({"teacher_type": "raw_88k2"})
    bessel_config = DistillationConfig.from_dict({"teacher_type": "bessel_88k2"})
    assert raw_config.energy_cap == pytest.approx(1.0e-3)
    assert raw_config.task_loss_weights.subtract == pytest.approx(1.0)
    assert raw_config.task_loss_weights.cap_strict == pytest.approx(4.0)
    assert bessel_config.energy_cap == pytest.approx(1.0)
    assert bessel_config.task_loss_weights.subtract == pytest.approx(0.0)
    assert bessel_config.task_loss_weights.cap_strict == pytest.approx(0.0)


def test_distillation_config_applies_hb_and_preserve_weight_aliases() -> None:
    config = DistillationConfig.from_dict(
        {
            "teacher_type": "raw_88k2",
            "hb_loss_weight": 1.4,
            "preserve_lb_weight": 1.8,
        }
    )
    assert config.task_loss_weights.mask == pytest.approx(1.4)
    assert config.task_loss_weights.stft == pytest.approx(1.4)
    assert config.task_loss_weights.preserve == pytest.approx(1.8)


def test_distillation_config_rejects_invalid_teacher_type() -> None:
    with pytest.raises(ValueError, match="teacher_type must be one of"):
        _ = DistillationConfig.from_dict({"teacher_type": "invalid"})


def test_load_distillation_config_uses_default_teacher_type_for_fallbacks(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "distill.yaml"
    config_path.write_text("require_cuda: false\n", encoding="utf-8")
    config = load_distillation_config(
        config_path,
        default_teacher_type="bessel_88k2",
    )
    assert config.teacher_type == "bessel_88k2"
    assert config.energy_cap == pytest.approx(1.0)


def test_apply_global_magnitude_pruning_reduces_nonzero_weights() -> None:
    """Pruning should increase zero-weight count."""
    model = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    nonzero_before = int(torch.count_nonzero(model.weight).item())
    apply_global_magnitude_pruning(model, amount=0.5)
    nonzero_after = int(torch.count_nonzero(model.weight).item())
    assert nonzero_after < nonzero_before


def test_train_stage1_distillation_runs_one_epoch(tmp_path: Path) -> None:
    """Distillation training should produce checkpoints on CPU."""
    batch = _make_batch(batch_size=2, length=256)
    loader = DataLoader(_SingleBatchDataset(batch, repeats=2), batch_size=None)
    config = DistillationConfig(
        epochs=1,
        learning_rate=1.0e-3,
        use_amp=False,
        require_cuda=False,
        log_interval=1,
        mask_config=_small_stft_config(),
        stft_configs=(_small_stft_config(),),
    )
    result = train_stage1_distillation(
        teacher=_IdentityTeacher(),
        student=_TinyStudent(),
        train_dataloader=loader,
        val_dataloader=loader,
        config=config,
        checkpoint_dir=tmp_path,
        model_config={"model_type": "nmse_light"},
    )
    assert result.last_checkpoint == tmp_path / "stage1_distill_raw88_last.pt"
    assert result.best_checkpoint == tmp_path / "stage1_distill_raw88_best.pt"
    assert result.last_checkpoint.exists()
    assert result.best_checkpoint.exists()
    assert len(result.train_history) == 1
    assert len(result.val_history) == 1
    assert result.train_history[0].distill_ratio >= 0.0
    assert result.val_history[0].distill_ratio >= 0.0


def test_train_stage1_distillation_respects_checkpoint_prefix(tmp_path: Path) -> None:
    """Distillation checkpoint names should follow teacher-aware prefixes."""
    batch = _make_batch(batch_size=2, length=256)
    loader = DataLoader(_SingleBatchDataset(batch, repeats=2), batch_size=None)
    config = DistillationConfig(
        epochs=1,
        learning_rate=1.0e-3,
        use_amp=False,
        require_cuda=False,
        log_interval=1,
        mask_config=_small_stft_config(),
        stft_configs=(_small_stft_config(),),
    )
    result = train_stage1_distillation(
        teacher=_IdentityTeacher(),
        student=_TinyStudent(),
        train_dataloader=loader,
        val_dataloader=loader,
        config=config,
        checkpoint_dir=tmp_path,
        checkpoint_prefix="stage1_distill_raw88",
    )
    assert result.last_checkpoint == tmp_path / "stage1_distill_raw88_last.pt"
    assert result.best_checkpoint == tmp_path / "stage1_distill_raw88_best.pt"
    assert result.last_checkpoint.exists()
    assert result.best_checkpoint.exists()


def test_train_stage1_distillation_default_prefix_uses_teacher_tag(
    tmp_path: Path,
) -> None:
    """Default checkpoint prefix should include teacher tag."""
    batch = _make_batch(batch_size=2, length=256)
    loader = DataLoader(_SingleBatchDataset(batch, repeats=2), batch_size=None)
    config = DistillationConfig(
        epochs=1,
        learning_rate=1.0e-3,
        use_amp=False,
        require_cuda=False,
        log_interval=1,
        teacher_type="bessel_88k2",
        mask_config=_small_stft_config(),
        stft_configs=(_small_stft_config(),),
    )
    result = train_stage1_distillation(
        teacher=_IdentityTeacher(),
        student=_TinyStudent(),
        train_dataloader=loader,
        val_dataloader=loader,
        config=config,
        checkpoint_dir=tmp_path,
    )
    assert result.last_checkpoint == tmp_path / "stage1_distill_bessel_last.pt"
    assert result.best_checkpoint == tmp_path / "stage1_distill_bessel_best.pt"
    assert result.last_checkpoint.exists()
    assert result.best_checkpoint.exists()


def test_distillation_loss_relative_normalization_increases_small_l2() -> None:
    hb_teacher = torch.full((2, 8), 1.0e-3, dtype=torch.float32)
    hb_student = torch.zeros((2, 8), dtype=torch.float32)
    raw = _distillation_loss(
        hb_student=hb_student,
        hb_teacher=hb_teacher,
        mode="l2",
        relative=False,
        eps=1.0e-8,
    )
    relative = _distillation_loss(
        hb_student=hb_student,
        hb_teacher=hb_teacher,
        mode="l2",
        relative=True,
        eps=1.0e-8,
    )
    assert float(raw.item()) < 1.0e-5
    assert float(relative.item()) > float(raw.item()) * 1.0e3


def test_distillation_config_parses_relative_fields() -> None:
    config = DistillationConfig.from_dict(
        {
            "teacher_type": "raw_88k2",
            "distillation_relative": False,
            "distillation_eps": 1.0e-6,
        }
    )
    assert config.distillation_relative is False
    assert config.distillation_eps == pytest.approx(1.0e-6)


def _small_stft_config() -> Any:
    from totton_audio_de_mirroring.training.losses import STFTLossConfig

    return STFTLossConfig(n_fft=64, hop_length=16, win_length=64)


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
    }
