"""Tests for the generative high-band target and dataset target_mode."""

from __future__ import annotations

import numpy as np
import pytest

from totton_audio_de_mirroring.data.mirror_detection import (
    build_generative_hb_target,
    project_teacher_hb_target,
)

SR = 88_200


def _hb_signals(length: int = 8820) -> tuple[np.ndarray, np.ndarray]:
    """Return (degraded HB input, richer native teacher HB) at 88.2kHz."""
    t = np.arange(length) / SR
    # Degraded input HB: weak content near 21kHz (what survives 44.1k).
    hb_in = 0.02 * np.sin(2 * np.pi * 21_000 * t)
    # Native teacher HB: genuine strong content at 30kHz (lost in 44.1k).
    teacher = 0.02 * np.sin(2 * np.pi * 21_000 * t) + 0.2 * np.sin(
        2 * np.pi * 30_000 * t
    )
    return hb_in.astype(np.float64), teacher.astype(np.float64)


def test_generative_target_length_matches_input() -> None:
    hb_in, teacher = _hb_signals()
    target = build_generative_hb_target(hb_in, teacher, SR)
    assert target.shape[0] == hb_in.shape[0]
    assert np.all(np.isfinite(target))


def test_generative_target_exceeds_suppression_target_energy() -> None:
    """Generation should retain real teacher HB energy that suppression discards."""
    hb_in, teacher = _hb_signals()
    gen = build_generative_hb_target(hb_in, teacher, SR, energy_cap=1.0)
    sup = project_teacher_hb_target(hb_in, teacher, SR, energy_cap=1.0)
    gen_energy = float(np.mean(gen**2))
    sup_energy = float(np.mean(sup**2))
    # The suppression target is capped to the (weak) input magnitude, so it
    # cannot contain the 30kHz teacher energy that the generative target keeps.
    assert gen_energy > sup_energy * 2.0


def test_generative_target_respects_energy_cap() -> None:
    hb_in, teacher = _hb_signals()
    cap = 1.0e-3
    target = build_generative_hb_target(hb_in, teacher, SR, energy_cap=cap)
    # crude time-domain energy proxy must remain bounded
    assert float(np.mean(target**2)) <= cap * 5.0


def test_generative_target_rejects_shape_mismatch() -> None:
    hb_in, _ = _hb_signals()
    with pytest.raises(ValueError):
        build_generative_hb_target(hb_in, hb_in[:-1], SR)


def test_dataset_generate_mode_matches_contract() -> None:
    from totton_audio_de_mirroring.data.dataset import MirrorSuppressionDataset
    from totton_audio_de_mirroring.data.pipeline_config import DataPipelineConfig

    config = DataPipelineConfig(
        num_samples=2,
        source_sample_rate=44_100,
        target_sample_rate=88_200,
        source_duration_sec=1.0,
        chunk_duration_sec=0.25,
        seed=1234,
        teacher_type="raw_88k2",
    )
    dataset = MirrorSuppressionDataset(config, target_mode="generate")
    sample = dataset[0]
    assert sample["high_band"].shape[-1] == int(0.25 * 88_200)
    assert sample["hb_target"].shape == sample["high_band"].shape
    assert sample["mirror_mask"].ndim == 2


def test_dataset_rejects_invalid_target_mode() -> None:
    from totton_audio_de_mirroring.data.dataset import MirrorSuppressionDataset
    from totton_audio_de_mirroring.data.pipeline_config import DataPipelineConfig

    config = DataPipelineConfig(num_samples=2, teacher_type="raw_88k2")
    with pytest.raises(ValueError, match="target_mode"):
        MirrorSuppressionDataset(config, target_mode="invalid")
