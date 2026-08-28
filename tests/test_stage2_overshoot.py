"""Tests for Stage 2 overshoot evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from totton_audio_de_mirroring.stage2.overshoot import (
    cascade_upsample,
    evaluate_stage2_overshoot,
    load_stage_taps,
    upsample_2x_fir,
)


def test_load_stage_taps_reads_files(tmp_path: Path) -> None:
    """Test that stage tap files are loaded in numeric order."""
    (tmp_path / "stage1_taps.txt").write_text("1.0\n0.0\n", encoding="utf-8")
    (tmp_path / "stage2_taps.txt").write_text("0.5\n0.5\n", encoding="utf-8")

    taps = load_stage_taps(tmp_path, num_stages=2)

    assert len(taps) == 2
    assert np.allclose(taps[0], np.array([1.0, 0.0]))
    assert np.allclose(taps[1], np.array([0.5, 0.5]))


def test_cascade_upsample_stage_count_controls_length() -> None:
    """Test that each stage doubles output length."""
    signal = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    stage_taps = (np.array([1.0], dtype=np.float64), np.array([1.0], dtype=np.float64))

    output = cascade_upsample(signal, stage_taps)

    assert output.shape == (signal.shape[0] * 4,)


def test_upsample_2x_fir_preserves_settled_dc_level() -> None:
    """Unity-sum interpolation taps should preserve a settled input level.

    Physical Basis:
        Zero stuffing reduces baseband amplitude by two, so the interpolation
        stage must restore that factor after applying a unity-DC FIR.
    """
    signal = np.ones(64, dtype=np.float64)
    taps = np.array([0.5, 0.5], dtype=np.float64)

    output = upsample_2x_fir(signal, taps)

    np.testing.assert_allclose(output[32:], 1.0, rtol=0.0, atol=1e-12)


def test_default_stage_taps_match_hirate_contract() -> None:
    """Default taps should preserve the canonical linear-phase ladder."""
    config_dir = Path("cpp/configs")
    stage_taps = load_stage_taps(config_dir=config_dir, num_stages=3)

    assert [taps.size for taps in stage_taps] == [255, 63, 39]
    for taps in stage_taps:
        np.testing.assert_array_equal(taps, taps[::-1])
        assert np.sum(taps) == pytest.approx(1.0, abs=1.0e-7)


@pytest.mark.parametrize(
    ("source_sample_rate", "expected_output_rate"),
    [(88_200, 705_600), (96_000, 768_000)],
)
def test_evaluate_stage2_overshoot_with_known_taps(
    source_sample_rate: int, expected_output_rate: int
) -> None:
    """Both rate families should improve on the retired transient baseline."""
    stage_taps = load_stage_taps(config_dir=Path("cpp/configs"), num_stages=3)

    result = evaluate_stage2_overshoot(
        stage_taps=stage_taps, source_sample_rate=source_sample_rate
    )

    assert result.output_sample_rate == expected_output_rate
    assert np.isfinite(result.step.ratio)
    assert np.isfinite(result.square.ratio)
    assert 0.0 <= result.step.ratio < 0.15
    assert 0.0 <= result.square.ratio < 0.19


def test_load_stage_taps_missing_file_raises(tmp_path: Path) -> None:
    """Test missing tap file raises an explicit error."""
    (tmp_path / "stage1_taps.txt").write_text("1.0\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        _ = load_stage_taps(tmp_path, num_stages=2)
