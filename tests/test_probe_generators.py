"""Tests for edge-rich and music-like signal generators."""

import numpy as np
import pytest

from totton_audio_de_mirroring.data.generator import generate_signal, list_signal_types
from totton_audio_de_mirroring.data.probe_generators import (
    PROBE_FAMILY_GENERATORS,
    generate_dense_square_wave,
    generate_isolated_click,
    generate_square_wave,
    generate_step_plateau,
)

SAMPLE_RATE = 88_200


def test_families_registered_in_generator() -> None:
    available = set(list_signal_types())
    assert set(PROBE_FAMILY_GENERATORS) <= available


@pytest.mark.parametrize("signal_type", sorted(PROBE_FAMILY_GENERATORS))
def test_all_families_generate_via_registry(signal_type: str) -> None:
    signal = generate_signal(
        signal_type, sample_rate=SAMPLE_RATE, duration_sec=0.5, seed=11
    )
    assert signal.shape == (SAMPLE_RATE // 2,)
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0


def test_square_wave_levels() -> None:
    signal = generate_square_wave(
        frequency_hz=100.0, sample_rate=SAMPLE_RATE, amplitude=0.5
    )
    assert set(np.round(np.unique(signal), 6)) <= {-0.5, 0.5}


def test_square_wave_duty_cycle() -> None:
    signal = generate_square_wave(frequency_hz=100.0, duty=0.3, sample_rate=SAMPLE_RATE)
    positive_fraction = float(np.mean(signal > 0))
    assert positive_fraction == pytest.approx(0.3, abs=0.02)


def test_dense_square_wave_covers_high_frequency_edges() -> None:
    signal = generate_dense_square_wave(
        frequency_hz=5_500.0,
        sample_rate=SAMPLE_RATE,
        duration_sec=0.1,
        amplitude=0.5,
    )
    transitions = np.count_nonzero(np.diff(signal))
    assert transitions == pytest.approx(1_100, abs=2)
    assert set(np.round(np.unique(signal), 6)) <= {-0.5, 0.5}


def test_step_plateau_has_flat_regions() -> None:
    rng = np.random.default_rng(3)
    signal = generate_step_plateau(sample_rate=SAMPLE_RATE, rng=rng)
    slope = np.abs(np.diff(signal))
    flat_fraction = float(np.mean(slope < 1e-9))
    assert flat_fraction > 0.5


def test_isolated_click_is_mostly_silent() -> None:
    rng = np.random.default_rng(4)
    signal = generate_isolated_click(sample_rate=SAMPLE_RATE, rng=rng)
    silent_fraction = float(np.mean(np.abs(signal) < 1e-12))
    assert silent_fraction > 0.99


def test_isolated_click_respects_center_fraction() -> None:
    signal = generate_isolated_click(
        sample_rate=SAMPLE_RATE,
        duration_sec=1.0,
        click_width_samples=3,
        center_fraction=0.25,
        rng=np.random.default_rng(4),
    )
    center = int(np.argmax(np.abs(signal)))
    assert center == pytest.approx(SAMPLE_RATE // 4, abs=2)


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        generate_square_wave(frequency_hz=50_000.0, sample_rate=SAMPLE_RATE)
    with pytest.raises(ValueError, match="duty"):
        generate_square_wave(duty=1.5)
    with pytest.raises(ValueError, match="RNG"):
        generate_step_plateau(rng=None)
    with pytest.raises(ValueError, match="center_fraction"):
        generate_isolated_click(center_fraction=1.5)
