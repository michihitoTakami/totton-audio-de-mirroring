"""Tests for coherent-line distortion diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from totton_audio_de_mirroring.evaluation.distortion import (
    added_am_sideband_db,
    ccif_imd_db,
    relative_line_levels_db,
    smpte_imd_db,
    thd_db,
    tone_amplitude,
)

SAMPLE_RATE = 48_000


def _time_axis() -> np.ndarray:
    """Return a coherent one-second test time axis."""
    return np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE


def test_tone_amplitude_recovers_peak_amplitude() -> None:
    """A coherent projection recovers the sinusoid peak amplitude."""
    time = _time_axis()
    waveform = 0.25 * np.sin(2.0 * np.pi * 1_000.0 * time)
    assert tone_amplitude(waveform, SAMPLE_RATE, 1_000.0) == pytest.approx(0.25)


def test_thd_reports_injected_second_harmonic() -> None:
    """THD includes an injected audio-band harmonic at its known ratio."""
    time = _time_axis()
    waveform = np.sin(2.0 * np.pi * 1_000.0 * time)
    waveform += 0.01 * np.sin(2.0 * np.pi * 2_000.0 * time)
    assert thd_db(waveform, SAMPLE_RATE) == pytest.approx(-40.0, abs=1.0e-8)


def test_smpte_imd_combines_symmetric_products() -> None:
    """SMPTE IMD root-sum-squares the injected sideband pair."""
    time = _time_axis()
    waveform = 0.25 * np.sin(2.0 * np.pi * 7_000.0 * time)
    for frequency_hz in (6_940.0, 7_060.0):
        waveform += 0.0025 * np.sin(2.0 * np.pi * frequency_hz * time)
    expected_db = 20.0 * np.log10(np.sqrt(2.0) * 0.0025 / 0.25)
    measured_db = smpte_imd_db(waveform, SAMPLE_RATE, max_sideband_order=1)
    assert measured_db == pytest.approx(expected_db, abs=1.0e-8)


def test_ccif_imd_reports_difference_product() -> None:
    """CCIF IMD detects an injected 1 kHz difference product."""
    time = _time_axis()
    waveform = 0.25 * np.sin(2.0 * np.pi * 19_000.0 * time)
    waveform += 0.25 * np.sin(2.0 * np.pi * 20_000.0 * time)
    waveform += 0.0025 * np.sin(2.0 * np.pi * 1_000.0 * time)
    expected_db = 20.0 * np.log10(0.0025 / np.hypot(0.25, 0.25))
    assert ccif_imd_db(waveform, SAMPLE_RATE) == pytest.approx(expected_db, abs=1.0e-8)


def test_added_am_sideband_detects_second_order_pair() -> None:
    """The AM diagnostic ignores wanted first-order and finds added order two."""
    time = _time_axis()
    carrier_hz = 10_000.0
    modulation_hz = 37.0
    waveform = np.sin(2.0 * np.pi * carrier_hz * time)
    waveform += 0.25 * np.sin(2.0 * np.pi * (carrier_hz - modulation_hz) * time)
    waveform += 0.25 * np.sin(2.0 * np.pi * (carrier_hz + modulation_hz) * time)
    waveform += 0.001 * np.sin(2.0 * np.pi * (carrier_hz + 2 * modulation_hz) * time)
    assert added_am_sideband_db(waveform, SAMPLE_RATE) == pytest.approx(
        -60.0, abs=1.0e-8
    )


def test_relative_line_levels_preserve_requested_order() -> None:
    """Line-level output remains aligned with the requested frequencies."""
    time = _time_axis()
    waveform = np.sin(2.0 * np.pi * 1_000.0 * time)
    waveform += 0.1 * np.sin(2.0 * np.pi * 2_000.0 * time)
    levels = relative_line_levels_db(waveform, SAMPLE_RATE, (2_000.0, 1_000.0), 1_000.0)
    assert levels == pytest.approx((-20.0, 0.0), abs=1.0e-8)


@pytest.mark.parametrize("bad_signal", [np.array([]), np.zeros((1, 4))])
def test_tone_amplitude_rejects_invalid_waveforms(bad_signal: np.ndarray) -> None:
    """The coherent projection rejects empty and non-mono inputs."""
    with pytest.raises(ValueError, match="non-empty 1D"):
        tone_amplitude(bad_signal, SAMPLE_RATE, 1_000.0)
