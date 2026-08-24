"""Tests for the Bessel comparison SRC used by CAPB gates."""

import numpy as np
import pytest

from totton_audio_de_mirroring.data.reference import upsample_bessel_reference


def test_bessel_reference_doubles_length_without_mutating_input() -> None:
    """The reference path should return a new, finite 2x waveform.

    Physical Basis:
        CAPB ringing gates require a deterministic comparison signal aligned
        to the same target sample rate as the candidate output.
    """
    source = np.sin(2.0 * np.pi * 440.0 * np.arange(441) / 44_100.0)
    original = source.copy()

    output = upsample_bessel_reference(source, 44_100, 88_200, 20_000.0, 6)

    assert output.shape == (source.size * 2,)
    assert np.all(np.isfinite(output))
    assert np.array_equal(source, original)


@pytest.mark.parametrize(
    ("source_sr", "target_sr", "cutoff_hz", "order"),
    [
        (0, 88_200, 20_000.0, 6),
        (44_100, 48_000, 20_000.0, 6),
        (44_100, 88_200, 44_100.0, 6),
        (44_100, 88_200, 20_000.0, 0),
    ],
)
def test_bessel_reference_rejects_invalid_src_contract(
    source_sr: int,
    target_sr: int,
    cutoff_hz: float,
    order: int,
) -> None:
    """Invalid rates, cutoff, and order should fail at entry."""
    with pytest.raises(ValueError):
        upsample_bessel_reference(np.ones(16), source_sr, target_sr, cutoff_hz, order)
