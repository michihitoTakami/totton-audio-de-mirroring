"""Tests for CAPB comparison-plot visibility."""

from scripts.report_capb_distortion import (
    _comparison_zorder as distortion_zorder,
)
from scripts.report_capb_impulse import _comparison_zorder as impulse_zorder


def test_capb_distortion_series_is_above_all_references() -> None:
    """CAPB distortion lines must remain visible where traces overlap."""
    references = ("ideal", "bessel", "sharp", "gentle")
    assert all(
        distortion_zorder("capb") > distortion_zorder(name) for name in references
    )


def test_capb_impulse_series_is_above_all_references() -> None:
    """CAPB impulse lines must remain visible where traces overlap."""
    references = ("bessel", "sharp", "gentle")
    assert all(impulse_zorder("capb") > impulse_zorder(name) for name in references)
