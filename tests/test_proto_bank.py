"""Tests for the CAPB prototype bank."""

import numpy as np
import pytest
from scipy import signal as sp_signal

from totton_audio_de_mirroring.models.proto_bank import (
    DEFAULT_PROTOTYPE_SPECS,
    BesselMagnitudePrototypeSpec,
    KaiserPrototypeSpec,
    blend_modulation_bounds,
    build_prototype_bank,
    design_bessel_magnitude_prototype,
    design_kaiser_prototype,
    summarize_bank,
    upsample_with_kernel,
    validate_bank,
)

TARGET_SR = 88_200
SOURCE_SR = 44_100


@pytest.fixture(scope="module")
def bank():
    """Build the default prototype bank once per module."""
    return build_prototype_bank()


def test_bank_has_expected_prototypes(bank) -> None:
    assert bank.names == ("sharp", "mid", "gentle")
    assert bank.kernels.shape[0] == 3
    assert bank.kernels.shape[1] % 2 == 1


def test_bank_kernels_are_symmetric(bank) -> None:
    """Symmetric kernels certify linear phase (flat group delay)."""
    flipped = bank.kernels[:, ::-1]
    np.testing.assert_allclose(bank.kernels, flipped, atol=1e-15)


def test_bank_group_delay_matches_length(bank) -> None:
    assert bank.group_delay_samples == (bank.kernels.shape[1] - 1) // 2


def test_validate_bank_passes(bank) -> None:
    results = validate_bank(bank)
    assert results["kernel_symmetry_rel"] <= 1e-12
    assert results["kaiser_passband_match_db"] <= -70.0


def test_blend_modulation_bounds_monotone(bank) -> None:
    """Response spread can only grow toward the transition band."""
    bounds = blend_modulation_bounds(bank)
    values = [bounds[key] for key in sorted(bounds, key=lambda k: int(k.split("_")[1]))]
    assert values == sorted(values)


def test_sharp_prototype_suppresses_images(bank) -> None:
    """A 19 kHz tone's 25.1 kHz image must vanish through the sharp kernel."""
    time_axis = np.arange(SOURCE_SR // 2) / SOURCE_SR
    tone = 0.5 * np.sin(2.0 * np.pi * 19_000.0 * time_axis)
    output = upsample_with_kernel(tone, bank.kernels[0], bank.upsample_ratio)

    spectrum = np.abs(np.fft.rfft(output * np.hanning(output.size)))
    freqs = np.fft.rfftfreq(output.size, d=1.0 / TARGET_SR)
    tone_level = spectrum[np.argmin(np.abs(freqs - 19_000.0))]
    image_level = spectrum[np.argmin(np.abs(freqs - 25_100.0))]
    assert 20.0 * np.log10(image_level / tone_level) <= -80.0


def test_upsampler_preserves_amplitude(bank) -> None:
    """Gain normalization must preserve tone amplitude through 2x upsampling."""
    time_axis = np.arange(SOURCE_SR // 2) / SOURCE_SR
    tone = 0.5 * np.sin(2.0 * np.pi * 1_000.0 * time_axis)
    for index in range(len(bank.names)):
        output = upsample_with_kernel(tone, bank.kernels[index], bank.upsample_ratio)
        core = output[output.size // 4 : -output.size // 4]
        assert np.max(np.abs(core)) == pytest.approx(0.5, rel=2e-3)


def test_gentle_matches_bessel_magnitude() -> None:
    """The gentle prototype must track the Bessel reference magnitude."""
    spec = BesselMagnitudePrototypeSpec(
        name="gentle", num_taps=101, cutoff_hz=20_000.0, order=6
    )
    taps = design_bessel_magnitude_prototype(spec)

    b, a = sp_signal.bessel(
        6, 20_000.0, btype="lowpass", output="ba", norm="phase", fs=TARGET_SR
    )
    freq_grid = np.linspace(0.0, TARGET_SR / 2, 512)
    _, bessel_resp = sp_signal.freqz(b, a, worN=freq_grid, fs=TARGET_SR)
    _, fir_resp = sp_signal.freqz(taps, worN=freq_grid, fs=TARGET_SR)
    deviation = np.max(np.abs(np.abs(fir_resp) / 2.0 - np.abs(bessel_resp)))
    assert 20.0 * np.log10(deviation) <= -50.0


def test_upsample_output_length(bank) -> None:
    signal = np.random.default_rng(0).normal(size=1_000)
    output = upsample_with_kernel(signal, bank.kernels[0], 2)
    assert output.shape == (2_000,)


def test_upsample_rejects_invalid_inputs(bank) -> None:
    with pytest.raises(ValueError, match="non-empty 1D"):
        upsample_with_kernel(np.zeros((2, 2)), bank.kernels[0], 2)
    with pytest.raises(ValueError, match="odd length"):
        upsample_with_kernel(np.zeros(16), np.zeros(10), 2)
    with pytest.raises(ValueError, match="upsample_ratio"):
        upsample_with_kernel(np.zeros(16), bank.kernels[0], 0)


def test_kaiser_spec_validation() -> None:
    with pytest.raises(ValueError, match="passband_edge_hz"):
        design_kaiser_prototype(KaiserPrototypeSpec("bad", 23_000.0, 22_000.0, 80.0))
    with pytest.raises(ValueError, match="Nyquist"):
        design_kaiser_prototype(KaiserPrototypeSpec("bad", 40_000.0, 45_000.0, 80.0))


def test_bessel_spec_validation() -> None:
    with pytest.raises(ValueError, match="odd"):
        design_bessel_magnitude_prototype(
            BesselMagnitudePrototypeSpec("bad", 100, 20_000.0, 6)
        )


def test_summarize_bank_reports_all_prototypes(bank) -> None:
    summary = summarize_bank(bank)
    assert set(summary) == set(bank.names)
    assert summary["sharp"]["image_band_max_db"] <= -90.0
    assert summary["sharp"]["passband_dev_db"] <= -100.0


def test_specs_default_tuple_is_consistent() -> None:
    names = [spec.name for spec in DEFAULT_PROTOTYPE_SPECS]
    assert names == ["sharp", "mid", "gentle"]
