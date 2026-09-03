"""Tests for the CAPB prototype bank."""

import numpy as np
import pytest
from scipy import signal as sp_signal

from totton_audio_de_mirroring.models.proto_bank import (
    DEFAULT_PROTOTYPE_SPECS,
    PROTOTYPE_SPECS_44K1,
    PROTOTYPE_SPECS_48K,
    RELEASE_PROTOTYPE_PROFILE,
    BesselMagnitudePrototypeSpec,
    KaiserPrototypeSpec,
    blend_modulation_bounds,
    build_prototype_bank,
    build_prototype_bank_for_profile,
    design_bessel_magnitude_prototype,
    design_kaiser_prototype,
    prototype_specs_for_target_rate,
    summarize_bank,
    supported_prototype_profiles,
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


@pytest.mark.parametrize("prototype_index", [0, 1, 2])
def test_delay_compensation_aligns_impulse(bank, prototype_index: int) -> None:
    """Every prototype must align an input impulse to the 2x timeline."""
    source = np.zeros(2_048, dtype=np.float64)
    source[800] = 1.0
    output = upsample_with_kernel(
        source, bank.kernels[prototype_index], bank.upsample_ratio
    )
    assert int(np.argmax(np.abs(output))) == 1_600


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
    with pytest.raises(ValueError, match="positive odd"):
        design_kaiser_prototype(
            KaiserPrototypeSpec("bad", 19_000.0, 23_000.0, 80.0, num_taps=1024)
        )


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


def test_default_specs_are_44k1_preset() -> None:
    """Regression guard: the 44.1k behavior is byte-identical to before."""
    assert DEFAULT_PROTOTYPE_SPECS is PROTOTYPE_SPECS_44K1
    assert prototype_specs_for_target_rate(88_200) is PROTOTYPE_SPECS_44K1
    assert prototype_specs_for_target_rate(96_000) is PROTOTYPE_SPECS_48K


def test_unsupported_target_rate_raises() -> None:
    with pytest.raises(ValueError, match="No prototype preset"):
        prototype_specs_for_target_rate(192_000)


@pytest.mark.parametrize(
    "profile,length",
    [
        ("long_sharp_1023_a120", 1023),
        ("long_sharp_1023_a140", 1023),
        ("long_sharp_1535_a120", 1535),
        ("long_sharp_2047_a120", 2047),
        ("long_sharp_2047_a140", 2047),
        ("long_sharp_2047_a160", 2047),
        ("long_sharp_3071_a120", 3071),
        ("long_sharp_3071_a140", 3071),
        ("long_sharp_4095_a120", 4095),
        ("long_sharp_4095_a140", 4095),
    ],
)
def test_long_fir_profiles_share_requested_length(profile: str, length: int) -> None:
    bank = build_prototype_bank_for_profile(TARGET_SR, profile)
    assert bank.kernels.shape == (3, length)
    assert bank.group_delay_samples == (length - 1) // 2
    assert bank.profile_name == profile
    assert len(bank.coefficient_hash) == 64


def test_variant_profile_redesigns_mid_and_gentle() -> None:
    bank = build_prototype_bank_for_profile(
        TARGET_SR, "v5_sharp1023_midflat70_gentleb4k24"
    )
    assert bank.names == ("sharp", "mid", "gentle")
    assert bank.kernels.shape == (3, 1023)
    mid = bank.kernels[1]
    gentle = bank.kernels[2]
    # The flat middle keeps a short support and its passband to 20 kHz.
    assert 80 <= np.count_nonzero(mid) <= 120
    freqs, response = sp_signal.freqz(mid, worN=1 << 14, fs=TARGET_SR)
    magnitude = 20 * np.log10(np.abs(response) / np.abs(response[0]))
    assert abs(magnitude[np.argmin(np.abs(freqs - 20_000.0))]) < 0.05
    assert magnitude[np.argmin(np.abs(freqs - 24_000.0))] < -60.0
    # The gentle endpoint stays 101 taps and loses less than 2 dB at 15 kHz.
    assert np.count_nonzero(gentle) == 101
    freqs, response = sp_signal.freqz(gentle, worN=1 << 14, fs=TARGET_SR)
    magnitude = 20 * np.log10(np.abs(response) / np.abs(response[0]))
    assert -2.0 < magnitude[np.argmin(np.abs(freqs - 15_000.0))] < -1.0


def test_long_fir_profiles_preserve_mid_and_gentle_responses() -> None:
    release = build_prototype_bank_for_profile(TARGET_SR, RELEASE_PROTOTYPE_PROFILE)
    candidate = build_prototype_bank_for_profile(TARGET_SR, "long_sharp_2047_a120")
    for name in ("mid", "gentle"):
        release_kernel = release.kernels[release.names.index(name)]
        candidate_kernel = candidate.kernels[candidate.names.index(name)]
        pad = (candidate_kernel.size - release_kernel.size) // 2
        np.testing.assert_array_equal(
            candidate_kernel, np.pad(release_kernel, (pad, pad))
        )


def test_supported_profiles_include_release_default() -> None:
    assert supported_prototype_profiles()[0] == RELEASE_PROTOTYPE_PROFILE


@pytest.fixture(scope="module")
def bank_48k():
    """Build the 48k-family prototype bank once per module."""
    return build_prototype_bank(PROTOTYPE_SPECS_48K, sample_rate=96_000)


def test_48k_bank_structure(bank_48k) -> None:
    assert bank_48k.names == ("sharp", "mid", "gentle")
    assert bank_48k.sample_rate == 96_000
    assert bank_48k.kernels.shape[1] % 2 == 1


def test_48k_bank_validates(bank_48k) -> None:
    results = validate_bank(bank_48k)
    assert results["kernel_symmetry_rel"] <= 1e-12
    assert results["kaiser_passband_match_db"] <= -70.0


def test_48k_sharp_kernel_shorter_than_44k1() -> None:
    """The 2 kHz (vs 1.05 kHz) transition budget must shrink the kernel."""
    sharp_44k1 = design_kaiser_prototype(DEFAULT_PROTOTYPE_SPECS[0], 88_200)
    sharp_48k = design_kaiser_prototype(PROTOTYPE_SPECS_48K[0], 96_000)
    assert sharp_48k.size < sharp_44k1.size / 1.5


def test_48k_sharp_prototype_suppresses_images(bank_48k) -> None:
    """A 19 kHz tone's 29 kHz image must vanish through the 48k sharp kernel."""
    source_sr, target_sr = 48_000, 96_000
    time_axis = np.arange(source_sr // 2) / source_sr
    tone = 0.5 * np.sin(2.0 * np.pi * 19_000.0 * time_axis)
    output = upsample_with_kernel(tone, bank_48k.kernels[0], bank_48k.upsample_ratio)

    spectrum = np.abs(np.fft.rfft(output * np.hanning(output.size)))
    freqs = np.fft.rfftfreq(output.size, d=1.0 / target_sr)
    tone_level = spectrum[np.argmin(np.abs(freqs - 19_000.0))]
    image_level = spectrum[np.argmin(np.abs(freqs - 29_000.0))]
    assert 20.0 * np.log10(image_level / tone_level) <= -80.0


def test_48k_summarize_bank_image_band(bank_48k) -> None:
    """Image band derives from the 24 kHz input Nyquist at 96 kHz."""
    summary = summarize_bank(bank_48k)
    assert summary["sharp"]["image_band_max_db"] <= -90.0
    assert summary["sharp"]["passband_dev_db"] <= -80.0


def test_48k_gentle_matches_bessel_magnitude() -> None:
    """The 96k gentle FIR must track the Bessel6@20k magnitude to -50 dB."""
    spec = PROTOTYPE_SPECS_48K[2]
    taps = design_bessel_magnitude_prototype(spec, 96_000)
    b, a = sp_signal.bessel(
        spec.order,
        spec.cutoff_hz,
        btype="lowpass",
        analog=False,
        output="ba",
        norm="phase",
        fs=96_000,
    )
    dense = np.linspace(0.0, 48_000.0, 1 << 13)
    _, h_fir = sp_signal.freqz(taps, worN=dense, fs=96_000)
    _, h_bessel = sp_signal.freqz(b, a, worN=dense, fs=96_000)
    # Kernels carry the 2x interpolation gain; compare unit-gain shapes.
    gain = np.abs(h_fir[0])
    error = np.max(np.abs(np.abs(h_fir) / gain - np.abs(h_bessel)))
    assert 20.0 * np.log10(error) <= -50.0
