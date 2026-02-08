import numpy as np
import pytest

from totton_audio_de_mirroring.data.degradation import (
    DegradationConfig,
    DegradationProfile,
    DegradationProfileManager,
    apply_degradation_profile,
    apply_random_degradation,
    upsample_bessel_reference,
)

SOURCE_SR = 44_100
TARGET_SR = 88_200
CUTOFF_HZ = 20_000.0


def test_default_config_is_bessel_only() -> None:
    config = DegradationConfig()
    assert config.methods == ("iir_bessel",)
    assert config.phase_modes == ("analog",)


def test_random_degradation_is_reproducible() -> None:
    signal = np.linspace(-0.5, 0.5, 128)

    output_a, profile_a = apply_random_degradation(
        signal,
        SOURCE_SR,
        TARGET_SR,
        seed=42,
    )
    output_b, profile_b = apply_random_degradation(
        signal,
        SOURCE_SR,
        TARGET_SR,
        seed=42,
    )

    assert profile_a == profile_b
    assert np.allclose(output_a, output_b)


def test_apply_degradation_profile_length() -> None:
    signal = np.random.default_rng(0).normal(0.0, 1.0, 64)
    rng = np.random.default_rng(0)

    profile = DegradationProfile(
        method="iir_bessel",
        cutoff_hz=CUTOFF_HZ,
        phase="analog",
        quantization_bits=16,
        dither="none",
        num_taps=None,
        iir_order=4,
    )
    degraded = apply_degradation_profile(
        signal,
        SOURCE_SR,
        TARGET_SR,
        profile,
        rng,
    )
    assert degraded.shape[-1] == signal.shape[-1] * 2


def test_invalid_ratio_raises() -> None:
    signal = np.random.randn(64)
    manager = DegradationProfileManager(DegradationConfig())
    profile = manager.sample_profile(rng=np.random.default_rng(0))

    with pytest.raises(ValueError, match="integer multiple"):
        apply_degradation_profile(
            signal,
            SOURCE_SR,
            48_000,
            profile,
            np.random.default_rng(0),
        )


def test_frequency_response_for_bessel_profile() -> None:
    num_samples = 4096
    time = np.arange(num_samples) / SOURCE_SR
    signal = np.sin(2 * np.pi * 1_000.0 * time)
    profile = DegradationProfile(
        method="iir_bessel",
        cutoff_hz=CUTOFF_HZ,
        phase="analog",
        quantization_bits=24,
        dither="none",
        num_taps=None,
        iir_order=6,
    )

    degraded = apply_degradation_profile(
        signal,
        SOURCE_SR,
        TARGET_SR,
        profile,
        np.random.default_rng(1),
    )
    spectrum = np.fft.rfft(degraded)
    freqs = np.fft.rfftfreq(degraded.size, 1.0 / TARGET_SR)

    peak_idx = int(np.argmax(np.abs(spectrum)))
    peak_freq = freqs[peak_idx]
    assert abs(peak_freq - 1_000.0) < 20.0

    high_band = freqs >= 30_000.0
    max_high = float(np.max(np.abs(spectrum[high_band])))
    max_low = float(np.max(np.abs(spectrum)))
    assert max_high < max_low * 0.1


def test_upsample_bessel_reference_length_and_type() -> None:
    signal = np.sin(2.0 * np.pi * 440.0 * np.arange(256) / SOURCE_SR)
    output = upsample_bessel_reference(
        signal=signal,
        source_sr=SOURCE_SR,
        target_sr=TARGET_SR,
        cutoff_hz=20_000.0,
        order=6,
    )
    assert output.shape[-1] == signal.shape[-1] * 2
    assert output.dtype == np.float64
