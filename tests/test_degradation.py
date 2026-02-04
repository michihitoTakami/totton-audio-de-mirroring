import numpy as np
import pytest

from totton_audio_de_mirroring.data.degradation import (
    DegradationConfig,
    DegradationProfile,
    DegradationProfileManager,
    apply_degradation_profile,
    apply_random_degradation,
)

SOURCE_SR = 44_100
TARGET_SR = 88_200
CUTOFF_HZ = 20_000.0


def test_default_config_has_minimum_methods() -> None:
    config = DegradationConfig()
    assert len(config.methods) >= 5


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

    profiles = [
        DegradationProfile(
            method="zoh",
            cutoff_hz=CUTOFF_HZ,
            phase="linear",
            quantization_bits=16,
            dither="none",
            num_taps=None,
            iir_order=None,
        ),
        DegradationProfile(
            method="linear",
            cutoff_hz=CUTOFF_HZ,
            phase="linear",
            quantization_bits=16,
            dither="none",
            num_taps=None,
            iir_order=None,
        ),
        DegradationProfile(
            method="sinc_short",
            cutoff_hz=CUTOFF_HZ,
            phase="linear",
            quantization_bits=16,
            dither="none",
            num_taps=64,
            iir_order=None,
        ),
        DegradationProfile(
            method="sinc_long",
            cutoff_hz=CUTOFF_HZ,
            phase="linear",
            quantization_bits=16,
            dither="none",
            num_taps=128,
            iir_order=None,
        ),
        DegradationProfile(
            method="iir_bessel",
            cutoff_hz=CUTOFF_HZ,
            phase="analog",
            quantization_bits=16,
            dither="none",
            num_taps=None,
            iir_order=4,
        ),
        DegradationProfile(
            method="iir_butter",
            cutoff_hz=CUTOFF_HZ,
            phase="analog",
            quantization_bits=16,
            dither="none",
            num_taps=None,
            iir_order=4,
        ),
    ]

    for profile in profiles:
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
