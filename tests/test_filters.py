import numpy as np
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.filters import (
    band_split,
    design_band_split_filters,
    design_bessel_fir,
)

SAMPLE_RATE = 88_200
CUTOFF_HZ = 20_000.0


def test_design_bessel_fir_length_and_gain() -> None:
    taps = design_bessel_fir(
        cutoff_hz=CUTOFF_HZ,
        sample_rate=SAMPLE_RATE,
        order=8,
        num_taps=1025,
    )
    assert taps.shape == (1025,)
    assert np.isfinite(taps).all()
    assert abs(float(np.sum(taps)) - 1.0) < 1e-3


def test_bessel_fir_group_delay_is_flat() -> None:
    taps = design_bessel_fir(
        cutoff_hz=CUTOFF_HZ,
        sample_rate=SAMPLE_RATE,
        order=8,
        num_taps=1025,
    )

    frequencies = np.linspace(500.0, 18_000.0, 256)
    _, group_delay = sp_signal.group_delay(
        (taps, [1.0]),
        w=frequencies,
        fs=SAMPLE_RATE,
    )

    delay_variation_samples = float(np.max(group_delay) - np.min(group_delay))
    delay_variation_ms = delay_variation_samples / SAMPLE_RATE * 1000.0

    assert delay_variation_ms < 1.0


def test_bessel_fir_step_response_has_minimal_overshoot() -> None:
    taps = design_bessel_fir(
        cutoff_hz=CUTOFF_HZ,
        sample_rate=SAMPLE_RATE,
        order=8,
        num_taps=1025,
    )

    step_signal = np.concatenate([np.zeros(2048), np.ones(4096)])
    filtered = sp_signal.lfilter(taps, [1.0], step_signal)

    max_value = float(np.max(filtered))
    assert max_value <= 1.08


def test_band_split_recombines_with_delay() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(0.0, 1.0, 4096)

    lowpass, highpass = design_band_split_filters(
        cutoff_hz=CUTOFF_HZ,
        sample_rate=SAMPLE_RATE,
        num_taps=1025,
    )

    low_band, high_band = band_split(signal, lowpass, highpass)
    recombined = low_band + high_band

    delay = (lowpass.size - 1) // 2
    aligned_recombined = recombined[delay:]
    aligned_original = signal[:-delay]

    error = np.mean(np.abs(aligned_recombined - aligned_original))
    assert error < 1e-2


def test_low_band_preservation_after_split() -> None:
    num_samples = 8192
    time = np.arange(num_samples) / SAMPLE_RATE
    signal = (
        0.6 * np.sin(2 * np.pi * 1_000.0 * time)
        + 0.3 * np.sin(2 * np.pi * 15_000.0 * time)
        + 0.2 * np.sin(2 * np.pi * 25_000.0 * time)
    )

    lowpass, highpass = design_band_split_filters(
        cutoff_hz=CUTOFF_HZ,
        sample_rate=SAMPLE_RATE,
        num_taps=1025,
    )

    low_band, high_band = band_split(signal, lowpass, highpass)
    recombined = low_band + high_band

    delay = (lowpass.size - 1) // 2
    aligned_recombined = recombined[delay:]
    aligned_original = signal[:-delay]

    original_fft = np.fft.rfft(aligned_original)
    recombined_fft = np.fft.rfft(aligned_recombined)
    freqs = np.fft.rfftfreq(aligned_original.size, 1.0 / SAMPLE_RATE)

    for tone_hz in (1_000.0, 15_000.0):
        idx = int(np.argmin(np.abs(freqs - tone_hz)))
        reference = original_fft[idx]
        candidate = recombined_fft[idx]
        denom = max(np.abs(reference), 1e-12)
        error = np.abs(candidate - reference) / denom
        assert error < 0.02
