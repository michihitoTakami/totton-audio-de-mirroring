import numpy as np
import pytest
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.generator import (
    GeneratorConfig,
    SignalRequest,
    SyntheticSignalGenerator,
    apply_soft_clip,
    generate_am_tone,
    generate_band_limited_noise,
    generate_fm_tone,
    generate_impulse_train,
    generate_linear_sweep,
    generate_log_sweep,
    generate_multitone,
    generate_percussive_transient,
    generate_pink_noise,
    generate_signal,
    generate_soft_clipped_tone,
    generate_white_noise,
    list_signal_types,
)

SAMPLE_RATE = 22_050
DURATION = 0.5


def _spectrum(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, 1.0 / sample_rate)
    return freqs, np.abs(spectrum)


def _peak_magnitude(freqs: np.ndarray, mags: np.ndarray, target_hz: float) -> float:
    idx = int(np.argmin(np.abs(freqs - target_hz)))
    return float(mags[idx])


def test_list_signal_types_has_minimum_coverage() -> None:
    names = list_signal_types()
    assert len(names) >= 10
    for key in (
        "multitone",
        "sweep_linear",
        "sweep_log",
        "impulse_train",
        "percussive",
        "am_tone",
        "fm_tone",
        "white_noise",
        "pink_noise",
        "band_limited_noise",
        "soft_clipped_tone",
    ):
        assert key in names


def test_generate_multitone_has_peaks() -> None:
    freqs = [440.0, 880.0, 1320.0]
    signal = generate_multitone(
        freqs,
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION,
        amplitude=0.8,
    )
    spectrum_freqs, mags = _spectrum(signal, SAMPLE_RATE)
    baseline = float(np.median(mags))
    for freq in freqs:
        assert _peak_magnitude(spectrum_freqs, mags, freq) > 10.0 * baseline


def test_linear_sweep_tracks_start_and_end() -> None:
    start_hz = 200.0
    end_hz = 4000.0
    signal = generate_linear_sweep(
        start_hz=start_hz,
        end_hz=end_hz,
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION,
    )
    freqs, _times, spec = sp_signal.spectrogram(
        signal, fs=SAMPLE_RATE, nperseg=512, noverlap=256
    )
    first_bin = int(np.argmax(spec[:, 0]))
    last_bin = int(np.argmax(spec[:, -1]))
    assert abs(freqs[first_bin] - start_hz) < 400.0
    assert freqs[last_bin] > 0.75 * end_hz


def test_log_sweep_tracks_start_and_end() -> None:
    start_hz = 100.0
    end_hz = 5000.0
    signal = generate_log_sweep(
        start_hz=start_hz,
        end_hz=end_hz,
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION,
    )
    freqs, _times, spec = sp_signal.spectrogram(
        signal, fs=SAMPLE_RATE, nperseg=512, noverlap=256
    )
    first_bin = int(np.argmax(spec[:, 0]))
    last_bin = int(np.argmax(spec[:, -1]))
    assert abs(freqs[first_bin] - start_hz) < 400.0
    assert freqs[last_bin] > 0.75 * end_hz


def test_impulse_train_shows_comb_spectrum() -> None:
    interval_sec = 0.01
    signal = generate_impulse_train(
        interval_sec=interval_sec,
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION,
    )
    freqs, mags = _spectrum(signal, SAMPLE_RATE)
    baseline = float(np.median(mags))
    fundamental = 1.0 / interval_sec
    for multiple in (1, 2, 3, 4, 5):
        peak = _peak_magnitude(freqs, mags, fundamental * multiple)
        assert peak > 5.0 * baseline


def test_percussive_transient_decays() -> None:
    signal = generate_percussive_transient(
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION,
        decay_rate=10.0,
        rng=np.random.default_rng(0),
    )
    midpoint = signal.size // 2
    early = np.sqrt(np.mean(signal[:midpoint] ** 2))
    late = np.sqrt(np.mean(signal[midpoint:] ** 2))
    assert early > 2.0 * late


def test_am_tone_has_sidebands() -> None:
    carrier = 2000.0
    mod = 200.0
    signal = generate_am_tone(
        carrier_hz=carrier,
        mod_hz=mod,
        modulation_index=0.6,
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION,
    )
    freqs, mags = _spectrum(signal, SAMPLE_RATE)
    baseline = float(np.median(mags))
    assert _peak_magnitude(freqs, mags, carrier) > 10.0 * baseline
    assert _peak_magnitude(freqs, mags, carrier - mod) > 5.0 * baseline
    assert _peak_magnitude(freqs, mags, carrier + mod) > 5.0 * baseline


def test_fm_tone_has_multiple_sidebands() -> None:
    carrier = 1000.0
    mod = 100.0
    signal = generate_fm_tone(
        carrier_hz=carrier,
        mod_hz=mod,
        modulation_index=4.0,
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION,
    )
    freqs, mags = _spectrum(signal, SAMPLE_RATE)
    band_mask = (freqs > carrier - 5 * mod) & (freqs < carrier + 5 * mod)
    band_mags = mags[band_mask]
    threshold = 0.1 * float(np.max(band_mags))
    assert int(np.sum(band_mags > threshold)) >= 4


def test_white_noise_is_roughly_flat() -> None:
    signal = generate_white_noise(
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION,
        rng=np.random.default_rng(1),
    )
    freqs, mags = _spectrum(signal, SAMPLE_RATE)
    low = mags[(freqs >= 500.0) & (freqs < 3000.0)]
    high = mags[(freqs >= 3000.0) & (freqs < 5500.0)]
    ratio = float(np.mean(low) / np.mean(high))
    assert 0.5 < ratio < 2.0


def test_pink_noise_has_negative_slope() -> None:
    signal = generate_pink_noise(
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION,
        rng=np.random.default_rng(2),
    )
    freqs, power = sp_signal.welch(signal, fs=SAMPLE_RATE, nperseg=1024)
    mask = (freqs >= 100.0) & (freqs <= 5000.0)
    log_freqs = np.log10(freqs[mask])
    log_power = np.log10(power[mask] + 1e-12)
    slope, _ = np.polyfit(log_freqs, log_power, 1)
    assert slope < -0.3


def test_band_limited_noise_energy_inside_band() -> None:
    signal = generate_band_limited_noise(
        low_hz=1000.0,
        high_hz=3000.0,
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION,
        rng=np.random.default_rng(3),
    )
    freqs, mags = _spectrum(signal, SAMPLE_RATE)
    inside = mags[(freqs >= 1000.0) & (freqs <= 3000.0)]
    outside = mags[(freqs < 800.0) | (freqs > 3500.0)]
    ratio = float(np.mean(inside) / np.mean(outside))
    assert ratio > 4.0


def test_soft_clipped_tone_has_harmonics() -> None:
    signal = generate_soft_clipped_tone(
        frequency_hz=1000.0,
        drive=3.0,
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION,
    )
    freqs, mags = _spectrum(signal, SAMPLE_RATE)
    fundamental = _peak_magnitude(freqs, mags, 1000.0)
    third = _peak_magnitude(freqs, mags, 3000.0)
    assert third / fundamental > 0.03


def test_apply_soft_clip_is_pure() -> None:
    signal = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    clipped = apply_soft_clip(signal, drive=2.0)
    assert np.all(signal == np.array([0.0, 0.5, -0.5], dtype=np.float32))
    assert clipped.dtype == signal.dtype


def test_generate_signal_dispatch_and_seed() -> None:
    one = generate_signal(
        "white_noise",
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION,
        seed=123,
    )
    two = generate_signal(
        "white_noise",
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION,
        seed=123,
    )
    assert np.allclose(one, two)


def test_generator_class_batch() -> None:
    generator = SyntheticSignalGenerator(
        GeneratorConfig(sample_rate=SAMPLE_RATE, duration_sec=DURATION, seed=10)
    )
    batch = generator.generate_batch(
        [
            SignalRequest("white_noise", {}),
            SignalRequest("multitone", {"frequencies_hz": [440.0, 880.0]}),
        ]
    )
    assert len(batch) == 2
    assert batch[0].size == batch[1].size


def test_invalid_signal_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown signal_type"):
        generate_signal("unknown", sample_rate=SAMPLE_RATE, duration_sec=DURATION)


def test_invalid_modulation_index_raises() -> None:
    with pytest.raises(ValueError, match="modulation_index"):
        generate_am_tone(
            carrier_hz=1000.0,
            mod_hz=100.0,
            modulation_index=1.5,
            sample_rate=SAMPLE_RATE,
            duration_sec=DURATION,
        )
