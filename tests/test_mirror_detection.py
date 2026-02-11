import numpy as np
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.mirror_detection import (
    MirrorDetectionConfig,
    detect_mirror_artifacts,
    generate_hb_target,
    project_teacher_hb_target,
)

SAMPLE_RATE = 88_200
DURATION_SEC = 0.5
N_FFT = 2048
HOP_LENGTH = 512


def _time_axis(sample_rate: int, duration_sec: float) -> np.ndarray:
    num_samples = int(round(sample_rate * duration_sec))
    return np.arange(num_samples, dtype=np.float64) / float(sample_rate)


def _sine(freq_hz: float, sample_rate: int, duration_sec: float) -> np.ndarray:
    time = _time_axis(sample_rate, duration_sec)
    return np.sin(2.0 * np.pi * freq_hz * time)


def _stft_magnitude(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs, _, stft = sp_signal.stft(
        signal,
        fs=SAMPLE_RATE,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        window="hann",
        boundary="zeros",
        padded=True,
    )
    return freqs, np.abs(stft)


def test_detect_mirror_artifacts_detects_symmetry() -> None:
    mirror_center = SAMPLE_RATE / 4.0
    low_freq = 21_000.0
    high_freq = 2.0 * mirror_center - low_freq
    signal = _sine(low_freq, SAMPLE_RATE, DURATION_SEC) + 0.8 * _sine(
        high_freq, SAMPLE_RATE, DURATION_SEC
    )

    config = MirrorDetectionConfig(
        cutoff_hz=20_000.0,
        mirror_center_hz=mirror_center,
        mirror_band_hz=(20_000.0, 22_000.0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        magnitude_threshold=1.5,
        symmetry_threshold=0.3,
    )

    result = detect_mirror_artifacts(signal, SAMPLE_RATE, config)
    idx_low = int(np.argmin(np.abs(result.freqs - low_freq)))
    idx_high = int(np.argmin(np.abs(result.freqs - high_freq)))

    detection_rate = float(
        np.mean(
            np.concatenate(
                [result.detection_mask[idx_low], result.detection_mask[idx_high]]
            )
        )
    )
    assert detection_rate > 0.8


def test_detect_mirror_artifacts_rejects_non_mirror() -> None:
    signal = _sine(30_000.0, SAMPLE_RATE, DURATION_SEC)
    config = MirrorDetectionConfig(
        cutoff_hz=20_000.0,
        mirror_center_hz=SAMPLE_RATE / 4.0,
        mirror_band_hz=(20_000.0, 22_000.0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        magnitude_threshold=2.0,
        symmetry_threshold=0.6,
    )
    result = detect_mirror_artifacts(signal, SAMPLE_RATE, config)
    assert not np.any(result.detection_mask)


def test_generate_hb_target_suppresses_mirror() -> None:
    mirror_center = SAMPLE_RATE / 4.0
    low_freq = 21_000.0
    high_freq = 2.0 * mirror_center - low_freq
    keep_freq = 30_000.0
    signal = (
        _sine(low_freq, SAMPLE_RATE, DURATION_SEC)
        + _sine(high_freq, SAMPLE_RATE, DURATION_SEC)
        + 0.6 * _sine(keep_freq, SAMPLE_RATE, DURATION_SEC)
    )

    config = MirrorDetectionConfig(
        cutoff_hz=20_000.0,
        mirror_center_hz=mirror_center,
        mirror_band_hz=(20_000.0, 22_000.0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        magnitude_threshold=1.5,
        symmetry_threshold=0.3,
    )

    result = generate_hb_target(
        signal,
        SAMPLE_RATE,
        detection_config=config,
        suppression_floor=0.1,
        energy_cap=10.0,
        envelope_min=1.0,
    )

    freqs, mag_in = _stft_magnitude(signal)
    _, mag_out = _stft_magnitude(result.target)

    idx_low = int(np.argmin(np.abs(freqs - low_freq)))
    idx_high = int(np.argmin(np.abs(freqs - high_freq)))
    idx_keep = int(np.argmin(np.abs(freqs - keep_freq)))

    mirror_ratio_low = float(np.mean(mag_out[idx_low]) / np.mean(mag_in[idx_low]))
    mirror_ratio_high = float(np.mean(mag_out[idx_high]) / np.mean(mag_in[idx_high]))
    mirror_ratio = min(mirror_ratio_low, mirror_ratio_high)
    keep_ratio = float(np.mean(mag_out[idx_keep]) / np.mean(mag_in[idx_keep]))

    assert mirror_ratio < 0.6
    assert keep_ratio > 0.8


def test_energy_cap_is_enforced() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, int(SAMPLE_RATE * DURATION_SEC))
    taps = sp_signal.firwin(513, [20_000.0, 40_000.0], pass_zero=False, fs=SAMPLE_RATE)
    hb_signal = sp_signal.lfilter(taps, [1.0], noise)

    config = MirrorDetectionConfig(
        cutoff_hz=20_000.0,
        mirror_center_hz=SAMPLE_RATE / 4.0,
        mirror_band_hz=(20_000.0, 22_000.0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    result = generate_hb_target(
        hb_signal,
        SAMPLE_RATE,
        detection_config=config,
        suppression_floor=1.0,
        energy_cap=1e-4,
        envelope_min=1.0,
    )

    freqs, mag_out = _stft_magnitude(result.target)
    highband = freqs >= 20_000.0
    energy = float(np.mean(mag_out[highband] ** 2))
    assert energy <= 1.05e-4


def test_project_teacher_hb_target_applies_suppression_floor_on_detected_bins() -> None:
    mirror_center = SAMPLE_RATE / 4.0
    low_freq = 21_000.0
    high_freq = 2.0 * mirror_center - low_freq
    signal = _sine(low_freq, SAMPLE_RATE, DURATION_SEC) + 0.8 * _sine(
        high_freq, SAMPLE_RATE, DURATION_SEC
    )

    config = MirrorDetectionConfig(
        cutoff_hz=20_000.0,
        mirror_center_hz=mirror_center,
        mirror_band_hz=(20_000.0, 22_000.0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        magnitude_threshold=1.5,
        symmetry_threshold=0.3,
    )
    detected_in = detect_mirror_artifacts(signal, SAMPLE_RATE, config)
    assert np.any(detected_in.detection_mask)

    projected = project_teacher_hb_target(
        signal,
        signal,
        SAMPLE_RATE,
        detection_config=config,
        suppression_floor=0.2,
        energy_cap=10.0,
        envelope_min=1.0,
    )
    detected_out = detect_mirror_artifacts(projected, SAMPLE_RATE, config)

    mask = detected_in.detection_mask
    input_mag = np.abs(detected_in.stft)[mask]
    output_mag = np.abs(detected_out.stft)[mask]
    ratio = float(np.mean(output_mag / (input_mag + 1.0e-12)))
    assert ratio <= 0.30
