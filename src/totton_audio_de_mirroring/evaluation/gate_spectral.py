"""Band-level helpers shared by the Stage 1 image and mirror gates.

Physical Basis:
    Image suppression is judged by comparing energy above the input Nyquist
    against the audible band, so every gate that touches the image band needs
    the same band-edge definition and the same level estimator.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

_EPSILON = 1e-300

MAIN_BAND_HIGH_HZ = 20_000.0
# Image-band guard offset above the input Nyquist; the actual band edge is
# rate-dependent (22.55 kHz at 88.2k target, 24.5 kHz at 96k target).
IMAGE_BAND_NYQUIST_OFFSET_HZ = 500.0
IMAGE_BAND_LOW_HZ = 22_550.0  # 44.1k-family value, kept for reference.
GAIN_BAND_HIGH_HZ = 10_000.0
GAIN_APPLICABILITY_HIGH_HZ = 8_000.0


def _band_level_db(
    signal: np.ndarray, sample_rate: int, low_hz: float, high_hz: float
) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band):
        return -300.0
    level = np.sqrt(np.mean(np.square(spectrum[band]))) / signal.size
    return float(20.0 * np.log10(max(level, _EPSILON)))


def _band_energy_fraction(
    signal: np.ndarray, sample_rate: int, cutoff_hz: float
) -> float:
    spectrum = np.square(np.abs(np.fft.rfft(signal)))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    total = float(np.sum(spectrum))
    if total <= 0.0:
        return 0.0
    return float(np.sum(spectrum[freqs <= cutoff_hz]) / total)


def image_band_low_hz(target_sample_rate: int) -> float:
    """Return the image-band lower edge for a 2x-upsampled target rate.

    Physical Basis:
        Mirror images of a 2x upsampler start at the input Nyquist
        (target_rate / 4); the fixed offset keeps the measurement clear of
        brickwall transition skirts. Evaluates to 22 550 Hz at 88.2 kHz and
        24 500 Hz at 96 kHz.
    """
    if target_sample_rate <= 0:
        raise ValueError(
            f"target_sample_rate must be positive, got {target_sample_rate}."
        )
    return target_sample_rate / 4.0 + IMAGE_BAND_NYQUIST_OFFSET_HZ


def _image_minus_main_db(signal: np.ndarray, sample_rate: int) -> float:
    image = _band_level_db(
        signal, sample_rate, image_band_low_hz(sample_rate), sample_rate / 2
    )
    main = _band_level_db(signal, sample_rate, 20.0, MAIN_BAND_HIGH_HZ)
    return image - main


def _peak_image_minus_main_db(signal: np.ndarray, sample_rate: int) -> float:
    """Return peak swept-image ridge relative to the main swept ridge.

    Physical Basis:
        An integrated image-band average can hide a narrow residual confined
        to the end of a sweep. The maximum Hann-STFT magnitude over time
        follows the swept ridge at each frequency and makes that worst-case
        residual binding.
    """
    nperseg = min(2_048, signal.size)
    if nperseg < 16:
        raise ValueError("signal is too short for peak image measurement.")
    frequencies, _, spectrum = sp_signal.stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg * 7 // 8,
        nfft=nperseg,
        boundary=None,
        padded=False,
    )
    envelope = np.max(np.abs(spectrum), axis=1)
    main = (frequencies >= 20.0) & (frequencies <= MAIN_BAND_HIGH_HZ)
    image = frequencies >= image_band_low_hz(sample_rate)
    main_peak = float(np.max(envelope[main]))
    image_peak = float(np.max(envelope[image]))
    return float(20.0 * np.log10(max(image_peak, _EPSILON) / max(main_peak, _EPSILON)))
