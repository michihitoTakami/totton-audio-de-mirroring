"""Mirror artifact detection and HB_target generation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

DEFAULT_CUTOFF_HZ = 20_000.0
DEFAULT_MIRROR_BAND_HZ = (20_000.0, 22_050.0)
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_WINDOW = "hann"
DEFAULT_MAG_THRESHOLD = 3.0
DEFAULT_SYMMETRY_THRESHOLD = 0.6
DEFAULT_SUPPRESSION_FLOOR = 0.2
DEFAULT_ENVELOPE_MIN = 0.2
DEFAULT_ENERGY_CAP = 1e-3
DEFAULT_MIRROR_CENTER_RATIO = 0.25


@dataclass(frozen=True)
class MirrorDetectionConfig:
    """Configuration for mirror artifact detection in STFT domain.

    Args:
        cutoff_hz: Low-band cutoff in Hz.
        mirror_center_hz: Center frequency for symmetry detection.
        mirror_band_hz: Lower-side band (Hz) to scan for mirror symmetry.
        n_fft: FFT size for STFT.
        hop_length: Hop length for STFT.
        window: Window name for STFT.
        magnitude_threshold: Threshold multiplier over median magnitude.
        symmetry_threshold: Minimum symmetry ratio for mirror detection.

    Physical Basis:
        Mirror artifacts appear as symmetric energy around the original
        Nyquist (typically source_sr / 2). The STFT reveals these geometric
        symmetries, enabling rule-based detection.
    """

    cutoff_hz: float = DEFAULT_CUTOFF_HZ
    mirror_center_hz: float | None = None
    mirror_band_hz: tuple[float, float] = DEFAULT_MIRROR_BAND_HZ
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH
    window: str = DEFAULT_WINDOW
    magnitude_threshold: float = DEFAULT_MAG_THRESHOLD
    symmetry_threshold: float = DEFAULT_SYMMETRY_THRESHOLD


@dataclass(frozen=True)
class MirrorDetectionResult:
    """Result of mirror artifact detection.

    Args:
        freqs: STFT frequency bins in Hz.
        times: STFT time bins in seconds.
        stft: Complex STFT of the input signal.
        magnitude: STFT magnitude.
        detection_mask: Boolean mask indicating detected mirror bins.
        mirror_pairs: Tuple of index pairs used for symmetry checks.

    Physical Basis:
        Mirror artifacts manifest as symmetric energy pairs. The detection
        mask flags time-frequency bins matching those symmetries.
    """

    freqs: np.ndarray
    times: np.ndarray
    stft: np.ndarray
    magnitude: np.ndarray
    detection_mask: np.ndarray
    mirror_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class HBTargetResult:
    """Result of HB_target generation.

    Args:
        target: Time-domain HB_target signal.
        detection: Mirror detection result used for suppression.
        suppression_mask: Applied suppression mask in STFT domain.
        envelope: Frequency envelope applied in STFT domain.
        energy_scale: Energy scaling factor applied for energy cap.

    Physical Basis:
        The target suppresses mirror artifacts while enforcing energy
        constraints and a gentle high-frequency decay profile.
    """

    target: np.ndarray
    detection: MirrorDetectionResult
    suppression_mask: np.ndarray
    envelope: np.ndarray
    energy_scale: float


def detect_mirror_artifacts(
    hb_signal: np.ndarray,
    sample_rate: int,
    config: MirrorDetectionConfig | None = None,
) -> MirrorDetectionResult:
    """Detect mirror artifacts in STFT domain.

    Args:
        hb_signal: High-band input signal (HB_in).
        sample_rate: Sample rate in Hz.
        config: Optional mirror detection configuration.

    Returns:
        MirrorDetectionResult with detection mask and metadata.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Mirror artifacts show symmetric energy around the source Nyquist.
        STFT magnitude symmetry checks identify these regions.
    """

    _validate_signal(hb_signal)
    _validate_sample_rate(sample_rate)

    cfg = config or MirrorDetectionConfig()
    mirror_center = _resolve_mirror_center(cfg.mirror_center_hz, sample_rate)
    _validate_detection_config(cfg, sample_rate, mirror_center)

    freqs, times, stft = _compute_stft(
        hb_signal,
        sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=cfg.window,
    )
    magnitude = np.abs(stft)

    highband_mask = freqs >= cfg.cutoff_hz
    median_mag = np.median(magnitude[highband_mask], axis=0)
    baseline = median_mag + 1e-12

    mirror_band_mask = (freqs >= cfg.mirror_band_hz[0]) & (
        freqs <= cfg.mirror_band_hz[1]
    )

    detection_mask = np.zeros_like(magnitude, dtype=bool)
    mirror_pairs: list[tuple[int, int]] = []

    for idx in np.where(mirror_band_mask)[0]:
        mirror_freq = 2.0 * mirror_center - freqs[idx]
        if mirror_freq <= 0.0 or mirror_freq >= freqs[-1]:
            continue
        mirror_idx = int(np.argmin(np.abs(freqs - mirror_freq)))
        if not highband_mask[mirror_idx]:
            continue

        mag_a = magnitude[idx]
        mag_b = magnitude[mirror_idx]
        strong = (mag_a > baseline * cfg.magnitude_threshold) & (
            mag_b > baseline * cfg.magnitude_threshold
        )
        symmetry = (
            np.minimum(mag_a, mag_b) / (np.maximum(mag_a, mag_b) + 1e-12)
            >= cfg.symmetry_threshold
        )
        detected = strong & symmetry

        if np.any(detected):
            detection_mask[idx] = detection_mask[idx] | detected
            detection_mask[mirror_idx] = detection_mask[mirror_idx] | detected
        mirror_pairs.append((int(idx), int(mirror_idx)))

    return MirrorDetectionResult(
        freqs=np.asarray(freqs, dtype=np.float64),
        times=np.asarray(times, dtype=np.float64),
        stft=np.asarray(stft, dtype=np.complex128),
        magnitude=np.asarray(magnitude, dtype=np.float64),
        detection_mask=detection_mask,
        mirror_pairs=tuple(mirror_pairs),
    )


def generate_hb_target(
    hb_signal: np.ndarray,
    sample_rate: int,
    detection_config: MirrorDetectionConfig | None = None,
    suppression_floor: float = DEFAULT_SUPPRESSION_FLOOR,
    energy_cap: float = DEFAULT_ENERGY_CAP,
    envelope_min: float = DEFAULT_ENVELOPE_MIN,
) -> HBTargetResult:
    """Generate HB_target by suppressing mirror artifacts.

    Args:
        hb_signal: High-band input signal (HB_in).
        sample_rate: Sample rate in Hz.
        detection_config: Optional mirror detection configuration.
        suppression_floor: Minimum gain applied to detected bins.
        energy_cap: Maximum allowed high-band mean energy.
        envelope_min: Minimum envelope gain at Nyquist.

    Returns:
        HBTargetResult containing target signal and masks.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Mirror artifacts are attenuated while enforcing a gentle spectral
        decay and fixed energy cap for IMD safety.
    """

    _validate_signal(hb_signal)
    _validate_sample_rate(sample_rate)
    _validate_unit_interval(suppression_floor, "suppression_floor")
    _validate_positive_float(energy_cap, "energy_cap")
    _validate_unit_interval(envelope_min, "envelope_min")

    detection = detect_mirror_artifacts(hb_signal, sample_rate, detection_config)
    suppression_mask = np.ones_like(detection.magnitude, dtype=np.float64)
    suppression_mask[detection.detection_mask] = suppression_floor

    suppressed_stft = detection.stft * suppression_mask

    envelope = _build_envelope(
        detection.freqs,
        cutoff_hz=_resolve_cutoff(detection_config, DEFAULT_CUTOFF_HZ),
        envelope_min=envelope_min,
    )
    shaped_stft = suppressed_stft * envelope[:, None]

    highband_mask = detection.freqs >= _resolve_cutoff(
        detection_config, DEFAULT_CUTOFF_HZ
    )
    energy = float(np.mean(np.abs(shaped_stft[highband_mask]) ** 2))
    energy_scale = 1.0
    if energy > energy_cap:
        energy_scale = float(np.sqrt(energy_cap / max(energy, 1e-12)))
        shaped_stft = _scale_highband(shaped_stft, highband_mask, energy_scale)

    target = _compute_istft(
        shaped_stft,
        sample_rate,
        n_fft=_resolve_n_fft(detection_config),
        hop_length=_resolve_hop_length(detection_config),
        window=_resolve_window(detection_config),
        target_length=hb_signal.shape[-1],
    )

    return HBTargetResult(
        target=target,
        detection=detection,
        suppression_mask=suppression_mask,
        envelope=envelope,
        energy_scale=energy_scale,
    )


def project_teacher_hb_target(
    hb_in: np.ndarray,
    teacher_hb: np.ndarray,
    sample_rate: int,
    *,
    detection_config: MirrorDetectionConfig | None = None,
    suppression_floor: float = DEFAULT_SUPPRESSION_FLOOR,
    energy_cap: float = DEFAULT_ENERGY_CAP,
    envelope_min: float = DEFAULT_ENVELOPE_MIN,
) -> np.ndarray:
    """Project teacher HB magnitude onto input phase to build a learnable target.

    Args:
        hb_in: High-band input signal derived from the degradation path (HB_in).
        teacher_hb: High-band teacher signal (raw 88.2kHz reference split to HB).
        sample_rate: Sample rate in Hz.
        detection_config: Optional STFT/envelope configuration (reuses MirrorDetectionConfig).
        suppression_floor: Maximum retained gain in detected mirror bins.
        energy_cap: Maximum allowed high-band mean energy (IMD safety gate).
        envelope_min: Minimum envelope gain at Nyquist.

    Returns:
        Time-domain HB target signal aligned to `hb_in` length.

    Physical Basis:
        Stage 1 NMSE applies attenuation masks in STFT magnitude while keeping
        the input phase. Using a raw teacher directly as a waveform target can
        over-constrain phase and can demand energy that is not present in the
        degraded input (e.g., >22.05kHz content lost by 44.1kHz downsampling).
        This function builds a target that:
        - follows the teacher magnitude where feasible (prevents over-suppression),
        - never exceeds the input magnitude (prevents hallucinated addition),
        - applies explicit mirror-bin attenuation with suppression_floor,
        - retains the input phase (aligns with the model's constraint),
        - applies envelope and energy caps for HF safety.
    """
    _validate_signal(hb_in)
    _validate_signal(teacher_hb)
    _validate_sample_rate(sample_rate)
    _validate_unit_interval(suppression_floor, "suppression_floor")
    _validate_positive_float(energy_cap, "energy_cap")
    _validate_unit_interval(envelope_min, "envelope_min")
    if hb_in.shape != teacher_hb.shape:
        raise ValueError("hb_in and teacher_hb must share shape.")

    cfg = detection_config or MirrorDetectionConfig()

    freqs, _, stft_in = _compute_stft(
        hb_in,
        sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=cfg.window,
    )
    _, _, stft_teacher = _compute_stft(
        teacher_hb,
        sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=cfg.window,
    )
    if stft_in.shape != stft_teacher.shape:
        raise ValueError("STFT shapes must match for projection.")

    mag_in = np.abs(stft_in)
    mag_teacher = np.abs(stft_teacher)
    mag_target = np.minimum(mag_teacher, mag_in)
    mirror_detection = detect_mirror_artifacts(
        hb_in,
        sample_rate,
        config=cfg,
    )
    mirror_mask = mirror_detection.detection_mask
    if mirror_mask.shape != mag_target.shape:
        raise ValueError("mirror detection mask shape mismatch.")
    mag_target[mirror_mask] = np.minimum(
        mag_target[mirror_mask],
        mag_in[mirror_mask] * suppression_floor,
    )

    phase_in = np.exp(1j * np.angle(stft_in))
    projected_stft = mag_target * phase_in

    envelope = _build_envelope(
        np.asarray(freqs, dtype=np.float64),
        cutoff_hz=float(cfg.cutoff_hz),
        envelope_min=float(envelope_min),
    )
    shaped_stft = projected_stft * envelope[:, None]

    highband_mask = np.asarray(freqs >= float(cfg.cutoff_hz))
    energy = float(np.mean(np.abs(shaped_stft[highband_mask]) ** 2))
    if energy > energy_cap:
        energy_scale = float(np.sqrt(energy_cap / max(energy, 1e-12)))
        shaped_stft = _scale_highband(shaped_stft, highband_mask, energy_scale)

    return _compute_istft(
        shaped_stft,
        sample_rate,
        n_fft=int(cfg.n_fft),
        hop_length=int(cfg.hop_length),
        window=str(cfg.window),
        target_length=hb_in.shape[-1],
    )


def _compute_stft(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    window: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute STFT for mirror detection.

    Args:
        signal: Time-domain input signal.
        sample_rate: Sample rate in Hz.
        n_fft: FFT size.
        hop_length: Hop length between frames.
        window: Window name.

    Returns:
        Tuple of (freqs, times, stft).

    Raises:
        ValueError: If STFT parameters are invalid.

    Physical Basis:
        Time-frequency analysis exposes symmetric mirror patterns that
        are difficult to isolate in the time domain alone.
    """

    _validate_positive_int(n_fft, "n_fft")
    _validate_positive_int(hop_length, "hop_length")
    if hop_length > n_fft:
        raise ValueError("hop_length must be less than or equal to n_fft.")

    freqs, times, stft = sp_signal.stft(
        signal,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window=window,
        boundary="zeros",
        padded=True,
    )
    return freqs, times, stft


def _compute_istft(
    stft: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    window: str,
    target_length: int,
) -> np.ndarray:
    """Compute inverse STFT and match target length.

    Args:
        stft: Complex STFT array.
        sample_rate: Sample rate in Hz.
        n_fft: FFT size.
        hop_length: Hop length between frames.
        window: Window name.
        target_length: Output length in samples.

    Returns:
        Time-domain signal aligned to target_length.

    Physical Basis:
        iSTFT reconstructs time-domain HB_target after spectral suppression
        while preserving phase relationships from the original STFT.
    """

    _, signal = sp_signal.istft(
        stft,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window=window,
        input_onesided=True,
    )
    return _match_length(signal.astype(np.float32), target_length)


def _match_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Match signal length by trimming or zero-padding.

    Args:
        signal: Input signal.
        target_length: Desired length.

    Returns:
        Signal resized to target_length.

    Physical Basis:
        Length alignment keeps HB_target compatible with HB_in sample count.
    """

    if signal.shape[0] == target_length:
        return signal
    if signal.shape[0] > target_length:
        return signal[:target_length]
    padding = np.zeros(target_length - signal.shape[0], dtype=signal.dtype)
    return np.concatenate([signal, padding])


def _build_envelope(
    freqs: np.ndarray,
    cutoff_hz: float,
    envelope_min: float,
) -> np.ndarray:
    """Build a gentle high-frequency decay envelope.

    Args:
        freqs: STFT frequency bins in Hz.
        cutoff_hz: Cutoff frequency in Hz.
        envelope_min: Minimum gain at Nyquist.

    Returns:
        Frequency-domain envelope.

    Physical Basis:
        A smooth decay beyond 20 kHz discourages excessive HF energy
        without introducing sharp spectral discontinuities.
    """

    nyquist = float(freqs[-1])
    envelope = np.ones_like(freqs, dtype=np.float64)
    highband = freqs >= cutoff_hz
    if nyquist <= cutoff_hz:
        return envelope
    slope = (1.0 - envelope_min) / max(nyquist - cutoff_hz, 1e-6)
    envelope[highband] = 1.0 - slope * (freqs[highband] - cutoff_hz)
    envelope = np.clip(envelope, envelope_min, 1.0)
    return envelope


def _scale_highband(
    stft: np.ndarray,
    highband_mask: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Scale high-band STFT bins by a factor.

    Args:
        stft: Complex STFT array.
        highband_mask: Boolean mask for high-band bins.
        scale: Scaling factor.

    Returns:
        STFT array with scaled high-band bins.

    Physical Basis:
        Uniform scaling enforces energy caps while preserving spectral
        relationships within the high band.
    """

    scaled = np.array(stft, copy=True)
    scaled[highband_mask] = scaled[highband_mask] * scale
    return scaled


def _resolve_mirror_center(mirror_center_hz: float | None, sample_rate: int) -> float:
    """Resolve mirror center frequency.

    Args:
        mirror_center_hz: Optional explicit center frequency.
        sample_rate: Sample rate in Hz.

    Returns:
        Mirror center frequency in Hz.

    Physical Basis:
        For 2× upsampling, the original Nyquist equals sample_rate / 4,
        which is the symmetry center for mirror artifacts.
    """

    if mirror_center_hz is not None:
        return float(mirror_center_hz)
    return float(sample_rate) * DEFAULT_MIRROR_CENTER_RATIO


def _resolve_cutoff(config: MirrorDetectionConfig | None, default: float) -> float:
    """Resolve cutoff frequency from configuration.

    Args:
        config: Optional detection config.
        default: Default cutoff in Hz.

    Returns:
        Cutoff frequency in Hz.

    Physical Basis:
        Consistent cutoff values preserve the low-band identity policy.
    """

    if config is None:
        return default
    return float(config.cutoff_hz)


def _resolve_n_fft(config: MirrorDetectionConfig | None) -> int:
    """Resolve FFT size from configuration.

    Args:
        config: Optional detection config.

    Returns:
        FFT size.

    Physical Basis:
        FFT size controls frequency resolution for mirror symmetry checks.
    """

    if config is None:
        return DEFAULT_N_FFT
    return int(config.n_fft)


def _resolve_hop_length(config: MirrorDetectionConfig | None) -> int:
    """Resolve hop length from configuration.

    Args:
        config: Optional detection config.

    Returns:
        Hop length in samples.

    Physical Basis:
        Hop length defines time resolution in the STFT representation.
    """

    if config is None:
        return DEFAULT_HOP_LENGTH
    return int(config.hop_length)


def _resolve_window(config: MirrorDetectionConfig | None) -> str:
    """Resolve STFT window name from configuration.

    Args:
        config: Optional detection config.

    Returns:
        Window name.

    Physical Basis:
        Consistent windowing controls leakage in STFT magnitude features.
    """

    if config is None:
        return DEFAULT_WINDOW
    return str(config.window)


def _validate_detection_config(
    config: MirrorDetectionConfig,
    sample_rate: int,
    mirror_center_hz: float,
) -> None:
    """Validate mirror detection configuration.

    Args:
        config: Mirror detection configuration.
        sample_rate: Sample rate in Hz.
        mirror_center_hz: Resolved mirror center frequency in Hz.

    Raises:
        ValueError: If configuration values are invalid.

    Physical Basis:
        Ensuring valid bounds keeps detection within the physically
        meaningful high-band region and avoids aliasing artifacts.
    """

    _validate_positive_float(config.cutoff_hz, "cutoff_hz")
    _validate_positive_float(config.magnitude_threshold, "magnitude_threshold")
    _validate_unit_interval(config.symmetry_threshold, "symmetry_threshold")
    _validate_positive_int(config.n_fft, "n_fft")
    _validate_positive_int(config.hop_length, "hop_length")
    if config.hop_length > config.n_fft:
        raise ValueError("hop_length must be less than or equal to n_fft.")

    nyquist = sample_rate / 2.0
    if config.cutoff_hz >= nyquist:
        raise ValueError("cutoff_hz must be below Nyquist.")
    _validate_positive_float(mirror_center_hz, "mirror_center_hz")
    if mirror_center_hz >= nyquist:
        raise ValueError("mirror_center_hz must be below Nyquist.")

    band_low, band_high = config.mirror_band_hz
    _validate_positive_float(band_low, "mirror_band_hz[0]")
    _validate_positive_float(band_high, "mirror_band_hz[1]")
    if band_low >= band_high:
        raise ValueError("mirror_band_hz[0] must be less than mirror_band_hz[1].")
    if band_low < config.cutoff_hz:
        raise ValueError("mirror_band_hz must be within the high-band region.")
    if band_high > mirror_center_hz:
        raise ValueError("mirror_band_hz must be at or below mirror_center_hz.")


def _validate_signal(signal: np.ndarray) -> None:
    """Validate input signal shape.

    Args:
        signal: Input signal array.

    Raises:
        ValueError: If signal shape is invalid.

    Physical Basis:
        Mirror detection assumes a single-channel time series.
    """

    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got {signal.ndim}D.")
    if signal.size == 0:
        raise ValueError("signal must not be empty.")


def _validate_sample_rate(sample_rate: int) -> None:
    """Validate sample rate.

    Args:
        sample_rate: Sample rate in Hz.

    Raises:
        ValueError: If sample_rate is invalid.

    Physical Basis:
        Positive sample rates ensure meaningful discrete-time analysis.
    """

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")


def _validate_positive_int(value: int, name: str) -> None:
    """Validate a positive integer parameter.

    Args:
        value: Integer value.
        name: Parameter name.

    Raises:
        ValueError: If value is invalid.

    Physical Basis:
        Positive integers are required for discrete-time configuration.
    """

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_positive_float(value: float, name: str) -> None:
    """Validate a positive float parameter.

    Args:
        value: Floating-point value.
        name: Parameter name.

    Raises:
        ValueError: If value is invalid.

    Physical Basis:
        Positive magnitudes ensure physically meaningful thresholds.
    """

    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_unit_interval(value: float, name: str) -> None:
    """Validate a parameter in [0, 1].

    Args:
        value: Floating-point value.
        name: Parameter name.

    Raises:
        ValueError: If value is invalid.

    Physical Basis:
        Gains and ratios are constrained to physically valid ranges.
    """

    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")


__all__ = [
    "HBTargetResult",
    "MirrorDetectionConfig",
    "MirrorDetectionResult",
    "detect_mirror_artifacts",
    "generate_hb_target",
]
