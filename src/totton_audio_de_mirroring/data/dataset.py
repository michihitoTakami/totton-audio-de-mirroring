"""Dataset pipeline for mirror suppression training data."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.degradation import (
    DegradationProfileManager,
    apply_degradation_profile,
    upsample_bessel_reference,
)
from totton_audio_de_mirroring.data.generator import (
    SignalRequest,
    apply_soft_clip,
    generate_signal,
)
from totton_audio_de_mirroring.data.mirror_detection import (
    detect_mirror_artifacts,
    generate_hb_target,
    project_teacher_hb_target,
)
from totton_audio_de_mirroring.data.pipeline_config import (
    AugmentationConfig,
    DataPipelineConfig,
    SignalSamplingConfig,
)
from totton_audio_de_mirroring.models.band_split import (
    BandSplitConfig,
    BandSplitProcessor,
)


class MirrorSuppressionDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """PyTorch Dataset for mirror suppression training data.

    Args:
        config: Data pipeline configuration.

    Physical Basis:
        Each item follows the pipeline: synthetic source -> degradation SRC
        -> band split -> mirror-suppressed target generation.
    """

    def __init__(self, config: DataPipelineConfig) -> None:
        _validate_pipeline_config(config)
        _validate_stage1_path(config)

        self._config = config
        self._degradation = DegradationProfileManager(config.degradation)
        self._band_split = BandSplitProcessor(
            _replace_band_split_sample_rate(
                config.band_split, config.target_sample_rate
            )
        )
        self._cache = (
            _LRUCache(config.cache.max_items) if config.cache.enabled else None
        )
        self._base_seed = (
            config.seed
            if config.seed is not None
            else int(np.random.SeedSequence().generate_state(1)[0])
        )

    def __len__(self) -> int:
        """Return dataset length.

        Physical Basis:
            Dataset length defines the number of synthetic training samples.
        """
        return self._config.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a training sample by index.

        Args:
            index: Sample index.

        Returns:
            Dictionary containing tensors and metadata.

        Raises:
            IndexError: If index is out of range.

        Physical Basis:
            Deterministic sampling by index supports reproducibility and
            caching without mutating stored signals.
        """
        if index < 0 or index >= self._config.num_samples:
            raise IndexError("index out of range")

        if self._cache is not None:
            cached = self._cache.get(index)
            if cached is not None:
                return cached

        rng = self._rng_for_index(index)
        request = _sample_signal_request(rng, self._config.signal_sampling)
        source_seed = int(rng.integers(0, 2**32 - 1))
        source_chunk: np.ndarray
        teacher_full: np.ndarray
        chunk_start: int
        if self._config.teacher_type == "raw_88k2":
            source_chunk, teacher_full, chunk_start = (
                _build_raw_teacher_source_chunk_and_reference(
                    request=request,
                    source_seed=source_seed,
                    source_sr=self._config.source_sample_rate,
                    target_sr=self._config.target_sample_rate,
                    source_duration_sec=self._config.source_duration_sec,
                    chunk_duration_sec=self._config.chunk_duration_sec,
                    random_chunk=self._config.random_chunk,
                    augmentation=self._config.augmentation,
                    rng=rng,
                )
            )
        else:
            source = generate_signal(
                request.signal_type,
                sample_rate=self._config.source_sample_rate,
                duration_sec=self._config.source_duration_sec,
                seed=source_seed,
                **dict(request.params),
            )
            source_chunk, chunk_start = _extract_chunk(
                source,
                self._config.source_sample_rate,
                self._config.source_duration_sec,
                self._config.chunk_duration_sec,
                self._config.random_chunk,
                rng,
            )
            source_chunk = apply_augmentations(
                source_chunk,
                self._config.augmentation,
                rng,
            )
            teacher_full = _build_teacher_reference(
                source_chunk,
                source_sr=self._config.source_sample_rate,
                target_sr=self._config.target_sample_rate,
                teacher_type=self._config.teacher_type,
                bessel_cutoff_hz=self._config.band_split.cutoff_hz,
                bessel_order=self._config.degradation.iir_order,
            )

        profile = self._degradation.sample_profile(rng=rng)
        x_full = apply_degradation_profile(
            source_chunk,
            self._config.source_sample_rate,
            self._config.target_sample_rate,
            profile,
            rng,
        )

        low_band, high_band = self._band_split.split(x_full)
        _, teacher_high_band = self._band_split.split(teacher_full)

        if self._config.teacher_type == "raw_88k2":
            hb_target = project_teacher_hb_target(
                high_band,
                teacher_high_band,
                self._config.target_sample_rate,
                detection_config=self._config.mirror_detection,
                energy_cap=self._config.hb_target.energy_cap,
                envelope_min=self._config.hb_target.envelope_min,
            )
        else:
            hb_target_result = generate_hb_target(
                teacher_high_band,
                self._config.target_sample_rate,
                detection_config=self._config.mirror_detection,
                suppression_floor=self._config.hb_target.suppression_floor,
                energy_cap=self._config.hb_target.energy_cap,
                envelope_min=self._config.hb_target.envelope_min,
            )
            hb_target = hb_target_result.target
        detection = detect_mirror_artifacts(
            high_band,
            self._config.target_sample_rate,
            config=self._config.mirror_detection,
        )
        _validate_training_sample_consistency(
            source=source_chunk,
            x_full=x_full,
            teacher_full=teacher_full,
            low_band=low_band,
            high_band=high_band,
            hb_target=hb_target,
            source_sr=self._config.source_sample_rate,
            target_sr=self._config.target_sample_rate,
            chunk_duration_sec=self._config.chunk_duration_sec,
        )
        mirror_mask = _to_tensor_2d(detection.detection_mask.astype(np.float32))

        sample = {
            "source": _to_tensor(source_chunk),
            "x_full": _to_tensor(x_full),
            "low_band": _to_tensor(low_band),
            "high_band": _to_tensor(high_band),
            "hb_target": _to_tensor(hb_target),
            "mirror_mask": mirror_mask,
            "teacher_type": self._config.teacher_type,
            "input_route": self._config.stage1_path.input_route,
            "target_route": self._config.stage1_path.target_route,
            "profile": profile,
            "signal_type": request.signal_type,
            "chunk_start": int(chunk_start),
        }

        if self._cache is not None:
            self._cache.set(index, sample)

        return sample

    def _rng_for_index(self, index: int) -> np.random.Generator:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed = int(self._base_seed + index + worker_id * 1_000_000)
        return np.random.default_rng(seed)


def apply_augmentations(
    signal: np.ndarray,
    config: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply on-the-fly augmentations to a signal.

    Args:
        signal: Input signal.
        config: Augmentation configuration.
        rng: RNG for reproducibility.

    Returns:
        Augmented signal.

    Physical Basis:
        Controlled gain, noise, and saturation improve robustness without
        violating Nyquist-limited content.
    """
    _validate_signal(signal)
    _validate_rng(rng)

    gain = rng.uniform(config.gain_range[0], config.gain_range[1])
    augmented = signal * gain

    if rng.random() < config.polarity_flip_prob:
        augmented = -augmented

    noise_std = rng.uniform(config.noise_std_range[0], config.noise_std_range[1])
    if noise_std > 0.0:
        noise = rng.normal(0.0, noise_std, size=signal.shape)
        augmented = augmented + noise

    if rng.random() < config.soft_clip_prob:
        drive = rng.uniform(
            config.soft_clip_drive_range[0], config.soft_clip_drive_range[1]
        )
        augmented = apply_soft_clip(augmented, drive=drive)

    return np.asarray(augmented, dtype=np.float32)


def _replace_band_split_sample_rate(
    config: BandSplitConfig, sample_rate: int
) -> BandSplitConfig:
    if config.sample_rate == sample_rate:
        return config
    return BandSplitConfig(
        cutoff_hz=config.cutoff_hz,
        sample_rate=sample_rate,
        num_taps=config.num_taps,
        window=config.window,
    )


def _sample_signal_request(
    rng: np.random.Generator,
    config: SignalSamplingConfig,
) -> SignalRequest:
    _validate_rng(rng)
    _validate_non_empty(config.signal_types, "signal_types")

    signal_type = str(rng.choice(config.signal_types))
    params: dict[str, float | int | Sequence[float]] = {}

    if signal_type == "multitone":
        count = int(
            rng.integers(
                config.multitone_count_range[0], config.multitone_count_range[1] + 1
            )
        )
        freqs = rng.uniform(
            config.frequency_range_hz[0], config.frequency_range_hz[1], size=count
        )
        params = {"frequencies_hz": np.sort(freqs).tolist()}
    elif signal_type == "sweep_linear" or signal_type == "sweep_log":
        start_min = config.sweep_start_range_hz[0]
        start_max = min(
            config.sweep_start_range_hz[1],
            config.sweep_end_range_hz[1] - 100.0,
        )
        if start_max < start_min:
            start_max = start_min
        start = rng.uniform(start_min, start_max)
        end_low = max(start + 100.0, config.sweep_end_range_hz[0])
        end = rng.uniform(end_low, config.sweep_end_range_hz[1])
        params = {"start_hz": float(start), "end_hz": float(end)}
    elif signal_type == "impulse_train":
        interval = rng.uniform(
            config.impulse_interval_range_sec[0],
            config.impulse_interval_range_sec[1],
        )
        params = {"interval_sec": float(interval)}
    elif signal_type == "percussive":
        decay = rng.uniform(
            config.percussive_decay_range[0],
            config.percussive_decay_range[1],
        )
        params = {"decay_rate": float(decay)}
    elif signal_type == "am_tone":
        carrier = rng.uniform(
            config.am_carrier_range_hz[0], config.am_carrier_range_hz[1]
        )
        mod = rng.uniform(config.am_mod_range_hz[0], config.am_mod_range_hz[1])
        index = rng.uniform(config.am_mod_index_range[0], config.am_mod_index_range[1])
        params = {
            "carrier_hz": float(carrier),
            "mod_hz": float(mod),
            "modulation_index": float(index),
        }
    elif signal_type == "fm_tone":
        carrier = rng.uniform(
            config.fm_carrier_range_hz[0], config.fm_carrier_range_hz[1]
        )
        mod = rng.uniform(config.fm_mod_range_hz[0], config.fm_mod_range_hz[1])
        index = rng.uniform(config.fm_mod_index_range[0], config.fm_mod_index_range[1])
        params = {
            "carrier_hz": float(carrier),
            "mod_hz": float(mod),
            "modulation_index": float(index),
        }
    elif signal_type == "band_limited_noise":
        low_min = config.band_noise_low_range_hz[0]
        low_max = min(
            config.band_noise_low_range_hz[1],
            config.band_noise_high_range_hz[1] - 500.0,
        )
        if low_max < low_min:
            low_max = low_min
        low = rng.uniform(low_min, low_max)
        high_min = max(low + 500.0, config.band_noise_high_range_hz[0])
        high_max = config.band_noise_high_range_hz[1]
        if high_min > high_max:
            high_min = high_max
        high = rng.uniform(high_min, high_max)
        params = {"low_hz": float(low), "high_hz": float(high)}
    elif signal_type == "soft_clipped_tone":
        freq = rng.uniform(config.frequency_range_hz[0], config.frequency_range_hz[1])
        drive = rng.uniform(
            config.soft_clip_drive_range[0], config.soft_clip_drive_range[1]
        )
        params = {"frequency_hz": float(freq), "drive": float(drive)}

    return SignalRequest(signal_type=signal_type, params=params)


def _extract_chunk(
    source: np.ndarray,
    sample_rate: int,
    duration_sec: float,
    chunk_duration_sec: float,
    random_chunk: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    _validate_signal(source)
    _validate_positive_int(sample_rate, "sample_rate")
    _validate_positive_float(duration_sec, "duration_sec")
    _validate_positive_float(chunk_duration_sec, "chunk_duration_sec")
    _validate_rng(rng)

    total_samples = int(source.shape[0])
    expected_samples = int(round(duration_sec * sample_rate))
    if abs(total_samples - expected_samples) > 1:
        raise ValueError("source length does not match duration settings.")
    chunk_samples = int(round(chunk_duration_sec * sample_rate))
    if chunk_samples <= 0 or total_samples <= 0:
        raise ValueError("duration settings must yield positive sample counts.")
    if chunk_samples > total_samples:
        raise ValueError("chunk_duration_sec exceeds source duration.")

    max_start = total_samples - chunk_samples
    if max_start <= 0 or not random_chunk:
        start = 0
    else:
        start = int(rng.integers(0, max_start + 1))

    return source[start : start + chunk_samples], start


def _build_teacher_reference(
    signal: np.ndarray,
    *,
    source_sr: int,
    target_sr: int,
    teacher_type: str,
    bessel_cutoff_hz: float,
    bessel_order: int,
) -> np.ndarray:
    """Build Stage 1 teacher reference at target sample rate.

    Args:
        signal: Source chunk at source sample rate.
        source_sr: Source sample rate.
        target_sr: Target sample rate.
        teacher_type: Teacher reference type.
        bessel_cutoff_hz: Bessel low-pass cutoff for bessel teacher mode.
        bessel_order: Bessel IIR order for bessel teacher mode.

    Returns:
        Teacher reference signal at target sample rate.

    Raises:
        ValueError: If teacher_type is unsupported.

    Physical Basis:
        Stage 1 uses degraded `x_full` as input while supervision is derived
        from an explicit teacher reference path (`raw_88k2` or `bessel_88k2`).
    """
    _validate_signal(signal)
    _validate_positive_int(source_sr, "source_sr")
    _validate_positive_int(target_sr, "target_sr")

    if teacher_type == "raw_88k2":
        return _upsample_raw_reference(signal, source_sr=source_sr, target_sr=target_sr)
    if teacher_type == "bessel_88k2":
        return upsample_bessel_reference(
            signal=signal,
            source_sr=source_sr,
            target_sr=target_sr,
            cutoff_hz=bessel_cutoff_hz,
            order=bessel_order,
        )
    raise ValueError(f"Unsupported teacher_type: {teacher_type!r}.")


def _build_raw_teacher_source_chunk_and_reference(
    *,
    request: SignalRequest,
    source_seed: int,
    source_sr: int,
    target_sr: int,
    source_duration_sec: float,
    chunk_duration_sec: float,
    random_chunk: bool,
    augmentation: AugmentationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build source chunk and teacher reference for the raw Stage 1 policy.

    Args:
        request: Synthetic signal request.
        source_seed: Reproducible seed for source generation.
        source_sr: Source sample rate.
        target_sr: Target sample rate.
        source_duration_sec: Full source duration in seconds.
        chunk_duration_sec: Chunk duration in seconds.
        random_chunk: Whether to sample chunk start randomly.
        augmentation: Augmentation configuration.
        rng: RNG for deterministic chunking and augmentation.

    Returns:
        Tuple of `(source_chunk, teacher_chunk, chunk_start_samples_at_source_sr)`.

    Physical Basis:
        Raw teacher supervision must carry genuine >22.05kHz information, so
        the teacher is synthesized natively at 88.2kHz. The 44.1kHz input
        chunk is then derived by downsampling the same teacher chunk.
    """
    _validate_rng(rng)
    _validate_positive_int(source_sr, "source_sr")
    _validate_positive_int(target_sr, "target_sr")
    _validate_positive_float(source_duration_sec, "source_duration_sec")
    _validate_positive_float(chunk_duration_sec, "chunk_duration_sec")

    teacher_source = generate_signal(
        request.signal_type,
        sample_rate=target_sr,
        duration_sec=source_duration_sec,
        seed=source_seed,
        **dict(request.params),
    )
    source_proxy = _downsample_raw_reference(
        teacher_source, source_sr=target_sr, target_sr=source_sr
    )
    _, chunk_start = _extract_chunk(
        source_proxy,
        source_sr,
        source_duration_sec,
        chunk_duration_sec,
        random_chunk,
        rng,
    )

    ratio = target_sr / source_sr
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError("target_sr must be an integer multiple of source_sr.")
    int_ratio = int(round(ratio))
    target_chunk_samples = int(round(chunk_duration_sec * target_sr))
    target_start = chunk_start * int_ratio
    teacher_chunk = teacher_source[target_start : target_start + target_chunk_samples]
    if teacher_chunk.shape[0] != target_chunk_samples:
        raise ValueError("teacher chunk extraction failed to match target length.")

    teacher_chunk = apply_augmentations(teacher_chunk, augmentation, rng)
    source_chunk = _downsample_raw_reference(
        teacher_chunk, source_sr=target_sr, target_sr=source_sr
    )
    return source_chunk, teacher_chunk, chunk_start


def _upsample_raw_reference(
    signal: np.ndarray, *, source_sr: int, target_sr: int
) -> np.ndarray:
    """Upsample via high-quality polyphase SRC for raw teacher references.

    Args:
        signal: Source chunk.
        source_sr: Source sample rate.
        target_sr: Target sample rate.

    Returns:
        Raw reference upsampled signal at target sample rate.

    Physical Basis:
        Polyphase sinc-style interpolation provides a neutral 2x reference
        path without the Bessel teacher coloration.
    """
    _validate_signal(signal)
    _validate_positive_int(source_sr, "source_sr")
    _validate_positive_int(target_sr, "target_sr")
    ratio = target_sr / source_sr
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError("target_sr must be an integer multiple of source_sr.")
    int_ratio = int(round(ratio))
    if int_ratio <= 0:
        raise ValueError("upsampling ratio must be positive.")

    upsampled = sp_signal.resample_poly(
        np.asarray(signal, dtype=np.float64),
        up=int_ratio,
        down=1,
        axis=-1,
        window=("kaiser", 8.6),
    )
    expected_len = signal.shape[-1] * int_ratio
    return np.asarray(upsampled[..., :expected_len], dtype=np.float64)


def _downsample_raw_reference(
    signal: np.ndarray, *, source_sr: int, target_sr: int
) -> np.ndarray:
    """Downsample via high-quality polyphase SRC for raw-teacher alignment.

    Args:
        signal: Source signal.
        source_sr: Source sample rate.
        target_sr: Target sample rate.

    Returns:
        Downsampled signal at target sample rate.

    Physical Basis:
        A controlled polyphase downsampling path converts native 88.2kHz
        teacher chunks into 44.1kHz inputs while preserving corresponding
        time structure for degradation-path synthesis.
    """
    _validate_signal(signal)
    _validate_positive_int(source_sr, "source_sr")
    _validate_positive_int(target_sr, "target_sr")
    ratio = source_sr / target_sr
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError("source_sr must be an integer multiple of target_sr.")
    int_ratio = int(round(ratio))
    if int_ratio <= 0:
        raise ValueError("downsampling ratio must be positive.")

    downsampled = sp_signal.resample_poly(
        np.asarray(signal, dtype=np.float64),
        up=1,
        down=int_ratio,
        axis=-1,
        window=("kaiser", 8.6),
    )
    expected_len = int(round(signal.shape[-1] / int_ratio))
    return np.asarray(downsampled[..., :expected_len], dtype=np.float64)


def _validate_training_sample_consistency(
    *,
    source: np.ndarray,
    x_full: np.ndarray,
    teacher_full: np.ndarray,
    low_band: np.ndarray,
    high_band: np.ndarray,
    hb_target: np.ndarray,
    source_sr: int,
    target_sr: int,
    chunk_duration_sec: float,
) -> None:
    """Validate generated training sample consistency before tensor export.

    Args:
        source: Source chunk at source sample rate.
        x_full: Degraded Stage 1 input at target sample rate.
        teacher_full: Teacher reference at target sample rate.
        low_band: Low-band split from x_full.
        high_band: High-band split from x_full.
        hb_target: Generated high-band target.
        source_sr: Source sample rate.
        target_sr: Target sample rate.
        chunk_duration_sec: Chunk duration in seconds.

    Raises:
        ValueError: If rates, lengths, channel rank, or peaks are inconsistent.

    Physical Basis:
        Stage 1 supervision requires strict alignment between input and target
        timelines; SR/length/channel/peak checks prevent silent data drift.
    """
    _validate_signal(source)
    _validate_signal(x_full)
    _validate_signal(teacher_full)
    _validate_signal(low_band)
    _validate_signal(high_band)
    _validate_signal(hb_target)
    _validate_positive_int(source_sr, "source_sr")
    _validate_positive_int(target_sr, "target_sr")
    _validate_positive_float(chunk_duration_sec, "chunk_duration_sec")

    expected_source_len = int(round(chunk_duration_sec * source_sr))
    expected_target_len = int(round(chunk_duration_sec * target_sr))
    if source.shape[-1] != expected_source_len:
        raise ValueError(
            "source length mismatch: "
            f"expected {expected_source_len}, got {source.shape[-1]}."
        )

    for name, signal in (
        ("x_full", x_full),
        ("teacher_full", teacher_full),
        ("low_band", low_band),
        ("high_band", high_band),
        ("hb_target", hb_target),
    ):
        if signal.shape[-1] != expected_target_len:
            raise ValueError(
                f"{name} length mismatch: expected {expected_target_len}, "
                f"got {signal.shape[-1]}."
            )
        if not np.all(np.isfinite(signal)):
            raise ValueError(f"{name} contains non-finite values.")
        peak = float(np.max(np.abs(signal)))
        if peak > 4.0:
            raise ValueError(f"{name} peak is too large: {peak:.6f} > 4.0.")

    if not np.all(np.isfinite(source)):
        raise ValueError("source contains non-finite values.")
    source_peak = float(np.max(np.abs(source)))
    if source_peak > 4.0:
        raise ValueError(f"source peak is too large: {source_peak:.6f} > 4.0.")


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    _validate_signal(array)
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def _to_tensor_2d(array: np.ndarray) -> torch.Tensor:
    """Convert a 2D NumPy array to a float32 torch tensor.

    Args:
        array: 2D input array.

    Returns:
        Torch tensor copy of the input.

    Physical Basis:
        Preserving the STFT time-frequency mask as a 2D tensor keeps
        mirror-region annotations aligned with frequency and time bins.
    """
    _validate_2d_array(array)
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def _validate_pipeline_config(config: DataPipelineConfig) -> None:
    if not isinstance(config, DataPipelineConfig):
        raise ValueError("config must be a DataPipelineConfig")


def _validate_stage1_path(config: DataPipelineConfig) -> None:
    """Validate Stage 1 input/target route consistency.

    Args:
        config: Data pipeline configuration to validate.

    Raises:
        ValueError: If strict route requirements are violated.

    Physical Basis:
        Stage 1 training assumes a fixed 2x path where `x_full` is formed
        from the 44.1kHz chunk and `hb_target` is derived from `high_band`.
    """
    if not config.stage1_path.strict_route_validation:
        return
    if config.target_sample_rate != config.source_sample_rate * 2:
        raise ValueError(
            "strict stage1_path requires target_sample_rate = source_sample_rate * 2."
        )


def _validate_signal(signal: np.ndarray) -> None:
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got {signal.ndim}D")
    if signal.size == 0:
        raise ValueError("signal must be non-empty")


def _validate_2d_array(array: np.ndarray) -> None:
    """Validate that an array is 2D and non-empty.

    Args:
        array: Array to validate.

    Physical Basis:
        Mirror masks are defined over (frequency, time) grids and must
        be strictly 2D to align with STFT magnitudes.
    """
    if array.ndim != 2:
        raise ValueError(f"array must be 2D, got {array.ndim}D")
    if array.size == 0:
        raise ValueError("array must be non-empty")


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_positive_float(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _validate_non_empty(values: Sequence[Any], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must be non-empty.")


def _validate_rng(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")


class _LRUCache:
    def __init__(self, max_items: int) -> None:
        _validate_positive_int(max_items, "max_items")
        self._max_items = max_items
        self._store: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def get(self, key: int) -> dict[str, Any] | None:
        if key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value
        return value

    def set(self, key: int, value: dict[str, Any]) -> None:
        if key in self._store:
            self._store.pop(key)
        self._store[key] = value
        if len(self._store) > self._max_items:
            self._store.popitem(last=False)
