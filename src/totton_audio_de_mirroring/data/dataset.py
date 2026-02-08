"""Dataset pipeline for mirror suppression training data."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from totton_audio_de_mirroring.data.degradation import (
    DegradationProfileManager,
    apply_degradation_profile,
)
from totton_audio_de_mirroring.data.generator import (
    SignalRequest,
    apply_soft_clip,
    generate_signal,
)
from totton_audio_de_mirroring.data.mirror_detection import (
    generate_hb_target,
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

        profile = self._degradation.sample_profile(rng=rng)
        x_full = apply_degradation_profile(
            source_chunk,
            self._config.source_sample_rate,
            self._config.target_sample_rate,
            profile,
            rng,
        )

        low_band, high_band = self._band_split.split(x_full)

        hb_target_result = generate_hb_target(
            high_band,
            self._config.target_sample_rate,
            detection_config=self._config.mirror_detection,
            suppression_floor=self._config.hb_target.suppression_floor,
            energy_cap=self._config.hb_target.energy_cap,
            envelope_min=self._config.hb_target.envelope_min,
        )
        mirror_mask = _to_tensor_2d(
            hb_target_result.detection.detection_mask.astype(np.float32)
        )

        sample = {
            "source": _to_tensor(source_chunk),
            "x_full": _to_tensor(x_full),
            "low_band": _to_tensor(low_band),
            "high_band": _to_tensor(high_band),
            "hb_target": _to_tensor(hb_target_result.target),
            "mirror_mask": mirror_mask,
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
