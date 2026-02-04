"""Dataset pipeline for mirror suppression training data."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from totton_audio_de_mirroring.data.degradation import (
    DegradationConfig,
    DegradationProfileManager,
    apply_degradation_profile,
)
from totton_audio_de_mirroring.data.generator import (
    SignalRequest,
    apply_soft_clip,
    generate_signal,
    list_signal_types,
)
from totton_audio_de_mirroring.data.mirror_detection import (
    MirrorDetectionConfig,
    generate_hb_target,
)
from totton_audio_de_mirroring.models.band_split import (
    BandSplitConfig,
    BandSplitProcessor,
)

DEFAULT_SOURCE_SR = 44_100
DEFAULT_TARGET_SR = 88_200
DEFAULT_DURATION_SEC = 1.0
DEFAULT_CHUNK_SEC = 0.25
DEFAULT_NUM_SAMPLES = 10_000
DEFAULT_CACHE_ITEMS = 128
DEFAULT_SEED = None


@dataclass(frozen=True)
class SignalSamplingConfig:
    """Configuration for synthetic signal parameter sampling.

    Args:
        signal_types: Signal generator names to sample.
        multitone_count_range: Range for number of tones.
        frequency_range_hz: Range for general tone frequencies.
        sweep_start_range_hz: Start frequency range for sweeps.
        sweep_end_range_hz: End frequency range for sweeps.
        impulse_interval_range_sec: Interval range for impulse trains.
        percussive_decay_range: Decay rate range for percussive signals.
        am_carrier_range_hz: Carrier frequency range for AM tones.
        am_mod_range_hz: Modulator frequency range for AM tones.
        am_mod_index_range: Modulation index range for AM tones.
        fm_carrier_range_hz: Carrier frequency range for FM tones.
        fm_mod_range_hz: Modulator frequency range for FM tones.
        fm_mod_index_range: Modulation index range for FM tones.
        band_noise_low_range_hz: Low cutoff range for band-limited noise.
        band_noise_high_range_hz: High cutoff range for band-limited noise.
        soft_clip_drive_range: Drive range for soft-clipped tones.

    Physical Basis:
        Sampling diverse signal parameters ensures broad spectral coverage
        without introducing non-physical ultrasonic content.
    """

    signal_types: tuple[str, ...] = field(default_factory=list_signal_types)
    multitone_count_range: tuple[int, int] = (2, 6)
    frequency_range_hz: tuple[float, float] = (40.0, 18_000.0)
    sweep_start_range_hz: tuple[float, float] = (20.0, 2_000.0)
    sweep_end_range_hz: tuple[float, float] = (5_000.0, 20_000.0)
    impulse_interval_range_sec: tuple[float, float] = (0.005, 0.05)
    percussive_decay_range: tuple[float, float] = (6.0, 20.0)
    am_carrier_range_hz: tuple[float, float] = (500.0, 10_000.0)
    am_mod_range_hz: tuple[float, float] = (20.0, 400.0)
    am_mod_index_range: tuple[float, float] = (0.2, 0.8)
    fm_carrier_range_hz: tuple[float, float] = (200.0, 8_000.0)
    fm_mod_range_hz: tuple[float, float] = (20.0, 800.0)
    fm_mod_index_range: tuple[float, float] = (1.0, 6.0)
    band_noise_low_range_hz: tuple[float, float] = (200.0, 4_000.0)
    band_noise_high_range_hz: tuple[float, float] = (6_000.0, 18_000.0)
    soft_clip_drive_range: tuple[float, float] = (1.2, 3.0)

    def __post_init__(self) -> None:
        _validate_non_empty(self.signal_types, "signal_types")
        known = set(list_signal_types())
        for name in self.signal_types:
            if name not in known:
                raise ValueError(f"Unknown signal_type in config: {name}")
        _validate_int_range(self.multitone_count_range, "multitone_count_range")
        _validate_float_range(self.frequency_range_hz, "frequency_range_hz")
        _validate_float_range(self.sweep_start_range_hz, "sweep_start_range_hz")
        _validate_float_range(self.sweep_end_range_hz, "sweep_end_range_hz")
        _validate_float_range(
            self.impulse_interval_range_sec, "impulse_interval_range_sec"
        )
        _validate_float_range(self.percussive_decay_range, "percussive_decay_range")
        _validate_float_range(self.am_carrier_range_hz, "am_carrier_range_hz")
        _validate_float_range(self.am_mod_range_hz, "am_mod_range_hz")
        _validate_float_range(self.am_mod_index_range, "am_mod_index_range")
        _validate_float_range(self.fm_carrier_range_hz, "fm_carrier_range_hz")
        _validate_float_range(self.fm_mod_range_hz, "fm_mod_range_hz")
        _validate_float_range(self.fm_mod_index_range, "fm_mod_index_range")
        _validate_float_range(self.band_noise_low_range_hz, "band_noise_low_range_hz")
        _validate_float_range(self.band_noise_high_range_hz, "band_noise_high_range_hz")
        if self.band_noise_low_range_hz[1] >= self.band_noise_high_range_hz[1]:
            raise ValueError(
                "band_noise_low_range_hz must be below band_noise_high_range_hz."
            )
        _validate_float_range(self.soft_clip_drive_range, "soft_clip_drive_range")


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for on-the-fly data augmentation.

    Args:
        gain_range: Linear gain range applied to the signal.
        polarity_flip_prob: Probability of flipping signal polarity.
        noise_std_range: Standard deviation range for additive noise.
        soft_clip_prob: Probability of applying soft clipping.
        soft_clip_drive_range: Drive range for soft clipping.

    Physical Basis:
        Mild amplitude, polarity, noise, and saturation variations improve
        robustness without introducing new frequency content.
    """

    gain_range: tuple[float, float] = (0.7, 1.0)
    polarity_flip_prob: float = 0.3
    noise_std_range: tuple[float, float] = (0.0, 0.002)
    soft_clip_prob: float = 0.2
    soft_clip_drive_range: tuple[float, float] = (1.2, 2.5)

    def __post_init__(self) -> None:
        _validate_float_range(self.gain_range, "gain_range")
        _validate_unit_interval(self.polarity_flip_prob, "polarity_flip_prob")
        _validate_non_negative_range(self.noise_std_range, "noise_std_range")
        _validate_unit_interval(self.soft_clip_prob, "soft_clip_prob")
        _validate_float_range(self.soft_clip_drive_range, "soft_clip_drive_range")


@dataclass(frozen=True)
class HBTargetConfig:
    """Configuration for HB_target normalization.

    Args:
        suppression_floor: Minimum gain for mirror suppression.
        energy_cap: Mean energy cap for the high band.
        envelope_min: Minimum envelope gain at Nyquist.

    Physical Basis:
        Fixed suppression and energy caps enforce safe high-band levels
        while retaining time-domain coherence.
    """

    suppression_floor: float = 0.2
    energy_cap: float = 1e-3
    envelope_min: float = 0.2

    def __post_init__(self) -> None:
        _validate_unit_interval(self.suppression_floor, "suppression_floor")
        _validate_positive_float(self.energy_cap, "energy_cap")
        _validate_unit_interval(self.envelope_min, "envelope_min")


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for in-memory caching.

    Args:
        enabled: Whether caching is enabled.
        max_items: Maximum number of cached items.

    Physical Basis:
        Cache reuse accelerates repeated sampling without changing
        statistical properties of the generated data.
    """

    enabled: bool = True
    max_items: int = DEFAULT_CACHE_ITEMS

    def __post_init__(self) -> None:
        if self.max_items <= 0:
            raise ValueError("max_items must be positive.")


@dataclass(frozen=True)
class DataPipelineConfig:
    """Configuration for the full data pipeline.

    Args:
        num_samples: Number of dataset items.
        source_sample_rate: Source sample rate (e.g., 44.1kHz).
        target_sample_rate: Target sample rate (e.g., 88.2kHz).
        source_duration_sec: Duration of source signals in seconds.
        chunk_duration_sec: Duration of chunks in seconds.
        random_chunk: Whether to sample random chunk positions.
        seed: Optional seed for reproducibility.
        signal_sampling: Sampling configuration for source signals.
        augmentation: On-the-fly augmentation configuration.
        degradation: Degradation configuration for SRC diversity.
        band_split: Band-split configuration.
        mirror_detection: Mirror detection configuration.
        hb_target: HB_target normalization configuration.
        cache: Cache configuration.

    Physical Basis:
        The pipeline couples synthetic source generation, degradation SRC,
        band-splitting, and mirror suppression targets to train NMSE.
    """

    num_samples: int = DEFAULT_NUM_SAMPLES
    source_sample_rate: int = DEFAULT_SOURCE_SR
    target_sample_rate: int = DEFAULT_TARGET_SR
    source_duration_sec: float = DEFAULT_DURATION_SEC
    chunk_duration_sec: float = DEFAULT_CHUNK_SEC
    random_chunk: bool = True
    seed: int | None = DEFAULT_SEED
    signal_sampling: SignalSamplingConfig = field(default_factory=SignalSamplingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    band_split: BandSplitConfig = field(default_factory=BandSplitConfig)
    mirror_detection: MirrorDetectionConfig = field(
        default_factory=MirrorDetectionConfig
    )
    hb_target: HBTargetConfig = field(default_factory=HBTargetConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    def __post_init__(self) -> None:
        _validate_positive_int(self.num_samples, "num_samples")
        _validate_positive_int(self.source_sample_rate, "source_sample_rate")
        _validate_positive_int(self.target_sample_rate, "target_sample_rate")
        _validate_positive_float(self.source_duration_sec, "source_duration_sec")
        _validate_positive_float(self.chunk_duration_sec, "chunk_duration_sec")
        if self.chunk_duration_sec > self.source_duration_sec:
            raise ValueError("chunk_duration_sec must not exceed source_duration_sec.")
        ratio = self.target_sample_rate / self.source_sample_rate
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError("target_sample_rate must be integer multiple of source.")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be an int or None.")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a nested dictionary.

        Returns:
            Configuration dictionary suitable for JSON/YAML serialization.

        Physical Basis:
            Serializable configs enable reproducible data generation.
        """
        return {
            "num_samples": self.num_samples,
            "source_sample_rate": self.source_sample_rate,
            "target_sample_rate": self.target_sample_rate,
            "source_duration_sec": self.source_duration_sec,
            "chunk_duration_sec": self.chunk_duration_sec,
            "random_chunk": self.random_chunk,
            "seed": self.seed,
            "signal_sampling": _dataclass_to_dict(self.signal_sampling),
            "augmentation": _dataclass_to_dict(self.augmentation),
            "degradation": _dataclass_to_dict(self.degradation),
            "band_split": _dataclass_to_dict(self.band_split),
            "mirror_detection": _dataclass_to_dict(self.mirror_detection),
            "hb_target": _dataclass_to_dict(self.hb_target),
            "cache": _dataclass_to_dict(self.cache),
        }

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> DataPipelineConfig:
        """Construct configuration from a dictionary.

        Args:
            raw: Mapping of configuration values.

        Returns:
            Parsed DataPipelineConfig instance.

        Raises:
            ValueError: If the input mapping is invalid.

        Physical Basis:
            Explicit parsing ensures configuration correctness before
            generating training data.
        """
        if not isinstance(raw, Mapping):
            raise ValueError("Config data must be a mapping.")

        signal_sampling = _build_dataclass(
            SignalSamplingConfig, raw.get("signal_sampling")
        )
        augmentation = _build_dataclass(AugmentationConfig, raw.get("augmentation"))
        degradation = _build_dataclass(DegradationConfig, raw.get("degradation"))
        band_split = _build_dataclass(BandSplitConfig, raw.get("band_split"))
        mirror_detection = _build_dataclass(
            MirrorDetectionConfig, raw.get("mirror_detection")
        )
        hb_target = _build_dataclass(HBTargetConfig, raw.get("hb_target"))
        cache = _build_dataclass(CacheConfig, raw.get("cache"))

        return DataPipelineConfig(
            num_samples=_coerce_int(
                raw.get("num_samples", DEFAULT_NUM_SAMPLES), "num_samples"
            ),
            source_sample_rate=_coerce_int(
                raw.get("source_sample_rate", DEFAULT_SOURCE_SR),
                "source_sample_rate",
            ),
            target_sample_rate=_coerce_int(
                raw.get("target_sample_rate", DEFAULT_TARGET_SR),
                "target_sample_rate",
            ),
            source_duration_sec=_coerce_float(
                raw.get("source_duration_sec", DEFAULT_DURATION_SEC),
                "source_duration_sec",
            ),
            chunk_duration_sec=_coerce_float(
                raw.get("chunk_duration_sec", DEFAULT_CHUNK_SEC),
                "chunk_duration_sec",
            ),
            random_chunk=bool(raw.get("random_chunk", True)),
            seed=_coerce_optional_int(raw.get("seed", DEFAULT_SEED), "seed"),
            signal_sampling=signal_sampling,
            augmentation=augmentation,
            degradation=degradation,
            band_split=band_split,
            mirror_detection=mirror_detection,
            hb_target=hb_target,
            cache=cache,
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
            else int(np.random.SeedSequence().entropy)
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

        sample = {
            "source": _to_tensor(source_chunk),
            "x_full": _to_tensor(x_full),
            "low_band": _to_tensor(low_band),
            "high_band": _to_tensor(high_band),
            "hb_target": _to_tensor(hb_target_result.target),
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


def load_data_config(path: Path) -> DataPipelineConfig:
    """Load a DataPipelineConfig from JSON or YAML.

    Args:
        path: Path to JSON/YAML configuration file.

    Returns:
        Parsed DataPipelineConfig.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
        RuntimeError: If parsing fails.

    Physical Basis:
        Persisted configs ensure reproducible dataset regeneration.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            import yaml  # type: ignore

            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        elif suffix == ".json":
            import json

            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            raise ValueError("Config file must be .json, .yaml, or .yml")
    except Exception as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(f"Failed to load config: {exc}") from exc

    return DataPipelineConfig.from_dict(data or {})


def save_data_config(config: DataPipelineConfig, path: Path) -> None:
    """Save a DataPipelineConfig to JSON or YAML.

    Args:
        config: Configuration to save.
        path: Destination file path.

    Raises:
        ValueError: If the file extension is unsupported.
        RuntimeError: If writing fails.

    Physical Basis:
        Persisting configurations enables repeatable data generation runs.
    """
    suffix = path.suffix.lower()
    data = config.to_dict()
    try:
        if suffix in {".yaml", ".yml"}:
            import yaml  # type: ignore

            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle, sort_keys=False)
        elif suffix == ".json":
            import json

            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        else:
            raise ValueError("Config file must be .json, .yaml, or .yml")
    except Exception as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(f"Failed to save config: {exc}") from exc


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


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


def _build_dataclass(cls: type[Any], raw: Any) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config section for {cls.__name__} must be mapping.")

    kwargs: dict[str, Any] = {}
    for dataclass_field in fields(cls):
        if dataclass_field.name in raw:
            value = raw[dataclass_field.name]
            if isinstance(value, list):
                value = tuple(value)
            kwargs[dataclass_field.name] = value
    return cls(**kwargs)


def _coerce_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating, str)):
        return int(value)
    raise ValueError(f"{name} must be an int.")


def _coerce_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, name)


def _coerce_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a float.")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"{name} must be a float.")


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    _validate_signal(array)
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def _validate_pipeline_config(config: DataPipelineConfig) -> None:
    if not isinstance(config, DataPipelineConfig):
        raise ValueError("config must be a DataPipelineConfig")


def _validate_signal(signal: np.ndarray) -> None:
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got {signal.ndim}D")
    if signal.size == 0:
        raise ValueError("signal must be non-empty")


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_positive_float(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _validate_unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")


def _validate_float_range(value_range: tuple[float, float], name: str) -> None:
    if value_range[0] <= 0.0 or value_range[1] <= 0.0:
        raise ValueError(f"{name} must be positive.")
    if value_range[0] > value_range[1]:
        raise ValueError(f"{name} must have min <= max.")


def _validate_non_negative_range(value_range: tuple[float, float], name: str) -> None:
    if value_range[0] < 0.0 or value_range[1] < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    if value_range[0] > value_range[1]:
        raise ValueError(f"{name} must have min <= max.")


def _validate_int_range(value_range: tuple[int, int], name: str) -> None:
    if value_range[0] <= 0 or value_range[1] <= 0:
        raise ValueError(f"{name} must be positive.")
    if value_range[0] > value_range[1]:
        raise ValueError(f"{name} must have min <= max.")


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
