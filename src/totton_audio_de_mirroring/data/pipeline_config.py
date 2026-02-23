"""Configuration helpers for the mirror suppression data pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from totton_audio_de_mirroring.data.degradation import DegradationConfig
from totton_audio_de_mirroring.data.generator import list_signal_types
from totton_audio_de_mirroring.data.mirror_detection import MirrorDetectionConfig
from totton_audio_de_mirroring.models.band_split import BandSplitConfig

DEFAULT_SOURCE_SR = 44_100
DEFAULT_TARGET_SR = 88_200
DEFAULT_DURATION_SEC = 1.0
DEFAULT_CHUNK_SEC = 0.25
DEFAULT_NUM_SAMPLES = 10_000
DEFAULT_CACHE_ITEMS = 128
DEFAULT_SEED = None
DEFAULT_INPUT_ROUTE = "source_chunk_44k1_to_x_full_88k2_via_degradation"
DEFAULT_TARGET_ROUTE = "high_band_to_hb_target_via_mirror_detection"
DEFAULT_TEACHER_TYPE = "raw_88k2"
LEGACY_DEFAULT_TEACHER_TYPE = "bessel_88k2"
ALLOWED_TEACHER_TYPES = ("raw_88k2", "bessel_88k2")
TeacherType = Literal["raw_88k2", "bessel_88k2"]


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
class Stage1PathConfig:
    """Explicit Stage 1 data-path specification for input/target generation.

    Args:
        input_route: Route identifier for Stage 1 input (`x_full`) generation.
        target_route: Route identifier for Stage 1 target (`hb_target`) generation.
        strict_route_validation: If true, enforce the fixed 44.1kHz->88.2kHz 2x route.

    Physical Basis:
        Fixing route identifiers in configuration prevents silent drift between
        documented design intent and implemented training data paths.
    """

    input_route: str = DEFAULT_INPUT_ROUTE
    target_route: str = DEFAULT_TARGET_ROUTE
    strict_route_validation: bool = True

    def __post_init__(self) -> None:
        _validate_non_empty(self.input_route, "input_route")
        _validate_non_empty(self.target_route, "target_route")
        if self.input_route != DEFAULT_INPUT_ROUTE:
            raise ValueError(
                "input_route must match the fixed Stage 1 route "
                f"'{DEFAULT_INPUT_ROUTE}'."
            )
        if self.target_route != DEFAULT_TARGET_ROUTE:
            raise ValueError(
                "target_route must match the fixed Stage 1 route "
                f"'{DEFAULT_TARGET_ROUTE}'."
            )


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
        teacher_type: Teacher reference path for Stage 1 target generation.
        stage1_path: Explicit Stage 1 input/target path specification.
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
    teacher_type: TeacherType = cast(TeacherType, DEFAULT_TEACHER_TYPE)
    stage1_path: Stage1PathConfig = field(default_factory=Stage1PathConfig)
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
        if self.stage1_path.strict_route_validation and round(ratio) != 2:
            raise ValueError(
                "strict stage1_path requires a fixed 2x route "
                "(source_sample_rate -> target_sample_rate)."
            )
        if self.teacher_type not in ALLOWED_TEACHER_TYPES:
            raise ValueError(
                "teacher_type must be one of "
                f"{ALLOWED_TEACHER_TYPES}, got {self.teacher_type!r}."
            )
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
            "teacher_type": self.teacher_type,
            "stage1_path": _dataclass_to_dict(self.stage1_path),
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
        stage1_path = _build_dataclass(Stage1PathConfig, raw.get("stage1_path"))
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
            random_chunk=_coerce_bool(raw.get("random_chunk", True), "random_chunk"),
            seed=_coerce_optional_int(raw.get("seed", DEFAULT_SEED), "seed"),
            signal_sampling=signal_sampling,
            augmentation=augmentation,
            degradation=degradation,
            band_split=band_split,
            mirror_detection=mirror_detection,
            hb_target=hb_target,
            teacher_type=_coerce_teacher_type(
                raw.get("teacher_type", LEGACY_DEFAULT_TEACHER_TYPE), "teacher_type"
            ),
            stage1_path=stage1_path,
            cache=cache,
        )


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
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating | str):
        return int(value)
    raise ValueError(f"{name} must be an int.")


def _coerce_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, name)


def _coerce_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a float.")
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"{name} must be a float.")


def _coerce_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, int | np.integer):
        return bool(value)
    raise ValueError(f"{name} must be a bool.")


def _coerce_teacher_type(value: Any, name: str) -> TeacherType:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")

    normalized = value.strip().lower()
    aliases = {
        "raw88": "raw_88k2",
        "raw_88k2": "raw_88k2",
        "native_88k2": "raw_88k2",
        "bessel": "bessel_88k2",
        "bessel_88k2": "bessel_88k2",
    }
    mapped = aliases.get(normalized)
    if mapped is None:
        raise ValueError(
            f"{name} must be one of {ALLOWED_TEACHER_TYPES}, got {value!r}."
        )
    return cast(TeacherType, mapped)


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
