import json
from pathlib import Path

import numpy as np
import pytest
import torch

from totton_audio_de_mirroring.data import dataset as dataset_module
from totton_audio_de_mirroring.data.dataloader import (
    DataLoaderConfig,
    create_dataloader,
)
from totton_audio_de_mirroring.data.pipeline_config import (
    AugmentationConfig,
    DataPipelineConfig,
    SignalSamplingConfig,
    load_data_config,
    save_data_config,
)
from totton_audio_de_mirroring.models.band_split import BandSplitConfig


def test_raw_teacher_builder_keeps_source_teacher_alignment() -> None:
    request = dataset_module.SignalRequest(
        signal_type="multitone",
        params={"frequencies_hz": [500.0, 1500.0, 3200.0]},
    )
    rng = np.random.default_rng(42)
    source_chunk, teacher_chunk, chunk_start = (
        dataset_module._build_raw_teacher_source_chunk_and_reference(
            request=request,
            source_seed=1234,
            source_sr=44_100,
            target_sr=88_200,
            source_duration_sec=0.5,
            chunk_duration_sec=0.25,
            random_chunk=True,
            augmentation=AugmentationConfig(
                gain_range=(1.0, 1.0),
                polarity_flip_prob=0.0,
                noise_std_range=(0.0, 0.0),
                soft_clip_prob=0.0,
                soft_clip_drive_range=(1.0, 1.0),
            ),
            rng=rng,
        )
    )

    expected_source = dataset_module._downsample_raw_reference(
        teacher_chunk,
        source_sr=88_200,
        target_sr=44_100,
    )
    assert source_chunk.shape == (11_025,)
    assert teacher_chunk.shape == (22_050,)
    assert 0 <= chunk_start <= 11_025
    assert np.allclose(source_chunk, expected_source)


def _small_config() -> DataPipelineConfig:
    return DataPipelineConfig(
        num_samples=4,
        source_duration_sec=0.5,
        chunk_duration_sec=0.25,
        seed=123,
        signal_sampling=SignalSamplingConfig(signal_types=("multitone",)),
        augmentation=AugmentationConfig(
            gain_range=(1.0, 1.0),
            polarity_flip_prob=0.0,
            noise_std_range=(0.0, 0.0),
            soft_clip_prob=0.0,
            soft_clip_drive_range=(1.0, 1.0),
        ),
        band_split=BandSplitConfig(num_taps=513, sample_rate=88_200),
    )


def test_dataset_item_shapes() -> None:
    config = _small_config()
    dataset = create_dataloader(config, DataLoaderConfig(batch_size=1)).dataset

    sample = dataset[0]

    length = int(round(config.chunk_duration_sec * config.target_sample_rate))
    assert sample["x_full"].shape == (length,)
    assert sample["low_band"].shape == (length,)
    assert sample["high_band"].shape == (length,)
    assert sample["hb_target"].shape == (length,)
    assert sample["mirror_mask"].ndim == 2
    assert sample["source"].shape == (
        int(round(config.chunk_duration_sec * config.source_sample_rate)),
    )
    assert isinstance(sample["profile"].method, str)
    assert sample["signal_type"] == "multitone"
    assert sample["teacher_type"] == config.teacher_type
    assert sample["input_route"] == config.stage1_path.input_route
    assert sample["target_route"] == config.stage1_path.target_route
    assert sample["x_full"].dtype == torch.float32


def test_dataset_reproducible() -> None:
    config = DataPipelineConfig(
        num_samples=2,
        source_duration_sec=0.5,
        chunk_duration_sec=0.25,
        seed=999,
        signal_sampling=SignalSamplingConfig(signal_types=("multitone",)),
        augmentation=AugmentationConfig(
            gain_range=(1.0, 1.0),
            polarity_flip_prob=0.0,
            noise_std_range=(0.0, 0.0),
            soft_clip_prob=0.0,
            soft_clip_drive_range=(1.0, 1.0),
        ),
        band_split=BandSplitConfig(num_taps=513, sample_rate=88_200),
    )
    dataset_a = create_dataloader(config, DataLoaderConfig(batch_size=1)).dataset
    dataset_b = create_dataloader(config, DataLoaderConfig(batch_size=1)).dataset

    sample_a = dataset_a[0]
    sample_b = dataset_b[0]

    assert torch.allclose(sample_a["x_full"], sample_b["x_full"])
    assert torch.allclose(sample_a["hb_target"], sample_b["hb_target"])


def test_dataloader_batches() -> None:
    config = _small_config()
    loader = create_dataloader(
        config,
        DataLoaderConfig(batch_size=2, shuffle=False, num_workers=0, drop_last=False),
    )

    batch = next(iter(loader))
    assert batch["x_full"].shape[0] == 2
    assert batch["hb_target"].shape[0] == 2
    assert batch["mirror_mask"].shape[0] == 2
    assert batch["chunk_start"].shape == (2,)
    assert batch["teacher_type"] == [config.teacher_type, config.teacher_type]


def test_config_roundtrip_json_yaml(tmp_path: Path) -> None:
    config = _small_config()

    json_path = tmp_path / "config.json"
    yaml_path = tmp_path / "config.yaml"

    save_data_config(config, json_path)
    save_data_config(config, yaml_path)

    config_json = load_data_config(json_path)
    config_yaml = load_data_config(yaml_path)

    assert config_json.chunk_duration_sec == config.chunk_duration_sec
    assert config_yaml.chunk_duration_sec == config.chunk_duration_sec

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw["num_samples"] == config.num_samples


def test_signal_sampling_config_validation() -> None:
    config = SignalSamplingConfig(signal_types=("white_noise",))
    rng = np.random.default_rng(0)
    request = config.signal_types[rng.integers(0, len(config.signal_types))]
    assert request == "white_noise"


def test_config_bool_coercion() -> None:
    config = DataPipelineConfig.from_dict({"random_chunk": "false"})
    assert config.random_chunk is False


def test_config_teacher_type_legacy_default_and_alias() -> None:
    legacy = DataPipelineConfig.from_dict({})
    assert legacy.teacher_type == "bessel_88k2"

    aliased = DataPipelineConfig.from_dict({"teacher_type": "raw88"})
    assert aliased.teacher_type == "raw_88k2"

    native_aliased = DataPipelineConfig.from_dict({"teacher_type": "native_88k2"})
    assert native_aliased.teacher_type == "raw_88k2"


def test_stage1_path_roundtrip_in_serialized_config(tmp_path: Path) -> None:
    config = _small_config()
    yaml_path = tmp_path / "config.yaml"
    save_data_config(config, yaml_path)
    loaded = load_data_config(yaml_path)
    assert loaded.stage1_path.input_route == config.stage1_path.input_route
    assert loaded.stage1_path.target_route == config.stage1_path.target_route
    assert loaded.stage1_path.strict_route_validation is True


def test_stage1_strict_path_requires_2x_ratio() -> None:
    with pytest.raises(
        ValueError, match="strict stage1_path requires a fixed 2x route"
    ):
        DataPipelineConfig(source_sample_rate=44_100, target_sample_rate=176_400)


def test_dataset_pipeline_route_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _small_config()
    calls: dict[str, dict[str, object]] = {}

    original_apply = dataset_module.apply_degradation_profile
    original_detect = dataset_module.detect_mirror_artifacts
    original_project = dataset_module.project_teacher_hb_target

    def wrapped_apply(
        signal: np.ndarray,
        source_sr: int,
        target_sr: int,
        profile: object,
        rng: np.random.Generator,
    ) -> np.ndarray:
        calls["apply_degradation_profile"] = {
            "source_sr": source_sr,
            "target_sr": target_sr,
            "signal_shape": signal.shape,
        }
        return original_apply(signal, source_sr, target_sr, profile, rng)

    def wrapped_detect(
        hb_signal: np.ndarray,
        sample_rate: int,
        *,
        config: object = None,
    ) -> object:
        calls["detect_mirror_artifacts"] = dict(
            calls.get("detect_mirror_artifacts", {})
        )
        calls["detect_mirror_artifacts"]["sample_rate"] = sample_rate
        calls["detect_mirror_artifacts"]["hb_shape"] = hb_signal.shape
        calls["detect_mirror_artifacts"]["config"] = config
        return original_detect(hb_signal, sample_rate, config=config)

    def wrapped_project(
        hb_in: np.ndarray,
        teacher_hb: np.ndarray,
        sample_rate: int,
        *,
        detection_config: object = None,
        suppression_floor: float = 0.0,
        energy_cap: float = 0.0,
        envelope_min: float = 0.0,
    ) -> np.ndarray:
        calls["project_teacher_hb_target"] = {
            "sample_rate": sample_rate,
            "hb_in_shape": hb_in.shape,
            "teacher_shape": teacher_hb.shape,
            "suppression_floor": suppression_floor,
            "energy_cap": energy_cap,
            "envelope_min": envelope_min,
            "detection_config": detection_config,
        }
        return original_project(
            hb_in,
            teacher_hb,
            sample_rate,
            detection_config=detection_config,
            suppression_floor=suppression_floor,
            energy_cap=energy_cap,
            envelope_min=envelope_min,
        )

    monkeypatch.setattr(dataset_module, "apply_degradation_profile", wrapped_apply)
    monkeypatch.setattr(dataset_module, "detect_mirror_artifacts", wrapped_detect)
    monkeypatch.setattr(dataset_module, "project_teacher_hb_target", wrapped_project)

    dataset = create_dataloader(config, DataLoaderConfig(batch_size=1)).dataset
    sample = dataset[0]

    apply_call = calls["apply_degradation_profile"]
    assert apply_call["source_sr"] == config.source_sample_rate
    assert apply_call["target_sr"] == config.target_sample_rate
    assert apply_call["signal_shape"] == (
        int(round(config.chunk_duration_sec * config.source_sample_rate)),
    )

    detect_call = calls["detect_mirror_artifacts"]
    assert detect_call["sample_rate"] == config.target_sample_rate
    assert detect_call["hb_shape"] == sample["high_band"].shape

    project_call = calls["project_teacher_hb_target"]
    assert project_call["sample_rate"] == config.target_sample_rate
    assert project_call["hb_in_shape"] == sample["high_band"].shape
    assert project_call["teacher_shape"] == sample["high_band"].shape
    assert project_call["suppression_floor"] == config.hb_target.suppression_floor
    assert project_call["energy_cap"] == config.hb_target.energy_cap
    assert project_call["envelope_min"] == config.hb_target.envelope_min


def test_dataset_mirror_mask_derived_from_input_high_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mirror_mask should be derived from input high_band, not teacher high-band."""
    config = _small_config()

    source_sr = config.source_sample_rate
    chunk_len = int(round(config.chunk_duration_sec * source_sr))
    t = np.arange(chunk_len, dtype=np.float64) / float(source_sr)
    source_chunk = 0.5 * np.sin(2.0 * np.pi * 21_000.0 * t)
    teacher_full = np.zeros(
        int(round(config.chunk_duration_sec * config.target_sample_rate))
    )

    def fake_raw_builder(**_: object) -> tuple[np.ndarray, np.ndarray, int]:
        return source_chunk.astype(np.float32), teacher_full.astype(np.float32), 0

    monkeypatch.setattr(
        dataset_module,
        "_build_raw_teacher_source_chunk_and_reference",
        fake_raw_builder,
    )

    original_detect = dataset_module.detect_mirror_artifacts

    def wrapped_detect(
        hb_signal: np.ndarray,
        sample_rate: int,
        *,
        config: object = None,
    ) -> object:
        energy = float(np.mean(np.square(hb_signal.astype(np.float64))))
        assert energy > 1.0e-9
        return original_detect(hb_signal, sample_rate, config=config)

    monkeypatch.setattr(dataset_module, "detect_mirror_artifacts", wrapped_detect)

    dataset = create_dataloader(config, DataLoaderConfig(batch_size=1)).dataset
    _ = dataset[0]


def test_dataset_hb_target_respects_energy_cap() -> None:
    """HB target energy should stay under configured cap."""
    config = _small_config()
    dataset = create_dataloader(config, DataLoaderConfig(batch_size=1)).dataset
    sample = dataset[0]
    hb_target = sample["hb_target"].detach().cpu().numpy().astype(np.float64)
    hb_energy = float(np.mean(np.square(hb_target)))
    assert hb_energy <= config.hb_target.energy_cap + 1.0e-9


def test_dataset_teacher_type_switches_reference_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_calls = {"count": 0}
    bessel_calls = {"count": 0}

    def fake_raw_builder(**_: object) -> tuple[np.ndarray, np.ndarray, int]:
        raw_calls["count"] += 1
        source = np.zeros(11_025, dtype=np.float64)
        teacher = np.zeros(22_050, dtype=np.float64)
        return source, teacher, 0

    def fake_teacher_reference(
        signal: np.ndarray,
        *,
        source_sr: int,
        target_sr: int,
        teacher_type: str,
        bessel_cutoff_hz: float,
        bessel_order: int,
    ) -> np.ndarray:
        del signal, source_sr, target_sr, teacher_type, bessel_cutoff_hz, bessel_order
        bessel_calls["count"] += 1
        return np.zeros(22_050, dtype=np.float64)

    monkeypatch.setattr(
        dataset_module,
        "_build_raw_teacher_source_chunk_and_reference",
        fake_raw_builder,
    )
    monkeypatch.setattr(
        dataset_module, "_build_teacher_reference", fake_teacher_reference
    )

    raw_config = _small_config()
    raw_dataset = create_dataloader(raw_config, DataLoaderConfig(batch_size=1)).dataset
    _ = raw_dataset[0]
    assert raw_calls["count"] == 1
    assert bessel_calls["count"] == 0

    raw_payload = raw_config.to_dict()
    raw_payload["teacher_type"] = "bessel_88k2"
    bessel_config = DataPipelineConfig.from_dict(raw_payload)
    bessel_dataset = create_dataloader(
        bessel_config, DataLoaderConfig(batch_size=1)
    ).dataset
    _ = bessel_dataset[0]
    assert raw_calls["count"] == 1
    assert bessel_calls["count"] == 1


def test_dataset_rejects_non_finite_teacher_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _small_config().to_dict()
    payload["teacher_type"] = "bessel_88k2"
    config = DataPipelineConfig.from_dict(payload)

    def fake_teacher_reference(
        signal: np.ndarray,
        *,
        source_sr: int,
        target_sr: int,
        teacher_type: str,
        bessel_cutoff_hz: float,
        bessel_order: int,
    ) -> np.ndarray:
        del source_sr, target_sr, teacher_type, bessel_cutoff_hz, bessel_order
        return np.full(signal.shape[0] * 2, np.nan, dtype=np.float64)

    monkeypatch.setattr(
        dataset_module, "_build_teacher_reference", fake_teacher_reference
    )

    dataset = create_dataloader(config, DataLoaderConfig(batch_size=1)).dataset
    with pytest.raises(ValueError, match="teacher_full contains non-finite values"):
        _ = dataset[0]


def test_raw_teacher_generates_at_target_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    sample_rates: list[int] = []
    original_generate_signal = dataset_module.generate_signal
    original_downsample = dataset_module._downsample_raw_reference

    def wrapped_generate_signal(
        signal_type: str,
        sample_rate: int = 44_100,
        duration_sec: float = 1.0,
        seed: int | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        sample_rates.append(sample_rate)
        return original_generate_signal(
            signal_type,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            seed=seed,
            **kwargs,
        )

    downsample_calls = {"count": 0}

    def wrapped_downsample(
        signal: np.ndarray, *, source_sr: int, target_sr: int
    ) -> np.ndarray:
        downsample_calls["count"] += 1
        return original_downsample(signal, source_sr=source_sr, target_sr=target_sr)

    monkeypatch.setattr(dataset_module, "generate_signal", wrapped_generate_signal)
    monkeypatch.setattr(dataset_module, "_downsample_raw_reference", wrapped_downsample)

    dataset = create_dataloader(config, DataLoaderConfig(batch_size=1)).dataset
    _ = dataset[0]

    assert sample_rates == [config.target_sample_rate]
    assert downsample_calls["count"] >= 2
