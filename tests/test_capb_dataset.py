"""Tests for the CAPB alias-free dataset."""

from pathlib import Path

import numpy as np
import pytest
import torch

from totton_audio_de_mirroring.data.capb_dataset import (
    TARGET_SAMPLE_RATE,
    UPSAMPLE_RATIO,
    BrickwallConfig,
    CAPBDataConfig,
    CAPBUpsampleDataset,
    compute_edge_mask,
    compute_flat_mask,
    compute_quiet_mask,
    load_capb_data_config,
)


@pytest.fixture(scope="module")
def dataset() -> CAPBUpsampleDataset:
    config = CAPBDataConfig(num_samples=8, seed=42)
    return CAPBUpsampleDataset(config)


def test_input_is_exact_decimation_of_target(dataset) -> None:
    """Core consistency: x == target[::2] exactly (alias-free by design)."""
    for index in range(4):
        sample = dataset[index]
        source = sample["source"].numpy()
        target = sample["target"].numpy()
        np.testing.assert_array_equal(source, target[::UPSAMPLE_RATIO])


def test_sample_shapes(dataset) -> None:
    sample = dataset[0]
    chunk_len = int(0.25 * TARGET_SAMPLE_RATE)
    assert sample["target"].shape == (chunk_len,)
    assert sample["source"].shape == (chunk_len // UPSAMPLE_RATIO,)
    assert sample["flat_mask"].shape == (chunk_len,)
    assert sample["quiet_mask"].shape == (chunk_len,)


def test_target_is_band_limited(dataset) -> None:
    """Target must have no content above the input Nyquist."""
    for index in range(4):
        target = dataset[index]["target"].numpy().astype(np.float64)
        spectrum = np.abs(np.fft.rfft(target * np.hanning(target.size)))
        freqs = np.fft.rfftfreq(target.size, d=1.0 / TARGET_SAMPLE_RATE)
        image = spectrum[freqs >= 22_500.0]
        main = spectrum[freqs <= 20_000.0]
        ratio_db = 20.0 * np.log10((np.max(image) + 1e-300) / (np.max(main) + 1e-300))
        assert ratio_db <= -80.0, dataset[index]["signal_type"]


def test_deterministic_by_index(dataset) -> None:
    first = dataset[3]
    second = dataset[3]
    torch.testing.assert_close(first["target"], second["target"])
    assert first["signal_type"] == second["signal_type"]


def test_flat_mask_marks_square_plateaus() -> None:
    """Plateaus of a clean 100 Hz square are flat except near edges."""
    from scipy import signal as sp_signal

    from totton_audio_de_mirroring.data.capb_dataset import compute_flat_mask

    time_axis = np.arange(TARGET_SAMPLE_RATE // 4) / TARGET_SAMPLE_RATE
    square = 0.5 * sp_signal.square(2.0 * np.pi * 100.0 * time_axis)
    mask = compute_flat_mask(square)
    assert float(np.mean(mask)) > 0.7


def test_flat_mask_ignores_tones() -> None:
    from totton_audio_de_mirroring.data.capb_dataset import compute_flat_mask

    time_axis = np.arange(TARGET_SAMPLE_RATE // 4) / TARGET_SAMPLE_RATE
    tone = 0.5 * np.sin(2.0 * np.pi * 1_000.0 * time_axis)
    assert float(np.mean(compute_flat_mask(tone))) < 0.01


def test_quiet_mask_marks_click_silence() -> None:
    from totton_audio_de_mirroring.data.capb_dataset import compute_quiet_mask

    signal = np.zeros(TARGET_SAMPLE_RATE // 4)
    signal[signal.size // 2] = 0.9
    mask = compute_quiet_mask(signal)
    assert float(np.mean(mask)) > 0.95
    assert mask[signal.size // 2] == 0.0


def test_dataset_masks_present_for_edge_family() -> None:
    config = CAPBDataConfig(num_samples=2, seed=7, signal_mix={"isolated_click": 1.0})
    dataset = CAPBUpsampleDataset(config)
    sample = dataset[0]
    assert float(sample["quiet_mask"].mean()) > 0.5


def test_stationary_noise_does_not_receive_slope_edge_mask() -> None:
    rng = np.random.default_rng(9)
    noise = rng.standard_normal(TARGET_SAMPLE_RATE // 20)
    flat = compute_flat_mask(noise)
    quiet = compute_quiet_mask(noise)
    mask = compute_edge_mask(flat, quiet)
    assert float(mask.mean()) < 0.01


def test_out_of_range_index_raises(dataset) -> None:
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_brickwall_config_validation() -> None:
    """The Nyquist bound is rate-dependent, so it is checked on the config."""
    with pytest.raises(ValueError, match="stopband_edge_hz"):
        CAPBDataConfig(
            brickwall=BrickwallConfig(
                passband_edge_hz=22_000.0, stopband_edge_hz=23_000.0
            )
        )
    with pytest.raises(ValueError, match="passband_edge_hz"):
        BrickwallConfig(passband_edge_hz=23_000.0, stopband_edge_hz=22_000.0)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="signal_mix"):
        CAPBDataConfig(signal_mix={})
    with pytest.raises(ValueError, match="chunk_duration_sec"):
        CAPBDataConfig(chunk_duration_sec=2.0, source_duration_sec=1.0)


def test_load_yaml_config() -> None:
    config = load_capb_data_config(Path("configs/data_generation_capb.yaml"))
    assert config.num_samples == 10_000
    assert config.brickwall.stopband_edge_hz == 22_050.0
    assert abs(sum(config.signal_mix.values()) - 1.0) < 1e-6
    assert config.source_sample_rate == 44_100
    assert config.target_sample_rate == 88_200
    assert config.near_nyquist_high_range_hz == (20_000.0, 21_500.0)
    assert config.signal_mix["isolated_click"] == pytest.approx(0.05)


def test_load_yaml_config_48k() -> None:
    config = load_capb_data_config(Path("configs/data_generation_capb_48k.yaml"))
    assert config.source_sample_rate == 48_000
    assert config.target_sample_rate == 96_000
    assert config.brickwall.passband_edge_hz == 23_700.0
    assert config.brickwall.stopband_edge_hz == 24_000.0
    assert config.near_nyquist_high_range_hz == (20_000.0, 23_400.0)
    assert abs(sum(config.signal_mix.values()) - 1.0) < 1e-6


def test_config_rejects_inconsistent_rates() -> None:
    with pytest.raises(ValueError, match="target_sample_rate"):
        CAPBDataConfig(source_sample_rate=48_000, target_sample_rate=88_200)


def test_config_rejects_bad_near_nyquist_range() -> None:
    with pytest.raises(ValueError, match="near_nyquist_high_range_hz"):
        CAPBDataConfig(near_nyquist_high_range_hz=(20_000.0, 30_000.0))


@pytest.fixture(scope="module")
def dataset_48k() -> CAPBUpsampleDataset:
    config = CAPBDataConfig(
        num_samples=4,
        seed=42,
        source_sample_rate=48_000,
        target_sample_rate=96_000,
        brickwall=BrickwallConfig(passband_edge_hz=23_700.0, stopband_edge_hz=24_000.0),
        near_nyquist_high_range_hz=(20_000.0, 23_400.0),
    )
    return CAPBUpsampleDataset(config)


def test_48k_input_is_exact_decimation_of_target(dataset_48k) -> None:
    for index in range(2):
        sample = dataset_48k[index]
        source = sample["source"].numpy()
        target = sample["target"].numpy()
        np.testing.assert_array_equal(source, target[::UPSAMPLE_RATIO])


def test_48k_sample_shapes(dataset_48k) -> None:
    sample = dataset_48k[0]
    chunk_len = int(0.25 * 96_000)
    assert sample["target"].shape == (chunk_len,)
    assert sample["source"].shape == (chunk_len // UPSAMPLE_RATIO,)


def test_48k_target_is_band_limited_below_24k(dataset_48k) -> None:
    """96 kHz teacher must have no content above the 24 kHz input Nyquist."""
    for index in range(2):
        target = dataset_48k[index]["target"].numpy().astype(np.float64)
        spectrum = np.abs(np.fft.rfft(target * np.hanning(target.size)))
        freqs = np.fft.rfftfreq(target.size, d=1.0 / 96_000)
        image = spectrum[freqs >= 24_500.0]
        main = spectrum[freqs <= 20_000.0]
        ratio_db = 20.0 * np.log10((np.max(image) + 1e-300) / (np.max(main) + 1e-300))
        assert ratio_db <= -80.0, dataset_48k[index]["signal_type"]
