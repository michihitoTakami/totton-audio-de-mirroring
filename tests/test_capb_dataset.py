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
    TransientSupervisionConfig,
    compute_edge_mask,
    compute_envelope_edge_mask,
    compute_flat_mask,
    compute_post_echo_mask,
    compute_pre_echo_mask,
    compute_quiet_mask,
    load_capb_data_config,
)
from totton_audio_de_mirroring.data.transient_supervision import cardinal_upsample


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


def test_cardinal_upsample_preserves_impulse_lattice_and_bandlimit() -> None:
    source = np.zeros(128, dtype=np.float64)
    source[63] = 1.0
    target = cardinal_upsample(source, 2)
    spectrum = np.abs(np.fft.rfft(target))

    assert np.allclose(target[::2], source, atol=1.0e-12)
    assert np.max(spectrum[65:]) < 1.0e-12


def test_sample_shapes(dataset) -> None:
    sample = dataset[0]
    chunk_len = int(0.25 * TARGET_SAMPLE_RATE)
    assert sample["target"].shape == (chunk_len,)
    assert sample["source"].shape == (chunk_len // UPSAMPLE_RATIO,)
    assert sample["flat_mask"].shape == (chunk_len,)
    assert sample["quiet_mask"].shape == (chunk_len,)
    assert sample["post_echo_mask"].shape == (chunk_len,)
    assert sample["safe_active_mask"].shape == (chunk_len,)
    assert sample["stationary"].shape == ()


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


def test_deterministic_seed_is_independent_of_worker_assignment(
    dataset: CAPBUpsampleDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Worker:
        id = 7

    baseline = dataset._rng_for_index(3).standard_normal(8)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: Worker())
    worker_values = dataset._rng_for_index(3).standard_normal(8)

    np.testing.assert_array_equal(baseline, worker_values)


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


def test_focused_edge_family_uses_dedicated_transient_masks() -> None:
    config = CAPBDataConfig(
        num_samples=2,
        seed=7,
        signal_mix={"isolated_click": 1.0},
        transient_supervision=TransientSupervisionConfig(enabled=True),
    )
    dataset = CAPBUpsampleDataset(config)
    sample = dataset[0]
    assert float(sample["flat_mask"].sum()) == 0.0
    assert float(sample["quiet_mask"].sum()) == 0.0
    assert float(sample["edge_mask"].sum()) > 0.0
    assert float(sample["pre_echo_mask"].sum()) > 0.0
    assert float(sample["post_echo_mask"].sum()) > 0.0
    assert not bool(sample["stationary"])


@pytest.fixture(scope="module")
def focused_click_dataset() -> CAPBUpsampleDataset:
    config = CAPBDataConfig(
        num_samples=40,
        seed=17,
        signal_mix={"isolated_click": 1.0},
        transient_supervision=TransientSupervisionConfig(enabled=True),
    )
    return CAPBUpsampleDataset(config)


def test_focused_click_always_has_gate_aligned_mask(focused_click_dataset) -> None:
    expected = round(3.5 * TARGET_SAMPLE_RATE / 1_000.0)
    for index in range(len(focused_click_dataset)):
        sample = focused_click_dataset[index]
        assert bool(sample["focused_event"])
        assert int(torch.count_nonzero(sample["pre_echo_mask"])) == expected
        assert int(torch.count_nonzero(sample["post_echo_mask"])) == expected
        torch.testing.assert_close(sample["source"], sample["target"][::2])


def test_focused_click_can_supervise_far_pre_echo_tail() -> None:
    transient = TransientSupervisionConfig(
        enabled=True,
        context_ms=12.0,
        far_pre_echo_guard_ms=4.0,
        far_pre_echo_window_ms=8.0,
    )
    config = CAPBDataConfig(
        num_samples=1,
        seed=19,
        signal_mix={"isolated_click": 1.0},
        transient_supervision=transient,
    )

    sample = CAPBUpsampleDataset(config)[0]

    assert int(torch.count_nonzero(sample["far_pre_echo_mask"])) == round(
        8.0 * TARGET_SAMPLE_RATE / 1_000.0
    )
    assert (
        int(torch.count_nonzero(sample["pre_echo_mask"] * sample["far_pre_echo_mask"]))
        == 0
    )


def test_far_pre_echo_window_requires_complete_context() -> None:
    with pytest.raises(ValueError, match="context_ms"):
        TransientSupervisionConfig(
            enabled=True,
            context_ms=5.0,
            far_pre_echo_guard_ms=4.0,
            far_pre_echo_window_ms=8.0,
        )


def test_transient_clean_and_augmented_views_are_both_present(
    focused_click_dataset,
) -> None:
    clean = [
        focused_click_dataset[index]
        for index in range(len(focused_click_dataset))
        if bool(focused_click_dataset[index]["transient_clean"])
    ]
    augmented = [
        focused_click_dataset[index]
        for index in range(len(focused_click_dataset))
        if not bool(focused_click_dataset[index]["transient_clean"])
    ]
    assert clean
    assert augmented
    assert all(float(sample["quiet_mask"].sum()) == 0.0 for sample in clean)
    assert all(float(sample["quiet_mask"].sum()) == 0.0 for sample in augmented)


def test_pre_echo_mask_matches_gate_window() -> None:
    mask = compute_pre_echo_mask(2_000, event_start=1_000, sample_rate=100_000)
    assert np.all(mask[600:950] == 1.0)
    assert float(mask[:600].sum() + mask[950:].sum()) == 0.0


def test_post_echo_mask_matches_gate_window() -> None:
    mask = compute_post_echo_mask(2_000, event_stop=1_000, sample_rate=100_000)
    assert np.all(mask[1_050:1_400] == 1.0)
    assert float(mask[:1_050].sum() + mask[1_400:].sum()) == 0.0


def test_safe_active_mask_excludes_transient_risk_windows(
    focused_click_dataset,
) -> None:
    sample = focused_click_dataset[0]
    risk = torch.maximum(sample["edge_mask"], sample["pre_echo_mask"])
    risk = torch.maximum(risk, sample["post_echo_mask"])

    assert int(torch.count_nonzero(sample["safe_active_mask"] * risk)) == 0


def test_envelope_edge_mask_ignores_stationary_carrier_cycles() -> None:
    time = np.arange(TARGET_SAMPLE_RATE // 2) / TARGET_SAMPLE_RATE
    signal = np.sin(2.0 * np.pi * 8_000.0 * time)
    signal[: TARGET_SAMPLE_RATE // 10] = 0.0

    mask = compute_envelope_edge_mask(signal, TARGET_SAMPLE_RATE)

    assert float(np.mean(mask)) < 0.05
    onset = TARGET_SAMPLE_RATE // 10
    assert float(np.sum(mask[onset - 512 : onset + 512])) > 0.0


def test_imd_two_tone_is_marked_stationary() -> None:
    config = CAPBDataConfig(num_samples=1, seed=11, signal_mix={"imd_two_tone": 1.0})
    sample = CAPBUpsampleDataset(config)[0]
    assert sample["signal_type"] == "imd_two_tone"
    assert bool(sample["stationary"])


@pytest.mark.parametrize("signal_type", ("sweep_log", "sweep_linear"))
def test_continuous_sweeps_require_fixed_stationary_routing(signal_type: str) -> None:
    """Sweeps must not create an interpolation-image ridge by weight motion."""
    config = CAPBDataConfig(num_samples=1, seed=11, signal_mix={signal_type: 1.0})
    sample = CAPBUpsampleDataset(config)[0]

    assert bool(sample["stationary"])


def test_sweep_start_sampling_covers_low_frequency_decades() -> None:
    """Log-uniform sweep starts must retain examples below 100 Hz."""
    config = CAPBDataConfig(num_samples=1, seed=11, signal_mix={"sweep_log": 1.0})
    dataset = CAPBUpsampleDataset(config)
    starts = [
        float(dataset._sample_request(np.random.default_rng(seed))[1]["start_hz"])
        for seed in range(64)
    ]

    assert min(starts) < 100.0
    assert max(starts) > 1_000.0


@pytest.mark.parametrize("signal_type", ["square_wave", "sawtooth_wave"])
def test_periodic_edge_signals_are_not_marked_stationary(signal_type: str) -> None:
    config = CAPBDataConfig(num_samples=1, seed=11, signal_mix={signal_type: 1.0})
    sample = CAPBUpsampleDataset(config)[0]
    assert sample["signal_type"] == signal_type
    assert not bool(sample["stationary"])


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
    assert config.signal_mix["imd_two_tone"] == pytest.approx(0.05)
    assert config.transient_supervision.enabled


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
