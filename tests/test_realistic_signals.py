"""Tests for real-audio-inspired procedural CAPB signals."""

import numpy as np
import pytest

from totton_audio_de_mirroring.data.capb_dataset import (
    CAPBDataConfig,
    CAPBUpsampleDataset,
    TransientSupervisionConfig,
)
from totton_audio_de_mirroring.data.generator import generate_signal


@pytest.mark.parametrize(
    ("signal_type", "params"),
    (
        (
            "damped_string",
            {"fundamental_hz": 220.0, "event_duration_ms": 80.0},
        ),
        (
            "clustered_impacts",
            {"impact_count": 4, "cluster_duration_ms": 30.0},
        ),
        (
            "flowing_noise",
            {"low_hz": 100.0, "high_hz": 12_000.0, "modulation_hz": 1.0},
        ),
        ("string_riff", {"fundamental_hz": 110.0, "interval_ms": 120.0}),
        ("impact_stream", {"event_rate_hz": 12.0}),
    ),
)
def test_realistic_signal_is_finite_and_bounded(
    signal_type: str, params: dict[str, float | int]
) -> None:
    signal = generate_signal(
        signal_type,
        sample_rate=44_100,
        duration_sec=0.5,
        seed=7,
        **params,
    )

    assert signal.shape == (22_050,)
    assert signal.dtype == np.float32
    assert np.all(np.isfinite(signal))
    assert float(np.max(np.abs(signal))) == pytest.approx(0.9)


@pytest.mark.parametrize("signal_type", ("damped_string", "clustered_impacts"))
def test_realistic_transient_dataset_preserves_lattice(signal_type: str) -> None:
    config = CAPBDataConfig(
        num_samples=1,
        seed=13,
        signal_mix={signal_type: 1.0},
        transient_supervision=TransientSupervisionConfig(
            enabled=True,
            focus_signal_types=(signal_type,),
        ),
    )

    sample = CAPBUpsampleDataset(config)[0]

    np.testing.assert_array_equal(
        sample["source"].numpy(), sample["target"].numpy()[::2]
    )
    assert bool(sample["focused_event"])
    assert float(sample["pre_echo_mask"].sum()) > 0.0
    assert float(sample["post_echo_mask"].sum()) > 0.0


def test_flowing_noise_is_stationary_training_family() -> None:
    sample = CAPBUpsampleDataset(
        CAPBDataConfig(num_samples=1, signal_mix={"flowing_noise": 1.0})
    )[0]

    assert bool(sample["stationary"])


def test_flowing_noise_rejects_invalid_band() -> None:
    with pytest.raises(ValueError, match="noise band"):
        generate_signal(
            "flowing_noise",
            sample_rate=44_100,
            duration_sec=0.5,
            low_hz=12_000.0,
            high_hz=24_000.0,
            modulation_hz=1.0,
        )
