import numpy as np
import pytest

from totton_audio_de_mirroring.dsp.multistage_upsampler import (
    UpsampleStageConfig,
    default_stage_configs,
    design_stage_taps,
    multistage_upsample,
    upsample_by_2,
)

INPUT_SAMPLE_RATE = 88_200
DEFAULT_SIGNAL_LEN = 1024


def test_design_stage_taps_returns_finite_response() -> None:
    config = UpsampleStageConfig(num_taps=63)
    taps = design_stage_taps(config, target_sample_rate=INPUT_SAMPLE_RATE * 2)
    assert taps.ndim == 1
    assert taps.size > 0
    assert np.isfinite(taps).all()


def test_upsample_by_2_doubles_length() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(0.0, 1.0, DEFAULT_SIGNAL_LEN)
    config = UpsampleStageConfig(num_taps=63)

    upsampled, new_rate = upsample_by_2(signal, INPUT_SAMPLE_RATE, config)

    assert upsampled.shape == (DEFAULT_SIGNAL_LEN * 2,)
    assert new_rate == INPUT_SAMPLE_RATE * 2


def test_multistage_upsample_reaches_8x_rate() -> None:
    rng = np.random.default_rng(1)
    signal = rng.normal(0.0, 1.0, 512)
    stages = default_stage_configs(num_stages=3, num_taps=63)

    upsampled, final_rate = multistage_upsample(
        signal,
        input_sample_rate=INPUT_SAMPLE_RATE,
        stages=stages,
    )

    assert upsampled.shape == (signal.shape[-1] * 8,)
    assert final_rate == INPUT_SAMPLE_RATE * 8


def test_multistage_upsample_rejects_empty_stage_list() -> None:
    signal = np.ones(128)

    with pytest.raises(ValueError, match="at least one stage"):
        multistage_upsample(signal, input_sample_rate=INPUT_SAMPLE_RATE, stages=[])


def test_design_stage_taps_rejects_invalid_cutoff() -> None:
    config = UpsampleStageConfig(cutoff_hz=INPUT_SAMPLE_RATE)

    with pytest.raises(ValueError, match="less than Nyquist"):
        design_stage_taps(config, target_sample_rate=INPUT_SAMPLE_RATE * 2)
