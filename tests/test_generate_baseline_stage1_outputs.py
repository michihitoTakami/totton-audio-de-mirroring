"""Tests for Issue #109 baseline per-channel inference generation."""

from __future__ import annotations

import numpy as np
import pytest
from scripts.generate_baseline_stage1_outputs import (
    run_per_channel_inference,
    to_float_channels,
)


class _DummyStage1Processor:
    """Dummy Stage1 processor for channel-wise inference tests.

    Physical Basis:
        Deterministic processor output makes it possible to verify channel
        independence without checkpoint/model variability.
    """

    def process(
        self,
        signal: np.ndarray,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray:
        _ = source_sample_rate
        _ = target_sample_rate
        signal_arr = np.asarray(signal, dtype=np.float64)
        return np.repeat(signal_arr, 2)


def test_to_float_channels_normalizes_int32() -> None:
    """Normalize int32 waveform into float64 channels.

    Physical Basis:
        24-bit-in-int32 WAV data requires explicit scaling to prevent overflow
        and clipping artifacts during inference.
    """

    raw = np.array([[2**30, -(2**30)]], dtype=np.int32)
    channels = to_float_channels(raw)
    assert channels.shape == (1, 2)
    assert channels.dtype == np.float64
    assert channels[0, 0] == pytest.approx(0.5)
    assert channels[0, 1] == pytest.approx(-0.5)


def test_run_per_channel_inference_preserves_stereo_shape() -> None:
    """Run inference for each channel and re-stack outputs.

    Physical Basis:
        Independent channel inference must preserve stereo layout so TFS and
        binaural metrics consume proper two-channel signals.
    """

    processor = _DummyStage1Processor()
    signal = np.array(
        [
            [0.1, 0.4],
            [0.2, 0.5],
            [0.3, 0.6],
        ],
        dtype=np.float64,
    )
    output = run_per_channel_inference(
        processor=processor,
        signal=signal,
        source_sample_rate=44_100,
        target_sample_rate=88_200,
    )
    expected_left = np.repeat(signal[:, 0], 2)
    expected_right = np.repeat(signal[:, 1], 2)
    assert output.shape == (6, 2)
    assert np.allclose(output[:, 0], expected_left)
    assert np.allclose(output[:, 1], expected_right)
