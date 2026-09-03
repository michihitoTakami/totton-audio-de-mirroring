"""Tests for CAPB real-audio routing diagnostics."""

import numpy as np
import pytest

from totton_audio_de_mirroring.evaluation.routing_analysis import (
    summarize_routing,
    transient_strength,
)


def test_transient_strength_peaks_at_impulse() -> None:
    signal = np.zeros(6_400)
    signal[3_200] = 1.0

    strength = transient_strength(signal, 100)

    assert 49 <= int(np.argmax(strength)) <= 50
    assert float(np.max(strength)) == pytest.approx(1.0)


def test_routing_summary_rewards_sharp_safe_and_gentle_risk() -> None:
    signal = 0.1 * np.sin(2.0 * np.pi * np.arange(6_400) / 64.0)
    signal[3_200] += 1.0
    strength = transient_strength(signal, 100)
    risk = strength >= np.quantile(strength, 0.95)
    weights = np.zeros((2, 100))
    weights[0] = 0.95
    weights[0, risk] = 0.05
    weights[1] = 1.0 - weights[0]

    summary = summarize_routing(signal, weights, sharp_index=0, gentle_index=1)

    assert summary.sharp_safe_mean > 0.9
    assert summary.gentle_risk_mean > 0.9
    assert summary.routing_contrast > 0.8


def test_routing_summary_rejects_nonconvex_weights() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        summarize_routing(np.ones(64), np.ones((2, 4)), 0, 1)
