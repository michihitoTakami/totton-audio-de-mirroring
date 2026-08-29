"""Tests for CAPB data-audit and transient-robustness helpers."""

import numpy as np
import pytest
from scripts.audit_capb_training_data import audit_dataset
from scripts.evaluate_capb_transient_robustness import _evaluate_offsets

from totton_audio_de_mirroring.data.capb_dataset import (
    TARGET_SAMPLE_RATE,
    CAPBDataConfig,
    CAPBUpsampleDataset,
    TransientSupervisionConfig,
)
from totton_audio_de_mirroring.inference.pipeline import ReferenceStage1Processor


def test_data_audit_counts_focused_events() -> None:
    config = CAPBDataConfig(
        num_samples=4,
        seed=5,
        signal_mix={"isolated_click": 1.0},
        transient_supervision=TransientSupervisionConfig(enabled=True),
    )
    result = audit_dataset(
        CAPBUpsampleDataset(config), TARGET_SAMPLE_RATE, fft_samples=2
    )
    assert result["family_counts"] == {"isolated_click": 4}
    assert result["transient_event_counts"] == {"isolated_click": 4}
    assert result["max_decimation_error"] == 0.0


def test_offset_evaluator_passes_reference_against_itself() -> None:
    processor = ReferenceStage1Processor()

    def reference(signal: np.ndarray) -> np.ndarray:
        return processor.process(signal, 44_100, 88_200)

    result = _evaluate_offsets(
        offsets=range(2),
        event_position=lambda offset, _hop: 22_050 + offset,
        candidate=reference,
        reference=reference,
    )
    assert result["all_passed"]
    assert len(result["rows"]) == 2


def test_offset_evaluator_supports_48k_family() -> None:
    processor = ReferenceStage1Processor()

    def reference(signal: np.ndarray) -> np.ndarray:
        return processor.process(signal, 48_000, 96_000)

    result = _evaluate_offsets(
        offsets=range(2),
        event_position=lambda offset, _hop: 24_000 + offset,
        candidate=reference,
        reference=reference,
        source_rate=48_000,
        target_rate=96_000,
    )
    assert result["all_passed"]


def test_offset_evaluator_rejects_event_outside_buffer() -> None:
    with pytest.raises(ValueError, match="outside"):
        _evaluate_offsets(
            offsets=range(1),
            event_position=lambda _offset, _hop: 44_100,
            candidate=lambda signal: np.repeat(signal, 2),
            reference=lambda signal: np.repeat(signal, 2),
        )
