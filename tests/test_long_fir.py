"""Tests for long-FIR phase and echo evaluation."""

import numpy as np
import pytest
from scripts.report_long_fir_candidates import _add_release_comparisons

from totton_audio_de_mirroring.evaluation.long_fir import (
    evaluate_long_echo,
    evaluate_phase_alignment,
    validate_phase_alignment,
)
from totton_audio_de_mirroring.models.proto_bank import (
    PrototypeBank,
    build_prototype_bank_for_profile,
)


@pytest.mark.parametrize("target_rate", [88_200, 96_000])
def test_long_bank_has_common_phase_and_center(target_rate: int) -> None:
    bank = build_prototype_bank_for_profile(target_rate, "long_sharp_2047_a120")
    metrics = evaluate_phase_alignment(bank)

    validate_phase_alignment(metrics)

    assert metrics.kernel_length == 2047
    assert metrics.peak_indices == (1023, 1023, 1023)
    assert metrics.max_phase_spread_deg <= 1.0e-6
    assert metrics.max_group_delay_spread_samples <= 1.0e-9


def test_phase_validation_rejects_shifted_prototype() -> None:
    source = build_prototype_bank_for_profile(88_200, "long_sharp_1023_a120")
    shifted = source.kernels.copy()
    shifted[1] = np.roll(shifted[1], 1)
    bank = PrototypeBank(
        sample_rate=source.sample_rate,
        upsample_ratio=source.upsample_ratio,
        names=source.names,
        kernels=shifted,
        group_delay_samples=source.group_delay_samples,
    )

    with pytest.raises(ValueError, match="symmetry"):
        validate_phase_alignment(evaluate_phase_alignment(bank))


def test_long_echo_separates_near_and_far_windows() -> None:
    signal = np.zeros(4_000, dtype=np.float64)
    center = 2_000
    signal[center - 100] = 2.0
    signal[center - 400] = 3.0
    signal[center + 100] = 4.0
    signal[center + 400] = 5.0

    metrics = evaluate_long_echo(signal, center_index=center, sample_rate=48_000)

    assert metrics.pre_0p5_4ms > 0.0
    assert metrics.pre_4_12ms > 0.0
    assert metrics.post_0p5_4ms > 0.0
    assert metrics.post_4_12ms > 0.0


def test_long_echo_requires_full_context() -> None:
    with pytest.raises(ValueError, match="before"):
        evaluate_long_echo(np.zeros(100), center_index=50, sample_rate=48_000)


def test_release_comparison_keeps_arithmetic_floor_diagnostic() -> None:
    rows = {
        "release_v4": {
            "response_float32_coefficients": {"image_max_db": -95.0},
            "fixed_fir_float32_error": {"relative_rms_db": -130.0},
        },
        "candidate": {
            "response_float32_coefficients": {"image_max_db": -135.0},
            "fixed_fir_float32_error": {"relative_rms_db": -126.0},
            "structural_passed": True,
        },
    }

    _add_release_comparisons(rows)

    assert rows["candidate"]["image_improvement_db"] == pytest.approx(40.0)
    assert rows["candidate"]["fixed_fir_floor_improvement_db"] == pytest.approx(-4.0)
    assert rows["candidate"]["screening_eligible"]
