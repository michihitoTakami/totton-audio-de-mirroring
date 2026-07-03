"""Tests for the deterministic probe suite."""

from pathlib import Path

import numpy as np
import pytest

from totton_audio_de_mirroring.evaluation.probe_suite import (
    DEFAULT_SOURCE_SAMPLE_RATE,
    TIER_CANONICAL,
    TIER_HELD_OUT,
    ProbeSpec,
    build_default_probe_suite,
    generate_probe,
    load_manifest,
    manifest_hash,
    save_manifest,
    suite_from_manifest,
    suite_manifest,
)

FIXTURE_PATH = Path("tests/fixtures/probes/probe_manifest.json")


@pytest.fixture(scope="module")
def suite():
    """Build the default probe suite once."""
    return build_default_probe_suite()


def test_suite_has_both_tiers(suite) -> None:
    tiers = {spec.tier for spec in suite}
    assert tiers == {TIER_CANONICAL, TIER_HELD_OUT}


def test_probe_ids_are_unique(suite) -> None:
    ids = [spec.probe_id for spec in suite]
    assert len(ids) == len(set(ids))


def test_held_out_frequencies_do_not_overlap_canonical(suite) -> None:
    """Goodhart guard: held-out square frequencies are unseen by training."""
    canonical = {
        spec.frequency_hz
        for spec in suite
        if spec.tier == TIER_CANONICAL and spec.kind == "square"
    }
    held_out = {
        spec.frequency_hz
        for spec in suite
        if spec.tier == TIER_HELD_OUT and spec.kind == "square"
    }
    assert canonical.isdisjoint(held_out)


def test_all_probes_generate(suite) -> None:
    for spec in suite:
        wave = generate_probe(spec)
        expected = int(round(spec.duration_sec * DEFAULT_SOURCE_SAMPLE_RATE))
        assert wave.shape == (expected,), spec.probe_id
        assert np.max(np.abs(wave)) <= spec.amplitude * 1.0001, spec.probe_id


def test_generation_is_deterministic(suite) -> None:
    for spec in suite:
        first = generate_probe(spec)
        second = generate_probe(spec)
        np.testing.assert_array_equal(first, second)


def test_dc_step_shape() -> None:
    spec = ProbeSpec(probe_id="step", kind="dc_step", tier=TIER_CANONICAL, step_sign=1)
    wave = generate_probe(spec)
    assert wave[0] == -spec.amplitude
    assert wave[-1] == spec.amplitude


def test_manifest_roundtrip(suite, tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    save_manifest(suite, path)
    restored = load_manifest(path)
    assert restored == suite


def test_manifest_hash_matches_frozen_fixture(suite) -> None:
    """The suite is frozen; changing it requires the golden-update procedure."""
    frozen = load_manifest(FIXTURE_PATH)
    assert manifest_hash(suite_manifest(frozen)) == manifest_hash(suite_manifest(suite))


def test_manifest_rejects_unknown_version(suite) -> None:
    manifest = suite_manifest(suite)
    manifest["version"] = 999
    with pytest.raises(ValueError, match="version"):
        suite_from_manifest(manifest)


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="kind"):
        ProbeSpec(probe_id="x", kind="nope", tier=TIER_CANONICAL)
    with pytest.raises(ValueError, match="tier"):
        ProbeSpec(probe_id="x", kind="square", tier="nope")
    with pytest.raises(ValueError, match="requires frequency_hz"):
        generate_probe(ProbeSpec(probe_id="x", kind="square", tier=TIER_CANONICAL))
