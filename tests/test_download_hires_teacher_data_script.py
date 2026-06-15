"""Tests for the hi-res teacher download CLI script."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scripts import download_hires_teacher_data as dl


def _write_hires_wav(
    path: Path, *, sample_rate: int = 88_200, include_ultrasonic: bool = True
) -> None:
    t = np.arange(int(2.0 * sample_rate)) / sample_rate
    signal = 0.3 * np.sin(2 * np.pi * 5_000 * t)
    if include_ultrasonic:
        signal = signal + 0.2 * np.sin(2 * np.pi * 30_000 * t)
    sf.write(str(path), signal.astype(np.float32), sample_rate, subtype="PCM_24")


@pytest.mark.parametrize(
    "license_id,allow_nc,expected",
    [
        ("CC0-1.0", False, True),
        ("CC-BY-4.0", False, True),
        ("public-domain", False, True),
        ("CC-BY-NC-4.0", False, False),
        ("CC-BY-NC-4.0", True, True),
        ("proprietary", False, False),
        ("MIT", False, False),
    ],
)
def test_license_is_allowed(license_id: str, allow_nc: bool, expected: bool) -> None:
    assert dl._license_is_allowed(license_id, allow_nc) is expected


def test_load_manifest_empty_entries(tmp_path: Path) -> None:
    manifest = tmp_path / "m.yaml"
    manifest.write_text("entries: []\n", encoding="utf-8")
    assert dl._load_manifest(manifest) == []


def test_load_manifest_rejects_missing_field(tmp_path: Path) -> None:
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        "entries:\n  - name: a.wav\n    url: file:///a.wav\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required field"):
        dl._load_manifest(manifest)


def test_process_entry_accepts_valid_file_url(tmp_path: Path) -> None:
    source = tmp_path / "src.wav"
    _write_hires_wav(source)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    entry = dl.ManifestEntry(
        name="track.wav",
        url=source.as_uri(),
        license="CC-BY-4.0",
        attribution="Tester - Tone (CC BY 4.0)",
        source_url="https://example.org/page",
        sha256=None,
    )
    record = dl._process_entry(
        entry,
        output_dir=out_dir,
        min_sample_rate=88_200,
        min_hf_energy_ratio=1.0e-6,
        allow_noncommercial=False,
    )
    assert record is not None
    assert record["sample_rate"] == 88_200
    assert (out_dir / "track.wav").exists()


def test_process_entry_rejects_low_sample_rate(tmp_path: Path) -> None:
    source = tmp_path / "low.wav"
    _write_hires_wav(source, sample_rate=48_000)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    entry = dl.ManifestEntry(
        name="low.wav",
        url=source.as_uri(),
        license="CC0-1.0",
        attribution="Tester",
        source_url="https://example.org",
        sha256=None,
    )
    record = dl._process_entry(
        entry,
        output_dir=out_dir,
        min_sample_rate=88_200,
        min_hf_energy_ratio=1.0e-6,
        allow_noncommercial=False,
    )
    assert record is None
    assert not (out_dir / "low.wav").exists()


def test_write_attribution_creates_files(tmp_path: Path) -> None:
    records = [
        {
            "name": "track.wav",
            "license": "CC-BY-4.0",
            "attribution": "Tester - Tone",
            "source_url": "https://example.org",
            "sample_rate": 88_200,
            "hf_energy_ratio": 0.3,
            "sha256": "abc",
        }
    ]
    dl._write_attribution(tmp_path, records)
    assert (tmp_path / "ATTRIBUTION.md").exists()
    assert (tmp_path / "downloaded_manifest.json").exists()
    assert "CC-BY-4.0" in (tmp_path / "ATTRIBUTION.md").read_text(encoding="utf-8")
