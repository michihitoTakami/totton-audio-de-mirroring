"""Download and validate hi-res teacher audio from a license-aware manifest.

The manifest is a YAML/JSON document::

    entries:
      - name: my_track.flac          # local filename
        url: https://example.org/my_track.flac
        license: CC-BY-4.0           # CC0 / CC-BY* / public-domain (SA/ND allowed)
        attribution: "Artist - Title (CC BY 4.0), https://example.org/page"
        source_url: https://example.org/page
        sha256: <optional hex digest>

Each downloaded file is validated to have a native sample rate >= the target
rate and genuine energy above 22.05kHz, then recorded in an attribution file.

Physical Basis:
    Genuine hi-res teachers must carry real >22.05kHz energy; files upsampled
    from a band-limited master add no information and are rejected. License
    metadata is mandatory so training data provenance stays auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from totton_audio_de_mirroring.data.hires_corpus import high_frequency_energy_ratio

# SA (share-alike) and ND (no-derivatives) do not restrict training use; NC does.
_ALLOWED_LICENSE_PREFIXES: tuple[str, ...] = (
    "cc0",
    "cc-by",
    "ccby",
    "public-domain",
    "publicdomain",
    "pd",
)
_NONCOMMERCIAL_MARKERS: tuple[str, ...] = ("nc", "noncommercial", "non-commercial")
_DEFAULT_SPLIT_HZ = 22_050.0


@dataclass(frozen=True)
class ManifestEntry:
    """One hi-res source entry."""

    name: str
    url: str
    license: str
    attribution: str
    source_url: str
    sha256: str | None


def main() -> None:
    """Download, validate, and record hi-res teacher audio."""
    args = _parse_args()
    entries = _load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accepted: list[dict[str, Any]] = []
    for entry in entries[: args.max_entries] if args.max_entries else entries:
        result = _process_entry(
            entry,
            output_dir=args.output_dir,
            min_sample_rate=args.min_sample_rate,
            min_hf_energy_ratio=args.min_hf_energy_ratio,
            allow_noncommercial=args.allow_noncommercial,
        )
        if result is not None:
            accepted.append(result)

    _write_attribution(args.output_dir, accepted)
    print(
        f"accepted={len(accepted)} of {len(entries)} entries -> {args.output_dir}",
        flush=True,
    )
    if not accepted:
        print(
            "No usable hi-res files were downloaded. Populate the manifest with "
            "CC-BY/CC0 sources at >= the target sample rate, or drop your own "
            "licensed hi-res files directly into the corpus directory.",
            flush=True,
        )


def _process_entry(
    entry: ManifestEntry,
    *,
    output_dir: Path,
    min_sample_rate: int,
    min_hf_energy_ratio: float,
    allow_noncommercial: bool,
) -> dict[str, Any] | None:
    """Download and validate one manifest entry.

    Returns:
        Attribution record on success, otherwise None.
    """
    if not _license_is_allowed(entry.license, allow_noncommercial):
        print(
            f"skip {entry.name}: license not allowed ({entry.license!r}).", flush=True
        )
        return None

    destination = output_dir / entry.name
    try:
        _download(entry.url, destination)
    except Exception as exc:
        print(f"skip {entry.name}: download failed ({exc}).", flush=True)
        return None

    if entry.sha256 is not None and not _sha256_matches(destination, entry.sha256):
        print(f"skip {entry.name}: sha256 mismatch.", flush=True)
        destination.unlink(missing_ok=True)
        return None

    validation = _validate_audio(destination, min_sample_rate, min_hf_energy_ratio)
    if validation is None:
        destination.unlink(missing_ok=True)
        return None

    sample_rate, hf_ratio = validation
    print(
        f"ok {entry.name}: sr={sample_rate} hf_ratio={hf_ratio:.3e}",
        flush=True,
    )
    return {
        "name": entry.name,
        "license": entry.license,
        "attribution": entry.attribution,
        "source_url": entry.source_url,
        "sample_rate": sample_rate,
        "hf_energy_ratio": hf_ratio,
        "sha256": _sha256_digest(destination),
    }


def _validate_audio(
    path: Path, min_sample_rate: int, min_hf_energy_ratio: float
) -> tuple[int, float] | None:
    """Validate sample rate and high-frequency energy of a downloaded file."""
    try:
        info = sf.info(str(path))
    except Exception as exc:
        print(f"skip {path.name}: unreadable audio ({exc}).", flush=True)
        return None
    sample_rate = int(info.samplerate)
    if sample_rate < min_sample_rate:
        print(
            f"skip {path.name}: sample rate {sample_rate} < {min_sample_rate}.",
            flush=True,
        )
        return None
    try:
        block, _ = sf.read(str(path), dtype="float64", always_2d=True)
    except Exception as exc:
        print(f"skip {path.name}: read failed ({exc}).", flush=True)
        return None
    mono = block[:, 0] if block.shape[1] == 1 else block.mean(axis=1)
    hf_ratio = high_frequency_energy_ratio(
        mono, sample_rate, split_hz=_DEFAULT_SPLIT_HZ
    )
    if hf_ratio < min_hf_energy_ratio:
        print(
            f"skip {path.name}: hf ratio {hf_ratio:.3e} < {min_hf_energy_ratio:.3e}.",
            flush=True,
        )
        return None
    return sample_rate, hf_ratio


def _license_is_allowed(license_id: str, allow_noncommercial: bool) -> bool:
    normalized = license_id.strip().lower().replace(" ", "")
    if not allow_noncommercial and any(
        marker in normalized for marker in _NONCOMMERCIAL_MARKERS
    ):
        return False
    return any(normalized.startswith(prefix) for prefix in _ALLOWED_LICENSE_PREFIXES)


def _download(url: str, destination: Path) -> None:
    """Download a URL to a destination path."""
    if not url.lower().startswith(("http://", "https://", "file://")):
        raise ValueError(f"Unsupported URL scheme: {url!r}.")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        data = response.read()
    destination.write_bytes(data)


def _sha256_digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _sha256_matches(path: Path, expected: str) -> bool:
    return _sha256_digest(path).lower() == expected.strip().lower()


def _write_attribution(output_dir: Path, records: list[dict[str, Any]]) -> None:
    """Write machine- and human-readable attribution records."""
    (output_dir / "downloaded_manifest.json").write_text(
        json.dumps({"entries": records}, indent=2),
        encoding="utf-8",
    )
    lines = ["# Hi-Res Teacher Data Attribution", ""]
    for record in records:
        lines.append(f"- **{record['name']}** ({record['license']})")
        lines.append(f"  - {record['attribution']}")
        lines.append(f"  - source: {record['source_url']}")
        lines.append(
            f"  - sample_rate: {record['sample_rate']} Hz, "
            f"hf_energy_ratio: {record['hf_energy_ratio']:.3e}"
        )
    (output_dir / "ATTRIBUTION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _load_manifest(path: Path) -> list[ManifestEntry]:
    """Load and validate manifest entries from YAML or JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore

        raw = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raise ValueError("Manifest must be .yaml, .yml, or .json")
    if not isinstance(raw, dict) or "entries" not in raw:
        raise ValueError("Manifest must contain an 'entries' list.")
    raw_entries = raw["entries"] or []
    if not isinstance(raw_entries, list):
        raise ValueError("Manifest 'entries' must be a list.")
    return [_parse_entry(item) for item in raw_entries]


def _parse_entry(raw: Any) -> ManifestEntry:
    if not isinstance(raw, dict):
        raise ValueError("Each manifest entry must be a mapping.")
    for field_name in ("name", "url", "license", "attribution", "source_url"):
        if not isinstance(raw.get(field_name), str) or not raw[field_name].strip():
            raise ValueError(f"Manifest entry missing required field: {field_name!r}.")
    sha256 = raw.get("sha256")
    return ManifestEntry(
        name=raw["name"].strip(),
        url=raw["url"].strip(),
        license=raw["license"].strip(),
        attribution=raw["attribution"].strip(),
        source_url=raw["source_url"].strip(),
        sha256=sha256.strip() if isinstance(sha256, str) and sha256.strip() else None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download hi-res teacher audio.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/hires_teacher_manifest.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hires_corpus"),
    )
    parser.add_argument("--min-sample-rate", type=int, default=88_200)
    parser.add_argument("--min-hf-energy-ratio", type=float, default=1.0e-6)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument(
        "--allow-noncommercial",
        action="store_true",
        help="Permit licenses containing NC (non-commercial) markers.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
