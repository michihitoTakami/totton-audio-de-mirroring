"""Audit CAPB training-family coverage, masks, and physical consistency."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from totton_audio_de_mirroring.data.capb_dataset import (
    UPSAMPLE_RATIO,
    CAPBUpsampleDataset,
    load_capb_data_config,
)

_TRANSIENT_TYPES = frozenset({"isolated_click", "tone_burst"})


def parse_args() -> argparse.Namespace:
    """Parse audit arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--fft-samples", type=int, default=256)
    return parser.parse_args()


def audit_dataset(
    dataset: CAPBUpsampleDataset,
    target_sample_rate: int,
    fft_samples: int,
) -> dict[str, Any]:
    """Collect deterministic dataset statistics without mutating samples.

    Physical Basis:
        Training coverage must count actual event-bearing chunks, not only
        requested generator labels. Exact decimation and image-band checks
        ensure the audit cannot reward data that violates the CAPB input
        information boundary.
    """
    if len(dataset) <= 0 or target_sample_rate <= 0 or fft_samples <= 0:
        raise ValueError("Dataset, sample rate, and fft_samples must be positive.")
    counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    mask_sums: dict[str, list[float]] = defaultdict(list)
    clean_count = 0
    augmented_count = 0
    max_decimation_error = 0.0
    worst_image_db = -300.0
    for index in range(len(dataset)):
        sample = dataset[index]
        signal_type = str(sample["signal_type"])
        counts[signal_type] += 1
        target = sample["target"].numpy().astype(np.float64)
        source = sample["source"].numpy().astype(np.float64)
        error = float(np.max(np.abs(source - target[::UPSAMPLE_RATIO])))
        max_decimation_error = max(max_decimation_error, error)
        for name in ("flat_mask", "quiet_mask", "edge_mask", "pre_echo_mask"):
            mask_sums[name].append(float(sample[name].float().mean()))
        if signal_type in _TRANSIENT_TYPES:
            if float(sample["quiet_mask"].float().mean()) < 0.999:
                event_counts[signal_type] += 1
            if bool(sample["transient_clean"]):
                clean_count += 1
            elif bool(sample["focused_event"]):
                augmented_count += 1
        if index < fft_samples:
            worst_image_db = max(
                worst_image_db,
                _image_to_main_db(target, target_sample_rate),
            )
    transient_total = clean_count + augmented_count
    return {
        "num_samples": len(dataset),
        "family_counts": dict(sorted(counts.items())),
        "transient_event_counts": dict(sorted(event_counts.items())),
        "focused_clean_count": clean_count,
        "focused_augmented_count": augmented_count,
        "focused_clean_fraction": (
            clean_count / transient_total if transient_total else None
        ),
        "mask_mean_fraction": {
            name: float(np.mean(values)) for name, values in sorted(mask_sums.items())
        },
        "max_decimation_error": max_decimation_error,
        "worst_sampled_image_to_main_db": worst_image_db,
        "fft_samples": min(fft_samples, len(dataset)),
    }


def _image_to_main_db(signal: np.ndarray, sample_rate: int) -> float:
    """Return image-band peak relative to the audible-band peak."""
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    main = spectrum[(frequencies >= 20.0) & (frequencies <= 20_000.0)]
    image = spectrum[frequencies >= sample_rate / 4.0 + 500.0]
    if main.size == 0 or image.size == 0:
        raise ValueError("Signal is too short for CAPB band audit.")
    return float(
        20.0
        * np.log10((float(np.max(image)) + 1.0e-300) / (float(np.max(main)) + 1.0e-300))
    )


def render_markdown(result: dict[str, Any], config_path: Path) -> str:
    """Render a compact human-readable audit report."""
    rows = "\n".join(
        f"| {name} | {count} |" for name, count in result["family_counts"].items()
    )
    event_rows = "\n".join(
        f"| {name} | {result['family_counts'].get(name, 0)} | "
        f"{result['transient_event_counts'].get(name, 0)} |"
        for name in sorted(_TRANSIENT_TYPES)
    )
    return f"""# CAPB training-data audit

- Config: `{config_path}`
- Samples: {result["num_samples"]}
- Exact-decimation maximum error: {result["max_decimation_error"]:.3e}
- Worst sampled image/main: {result["worst_sampled_image_to_main_db"]:.2f} dB
- Focused clean/augmented: {result["focused_clean_count"]} / {result["focused_augmented_count"]}

## Family counts

| Family | Count |
|---|---:|
{rows}

## Sparse transient containment

| Family | Requested chunks | Event-bearing chunks |
|---|---:|---:|
{event_rows}
"""


def main() -> None:
    """Run the audit and write JSON/Markdown evidence."""
    args = parse_args()
    config = load_capb_data_config(args.data_config)
    if args.num_samples is not None:
        from dataclasses import replace

        config = replace(config, num_samples=args.num_samples)
    result = audit_dataset(
        CAPBUpsampleDataset(config), config.target_sample_rate, args.fft_samples
    )
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "data_audit.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        (args.output_dir / "data_audit.md").write_text(
            render_markdown(result, args.data_config), encoding="utf-8"
        )
    except OSError as error:
        raise RuntimeError(f"Failed to write data audit: {error}") from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
