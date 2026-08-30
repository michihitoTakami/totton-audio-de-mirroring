"""Evaluate strict-FP32 CAPB cross-rate release quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from totton_audio_de_mirroring.evaluation.release_quality import (
    evaluate_release_quality,
)


def main() -> None:
    """Load evidence, evaluate it, and write JSON/Markdown results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distortion-summary", type=Path, required=True)
    parser.add_argument("--impulse-metrics", type=Path, required=True)
    parser.add_argument("--gate-44k1", type=Path, required=True)
    parser.add_argument("--gate-48k", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate_release_quality(
        _load_json(args.distortion_summary),
        _load_json(args.impulse_metrics),
        _load_json(args.gate_44k1),
        _load_json(args.gate_48k),
    )
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "release_quality.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        markdown = _render_markdown(result)
        (args.output_dir / "release_quality.md").write_text(markdown, encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"Failed to write release-quality report: {error}"
        ) from error
    print(markdown)
    if not result["all_passed"]:
        raise SystemExit(1)


def _load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON object with actionable errors."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Failed to load JSON report {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report must contain an object: {path}")
    return payload


def _render_markdown(result: dict[str, Any]) -> str:
    """Render one compact cross-rate release report."""
    rows = []
    for check in result["checks"]:
        value = check.get("value", "-")
        threshold = check.get("threshold", "-")
        rows.append(
            f"| {check['check_id']} | {value} | {threshold} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )
    return "\n".join(
        [
            "# CAPB strict-FP32 release quality",
            "",
            f"Overall: **{'PASS' if result['all_passed'] else 'FAIL'}**",
            "",
            "| Check | Value | Maximum | Result |",
            "|---|---:|---:|---|",
            *rows,
            "",
        ]
    )


if __name__ == "__main__":
    main()
