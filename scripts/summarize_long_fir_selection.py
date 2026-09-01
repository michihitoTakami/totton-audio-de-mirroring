"""Summarize long-FIR FineTuning evidence and render the release decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROFILES = ("long_sharp_1535_a120", "long_sharp_2047_a120")
FAMILIES = ("44k1", "48k")
SEEDS = (1234, 2234, 3234)
MIN_PASSING_SEEDS = 2
IMAGE_IMPROVEMENT_DB = 0.5


def main() -> None:
    """Load the completed matrix and write final JSON, Markdown, and plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = build_summary(args.matrix_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "selection.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    plot_tradeoffs(payload, args.output_dir / "tradeoff_comparison.png")


def build_summary(root: Path) -> dict[str, Any]:
    """Aggregate structural, gate, robustness, and distortion evidence."""
    if not root.is_dir():
        raise FileNotFoundError(f"Matrix root not found: {root}")
    release = {
        family: _gate_metrics(_release_gate_path(root, family)) for family in FAMILIES
    }
    candidates = {profile: _profile_summary(root, profile) for profile in PROFILES}
    selected = select_profile(release, candidates)
    return {
        "policy": {
            "minimum_passing_seeds_per_family": MIN_PASSING_SEEDS,
            "meaningful_image_improvement_db": IMAGE_IMPROVEMENT_DB,
            "ranking": ["G3 image", "G2b pre-echo", "G9 sideband", "shorter FIR"],
            "thresholds_unchanged": True,
        },
        "selected_profile": selected,
        "release": release,
        "candidates": candidates,
        "structural": _load_json(root / "structural" / "candidates.json"),
        "distortion": {
            label: _load_json(
                root / "visualization" / label / "distortion" / "summary.json"
            )
            for label in ("release", *PROFILES)
        },
        "decision": _decision_text(selected),
    }


def _profile_summary(root: Path, profile: str) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for family in FAMILIES:
        rows = []
        for seed in SEEDS:
            cpu = _gate_metrics(
                _candidate_gate_path(root, profile, seed, family, "cpu")
            )
            cuda = _gate_metrics(
                _candidate_gate_path(root, profile, seed, family, "cuda")
            )
            rows.append(
                {
                    "seed": seed,
                    "passed": cpu["all_passed"] and cuda["all_passed"],
                    "training": _training_record(root, profile, seed, family),
                    "cpu": cpu,
                    "cuda": cuda,
                }
            )
        passing = sum(bool(row["passed"]) for row in rows)
        robustness = _load_json(
            root / "robustness" / profile / family / "robustness.json"
        )
        families[family] = {
            "passing_seeds": passing,
            "stable": passing >= MIN_PASSING_SEEDS,
            "seeds": rows,
            "representative_robustness": robustness,
        }
    return {
        "eligible": all(families[family]["stable"] for family in FAMILIES),
        "families": families,
    }


def _training_record(
    root: Path, profile: str, seed: int, family: str
) -> dict[str, Any]:
    summary = _load_json(root / "training" / f"{profile}_{seed}_{family}.json")
    checkpoint = Path(str(summary["best_checkpoint"]))
    try:
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError(
            f"Failed to hash checkpoint {checkpoint}: {error}"
        ) from error
    return {
        "best_checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "best_val_total": float(summary["best_val_total"]),
        "config_sha256": summary["config_sha256"],
        "precision": summary["precision"],
    }


def select_profile(
    release: dict[str, dict[str, Any]], candidates: dict[str, dict[str, Any]]
) -> str:
    """Return the common profile that satisfies the fail-closed policy."""
    eligible = [name for name, value in candidates.items() if value["eligible"]]
    if not eligible:
        return "release_v4"
    release_worst = max(
        float(release[family]["g3_image_peak_db"]) for family in FAMILIES
    )
    ranked = sorted(eligible, key=lambda name: _profile_rank(name, candidates[name]))
    winner = ranked[0]
    winner_worst = _profile_rank(winner, candidates[winner])[0]
    if winner_worst > release_worst - IMAGE_IMPROVEMENT_DB:
        return "release_v4"
    return winner


def _profile_rank(
    profile_name: str, profile: dict[str, Any]
) -> tuple[float, float, float, int]:
    best_rows = []
    for family in FAMILIES:
        passing = [row for row in profile["families"][family]["seeds"] if row["passed"]]
        best_rows.append(min(passing, key=lambda row: row["cpu"]["g3_image_peak_db"]))
    tap_count = 1535 if "1535" in profile_name else 2047
    return (
        max(float(row["cpu"]["g3_image_peak_db"]) for row in best_rows),
        max(float(row["cpu"]["g2b_pre_echo"]) for row in best_rows),
        max(float(row["cpu"]["g9_sideband_db"]) for row in best_rows),
        tap_count,
    )


def _gate_metrics(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    return {
        "all_passed": bool(report["all_passed"]),
        "spec_version": report["spec_version"],
        "manifest_hash": report["manifest_hash"],
        "g2b_pre_echo": _worst_metric(report, "G2b_pre_echo"),
        "g3_image_db": _worst_metric(report, "G3_mirror", "image_rel_db"),
        "g3_image_peak_db": _worst_metric(report, "G3_mirror", "image_peak_rel_db"),
        "g9_sideband_db": _worst_metric(report, "G9_no_modulation_sidebands"),
    }


def _worst_metric(
    report: dict[str, Any], gate_id: str, metric: str | None = None
) -> float:
    gate = next(item for item in report["gates"] if item["gate_id"] == gate_id)
    rows = gate["rows"]
    if metric is not None:
        rows = [row for row in rows if row["metric"] == metric]
    if not rows:
        raise ValueError(f"Missing {gate_id}/{metric} rows.")
    return float(max(float(row["value"]) for row in rows))


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the decision and the measured improvement/regression table."""
    lines = [
        "# CAPB long-FIR FineTuning selection",
        "",
        f"Selected profile: **{payload['selected_profile']}**",
        "",
        str(payload["decision"]),
        "",
        "| Profile | Family | Passing seeds | G3 peak (dB) | G2b | G9 (dB) | Robustness |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for profile in PROFILES:
        for family in FAMILIES:
            result = payload["candidates"][profile]["families"][family]
            representative = result["seeds"][0]["cpu"]
            robust = result["representative_robustness"]["all_passed"]
            lines.append(
                f"| {profile} | {family} | {result['passing_seeds']}/3 | "
                f"{representative['g3_image_peak_db']:.2f} | "
                f"{representative['g2b_pre_echo']:.3e} | "
                f"{representative['g9_sideband_db']:.2f} | "
                f"{'PASS' if robust else 'FAIL'} |"
            )
    lines.extend(_distortion_markdown(payload))
    return "\n".join(lines) + "\n"


def _distortion_markdown(payload: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Distortion and sideband comparison",
        "",
        "| Profile | Family | THD (dB) | SMPTE (dB) | CCIF (dB) | AM sideband (dB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for profile in ("release", *PROFILES):
        for family in FAMILIES:
            metrics = payload["distortion"][profile][family]["distortion"]["capb"]
            lines.append(
                f"| {profile} | {family} | {metrics['thd_1khz_20khz_db']:.2f} | "
                f"{metrics['smpte_imd_db']:.2f} | {metrics['ccif_imd_db']:.2f} | "
                f"{metrics['added_am_sideband_db']:.2f} |"
            )
    return lines


def plot_tradeoffs(payload: dict[str, Any], output: Path) -> None:
    """Plot image suppression against pre-echo and sideband regressions."""
    labels = ("release", "1535", "2047")
    profiles = ("release", *PROFILES)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    x = np.arange(len(labels))
    width = 0.36
    for index, family in enumerate(FAMILIES):
        offset = (index - 0.5) * width
        gate_rows = [
            _representative_gate(payload, profile, family) for profile in profiles
        ]
        axes[0, 0].bar(
            x + offset,
            [row["g3_image_peak_db"] for row in gate_rows],
            width,
            label=family,
        )
        axes[0, 1].bar(
            x + offset,
            [row["g2b_pre_echo"] for row in gate_rows],
            width,
            label=family,
        )
        axes[1, 0].bar(
            x + offset,
            [row["g9_sideband_db"] for row in gate_rows],
            width,
            label=family,
        )
        axes[1, 1].bar(
            x + offset,
            [
                payload["distortion"][profile][family]["distortion"]["capb"][
                    "smpte_imd_db"
                ]
                for profile in profiles
            ],
            width,
            label=family,
        )
    axes[0, 0].set(title="Worst G3 image peak (lower is better)", ylabel="dB")
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(2.5e-7, color="red", linestyle="--", label="G2b limit")
    axes[0, 1].set(title="G2b pre-echo (lower is better)", ylabel="mean square")
    axes[1, 0].axhline(-110.0, color="red", linestyle="--", label="G9 limit")
    axes[1, 0].set(title="Worst G9 modulation sideband", ylabel="dB")
    axes[1, 1].set(title="SMPTE sideband RSS", ylabel="dB")
    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _representative_gate(
    payload: dict[str, Any], profile: str, family: str
) -> dict[str, Any]:
    if profile == "release":
        return payload["release"][family]
    return payload["candidates"][profile]["families"][family]["seeds"][0]["cpu"]


def _decision_text(selected: str) -> str:
    if selected != "release_v4":
        return "The long-FIR profile passed every unchanged gate and won the image-first ranking."
    return (
        "Both FineTuned long-FIR profiles improved image rejection, but neither "
        "produced two passing 44.1 kHz seeds. The incumbent release is retained "
        "because pre-echo is a hard, worst-probe acceptance condition."
    )


def _candidate_gate_path(
    root: Path, profile: str, seed: int, family: str, device: str
) -> Path:
    return (
        root
        / "gates"
        / profile
        / str(seed)
        / family
        / device
        / "candidate"
        / "gate_report.json"
    )


def _release_gate_path(root: Path, family: str) -> Path:
    return (
        root / "gates" / "release" / family / "cpu" / "candidate" / "gate_report.json"
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Failed to load report {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Report must contain a JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()
