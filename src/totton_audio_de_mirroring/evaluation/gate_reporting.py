"""Serializers for a Stage 1 gate report.

Physical Basis:
    A pass is only meaningful next to the probe suite, the threshold set and
    the gate spec it was earned against, and an omitted row must never read as
    a passing row, so both serializers carry provenance and skip records.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from totton_audio_de_mirroring.evaluation.gate_types import GateReport


def report_to_dict(report: GateReport) -> dict[str, Any]:
    """Serialize a GateReport into a JSON-compatible dictionary."""
    return {
        "all_passed": report.all_passed,
        "spec_version": report.spec_version,
        "manifest_hash": report.manifest_hash,
        "config": report.config,
        "gates": [
            {
                "gate_id": gate.gate_id,
                "passed": gate.passed,
                "worst_probe_id": gate.worst_probe_id,
                "rows": [asdict(row) for row in gate.rows],
                "skipped": [asdict(item) for item in gate.skipped],
            }
            for gate in report.gates
        ],
    }


def render_markdown_report(report: GateReport) -> str:
    """Render a GateReport as a per-probe markdown table (worst-first)."""
    lines = [
        "# Stage 1 probe gate report",
        "",
        f"- all_passed: **{report.all_passed}**",
        f"- spec_version: {report.spec_version}",
        f"- manifest_hash: {report.manifest_hash}",
        "",
    ]
    for gate in report.gates:
        status = "PASS" if gate.passed else "FAIL"
        lines += [
            f"## {gate.gate_id}: {status}"
            + (f" (worst: {gate.worst_probe_id})" if gate.worst_probe_id else ""),
            "",
            "| probe | tier | metric | value | threshold | binding | pass |",
            "|---|---|---|---|---|---|---|",
        ]
        rows = sorted(gate.rows, key=lambda row: row.passed)
        for row in rows:
            lines.append(
                f"| {row.probe_id} | {row.tier} | {row.metric} |"
                f" {row.value:.4g} | {row.threshold:.4g} | {row.binding} |"
                f" {'PASS' if row.passed else 'FAIL'} |"
            )
        lines.append("")
        if gate.skipped:
            lines += [
                "Not gated (no row emitted - this is NOT a pass):",
                "",
                "| probe | tier | metric group | half period | reason |",
                "|---|---|---|---|---|",
            ]
            for item in gate.skipped:
                half = (
                    f"{item.half_period_ms:.4g} ms"
                    if item.half_period_ms is not None
                    else "n/a"
                )
                lines.append(
                    f"| {item.probe_id} | {item.tier} | {item.metric_group} |"
                    f" {half} | {item.reason} |"
                )
            lines.append("")
    return "\n".join(lines)
