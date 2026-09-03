"""Evaluate hard synthetic sharp/gentle CAPB routing gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from totton_audio_de_mirroring.data.capb_dataset import (
    CAPBDataConfig,
    CAPBUpsampleDataset,
    load_capb_data_config,
)
from totton_audio_de_mirroring.models.capb import CAPB, capb_from_checkpoint

_STATIONARY_SAFE_TYPES = (
    "flowing_noise",
    "multitone",
    "pink_noise",
    "band_limited_noise",
    "near_nyquist_noise",
)
_GENTLE_ONLY_FRACTION = 0.99
_SAFE_TYPES = (*_STATIONARY_SAFE_TYPES, "damped_string")
_FOCUSED_TYPES = (
    "isolated_click",
    "tone_burst",
    "damped_string",
    "clustered_impacts",
)
_EDGE_TYPES = ("square_wave", "step_plateau", "string_riff", "impact_stream")


def parse_args() -> argparse.Namespace:
    """Parse strict routing-gate arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples-per-type", type=int, default=16)
    parser.add_argument("--stationary-sharp-min", type=float, default=0.995)
    parser.add_argument("--transient-gentle-min", type=float, default=0.90)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--border-trim", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    """Run canonical and held-out seed blocks and fail on the worst sample."""
    args = parse_args()
    if args.samples_per_type <= 0:
        raise ValueError("--samples-per-type must be positive.")
    if args.border_trim < 0:
        raise ValueError("--border-trim must be non-negative.")
    thresholds = (args.stationary_sharp_min, args.transient_gentle_min)
    if any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("Routing thresholds must lie in (0, 1).")
    model = _load_model(args.checkpoint, args.device)
    data_config = load_capb_data_config(args.data_config)
    expected_rate = _expected_input_rate(args.checkpoint)
    if data_config.source_sample_rate != expected_rate:
        raise ValueError("Checkpoint and data-config rate families do not match.")
    blocks = {
        "canonical": _evaluate_block(
            model,
            data_config,
            args.samples_per_type,
            seed=17,
            border_trim=args.border_trim,
        ),
        "held_out": _evaluate_block(
            model,
            data_config,
            args.samples_per_type,
            seed=1_000_003,
            border_trim=args.border_trim,
        ),
    }
    rows = _gate_rows(blocks, args.stationary_sharp_min, args.transient_gentle_min)
    payload = {
        "all_passed": all(bool(row["passed"]) for row in rows),
        "routing_prior": model.routing_prior.to_dict(),
        "stationary_sharp_min": args.stationary_sharp_min,
        "transient_gentle_min": args.transient_gentle_min,
        "samples_per_type": args.samples_per_type,
        "rows": rows,
        "blocks": blocks,
    }
    _write_json(args.report, payload)
    print(json.dumps({"all_passed": payload["all_passed"], "rows": rows}))
    if not payload["all_passed"]:
        raise SystemExit(1)


def _evaluate_block(
    model: CAPB,
    base_config: CAPBDataConfig,
    samples_per_type: int,
    seed: int,
    border_trim: int,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    signal_types = dict.fromkeys((*_SAFE_TYPES, *_FOCUSED_TYPES, *_EDGE_TYPES))
    for signal_type in signal_types:
        config = replace(
            base_config,
            num_samples=samples_per_type,
            seed=seed,
            signal_mix={signal_type: 1.0},
        )
        dataset = CAPBUpsampleDataset(config)
        per_mask: dict[str, list[float]] = {
            "safe_active": [],
            "pre_echo": [],
            "post_echo": [],
            "edge": [],
        }
        for index in range(samples_per_type):
            sample = dataset[index]
            weights = _controller_weights(model, sample["source"])
            for mask_name, output_name in (
                ("safe_active_mask", "safe_active"),
                ("pre_echo_mask", "pre_echo"),
                ("post_echo_mask", "post_echo"),
                ("edge_mask", "edge"),
            ):
                prototypes = _target_prototypes(model, signal_type, output_name)
                prototype_indices = [
                    model.prototype_names.index(prototype) for prototype in prototypes
                ]
                exclude = sample["edge_mask"] if output_name == "safe_active" else None
                value = _masked_weight(
                    np.sum(weights[prototype_indices], axis=0),
                    sample[mask_name],
                    exclude,
                    border_trim,
                )
                if np.isfinite(value):
                    per_mask[output_name].append(value)
        result[signal_type] = {
            name: float(min(values)) if values else float("nan")
            for name, values in per_mask.items()
        }
    return result


def _target_prototypes(
    model: CAPB, signal_type: str, output_name: str
) -> tuple[str, ...]:
    """Return physically protective prototypes for a routing label.

    Physical Basis:
        Three-prototype CAPB may use middle only for sparse focused impulses,
        where it preserves transient gain while meeting echo limits. When the
        checkpoint's routing prior sends the impulse mass to gentle, middle
        no longer counts as protective. Sustained plateau edges use gentle;
        active non-risk content uses sharp.
    """
    if output_name == "safe_active":
        return ("sharp",)
    has_middle = "mid" in model.prototype_names
    gentle_only = model.routing_prior.focused_gentle_fraction >= _GENTLE_ONLY_FRACTION
    focused = signal_type in _FOCUSED_TYPES
    if (
        has_middle
        and not gentle_only
        and (output_name in {"pre_echo", "post_echo"} or focused)
    ):
        return ("mid", "gentle")
    return ("gentle",)


def _controller_weights(model: CAPB, source: torch.Tensor) -> np.ndarray:
    device = next(model.parameters()).device
    tensor = source.to(device).unsqueeze(0)
    with torch.no_grad():
        weights = model.controller_weights(tensor)
    return np.asarray(weights.squeeze(0).cpu(), dtype=np.float64)


def _masked_weight(
    weight: np.ndarray,
    target_mask: torch.Tensor,
    exclude_mask: torch.Tensor | None = None,
    border_trim: int = 0,
) -> float:
    """Return mean controller weight over target-rate labelled frames."""
    if weight.ndim != 1 or weight.size == 0:
        raise ValueError("weight must be a non-empty vector.")
    if target_mask.dim() != 1 or target_mask.numel() == 0:
        raise ValueError("target_mask must be a non-empty vector.")
    if border_trim < 0 or 2 * border_trim >= target_mask.numel():
        raise ValueError("border_trim must leave a non-empty mask interior.")
    clean_mask = target_mask.to(torch.float32).clone()
    if border_trim:
        clean_mask[:border_trim] = 0.0
        clean_mask[-border_trim:] = 0.0
    mask = F.adaptive_max_pool1d(clean_mask.reshape(1, 1, -1), weight.size)
    frame_mask = np.asarray(mask.reshape(-1), dtype=np.float64) > 0.5
    if exclude_mask is not None:
        if exclude_mask.shape != target_mask.shape:
            raise ValueError("exclude_mask must share target_mask shape.")
        clean_exclude = exclude_mask.to(torch.float32).clone()
        if border_trim:
            clean_exclude[:border_trim] = 0.0
            clean_exclude[-border_trim:] = 0.0
        excluded = F.adaptive_max_pool1d(clean_exclude.reshape(1, 1, -1), weight.size)
        frame_mask &= np.asarray(excluded.reshape(-1), dtype=np.float64) <= 0.5
    return float(np.mean(weight[frame_mask])) if np.any(frame_mask) else float("nan")


def _gate_rows(
    blocks: dict[str, dict[str, dict[str, float]]],
    stationary_threshold: float,
    transient_threshold: float,
) -> list[dict[str, Any]]:
    specifications = {
        "R1_stationary_sharp": (
            [(signal_type, "safe_active") for signal_type in _STATIONARY_SAFE_TYPES],
            stationary_threshold,
        ),
        "R2_pre_echo_protective": (
            [(signal_type, "pre_echo") for signal_type in _FOCUSED_TYPES],
            transient_threshold,
        ),
        "R3_post_echo_protective": (
            [(signal_type, "post_echo") for signal_type in _FOCUSED_TYPES],
            transient_threshold,
        ),
        "R4_abrupt_edge_protective": (
            [("square_wave", "edge")]
            + [
                (signal_type, "edge")
                for signal_type in ("string_riff", "impact_stream")
            ]
            + [(signal_type, "edge") for signal_type in _FOCUSED_TYPES],
            transient_threshold,
        ),
    }
    rows: list[dict[str, Any]] = []
    for gate_id, (metrics, threshold) in specifications.items():
        candidates = [
            (tier, signal_type, blocks[tier][signal_type][metric])
            for tier in blocks
            for signal_type, metric in metrics
        ]
        finite = [item for item in candidates if np.isfinite(item[2])]
        tier, signal_type, value = min(finite, key=lambda item: item[2])
        rows.append(
            {
                "gate_id": gate_id,
                "passed": bool(value >= threshold),
                "worst_tier": tier,
                "worst_signal_type": signal_type,
                "value": value,
                "threshold": threshold,
            }
        )
    return rows


def _load_model(path: Path, device: str) -> CAPB:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Failed to load checkpoint {path}: {error}") from error
    return capb_from_checkpoint(state).to(torch.device(device)).eval()


def _expected_input_rate(path: Path) -> int:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Failed to inspect checkpoint {path}: {error}") from error
    return int(
        state.get(
            "expected_input_rate", int(state.get("target_sample_rate", 88_200)) // 2
        )
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=True))
    except OSError as error:
        raise RuntimeError(
            f"Failed to write routing gate report {path}: {error}"
        ) from error


if __name__ == "__main__":
    main()
