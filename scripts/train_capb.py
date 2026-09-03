"""Train the CAPB Stage 1 upsampler.

Usage:
    uv run python scripts/train_capb.py \
        --data-config configs/data_generation_capb.yaml \
        --config configs/training_stage1_capb.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

from totton_audio_de_mirroring.data.capb_dataset import (
    CAPBDataConfig,
    load_capb_data_config,
)
from totton_audio_de_mirroring.models.capb import (
    SUPPORTED_CONTROLLER_FEATURE_MODES,
    SUPPORTED_FIR_COMPUTE_DTYPES,
    RoutingPriorConfig,
)
from totton_audio_de_mirroring.models.proto_bank import supported_prototype_profiles
from totton_audio_de_mirroring.training.capb_trainer import (
    CAPBTrainingConfig,
    load_capb_training_config,
    train_capb,
)


def main() -> None:
    """Run CAPB training from YAML configurations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--prototype-profile",
        choices=supported_prototype_profiles(),
        default=None,
    )
    parser.add_argument(
        "--fir-compute-dtype",
        choices=SUPPORTED_FIR_COMPUTE_DTYPES,
        default=None,
    )
    parser.add_argument("--initial-controller-only", action="store_true")
    parser.add_argument("--controller-dilation", type=int, default=None)
    parser.add_argument(
        "--controller-feature-mode",
        choices=SUPPORTED_CONTROLLER_FEATURE_MODES,
        default=None,
    )
    parser.add_argument("--border-trim", type=int, default=None)
    parser.add_argument("--far-pre-echo-window-ms", type=float, default=None)
    parser.add_argument("--pre-echo-excess", type=float, default=None)
    parser.add_argument("--post-echo-excess", type=float, default=None)
    parser.add_argument("--edge-ring", type=float, default=None)
    parser.add_argument("--prototype-routing", type=float, default=None)
    parser.add_argument("--stationary-modulation", type=float, default=None)
    parser.add_argument("--focused-gentle-fraction", type=float, default=None)
    parser.add_argument("--level-change-threshold", type=float, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    data_config = load_capb_data_config(args.data_config)
    training_config = load_capb_training_config(args.config)
    data_config, training_config = _override_seed(
        data_config, training_config, args.seed
    )
    if args.num_samples is not None:
        data_config = replace(data_config, num_samples=args.num_samples)
    if args.far_pre_echo_window_ms is not None:
        if args.far_pre_echo_window_ms < 0.0:
            raise ValueError("--far-pre-echo-window-ms must be non-negative.")
        far_guard_ms = data_config.transient_supervision.far_pre_echo_guard_ms
        transient = replace(
            data_config.transient_supervision,
            context_ms=max(
                data_config.transient_supervision.context_ms,
                far_guard_ms + args.far_pre_echo_window_ms,
            ),
            far_pre_echo_window_ms=args.far_pre_echo_window_ms,
        )
        data_config = replace(data_config, transient_supervision=transient)
    overrides: dict[str, object] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.learning_rate is not None:
        if args.learning_rate <= 0.0:
            raise ValueError("--learning-rate must be positive.")
        overrides["learning_rate"] = args.learning_rate
    if args.device is not None:
        overrides["device"] = args.device
    if args.allow_tf32:
        overrides["allow_tf32"] = True
    if args.checkpoint_dir is not None:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.init_checkpoint is not None:
        overrides["initial_checkpoint"] = args.init_checkpoint
    if args.prototype_profile is not None:
        overrides["prototype_profile"] = args.prototype_profile
    if args.fir_compute_dtype is not None:
        overrides["fir_compute_dtype"] = args.fir_compute_dtype
    if args.initial_controller_only:
        overrides["initial_controller_only"] = True
    if args.controller_dilation is not None:
        if args.controller_dilation <= 0:
            raise ValueError("--controller-dilation must be positive.")
        overrides["controller_dilation"] = args.controller_dilation
    if args.controller_feature_mode is not None:
        overrides["controller_feature_mode"] = args.controller_feature_mode
    if args.border_trim is not None:
        if args.border_trim < 0:
            raise ValueError("--border-trim must be non-negative.")
        overrides["border_trim"] = args.border_trim
    if args.checkpoint_interval is not None:
        if args.checkpoint_interval < 0:
            raise ValueError("--checkpoint-interval must be non-negative.")
        overrides["checkpoint_interval_epochs"] = args.checkpoint_interval
    loss_overrides: dict[str, float] = {}
    if args.pre_echo_excess is not None:
        loss_overrides["pre_echo_excess"] = args.pre_echo_excess
    if args.post_echo_excess is not None:
        loss_overrides["post_echo_excess"] = args.post_echo_excess
    if args.edge_ring is not None:
        loss_overrides["edge_ring"] = args.edge_ring
    if args.prototype_routing is not None:
        loss_overrides["prototype_routing"] = args.prototype_routing
    if args.stationary_modulation is not None:
        loss_overrides["stationary_modulation"] = args.stationary_modulation
    if any(value < 0.0 for value in loss_overrides.values()):
        raise ValueError("Loss-weight overrides must be non-negative.")
    if loss_overrides:
        overrides["loss_weights"] = replace(
            training_config.loss_weights, **loss_overrides
        )
    prior_overrides: dict[str, float] = {}
    if args.focused_gentle_fraction is not None:
        prior_overrides["focused_gentle_fraction"] = args.focused_gentle_fraction
    if args.level_change_threshold is not None:
        prior_overrides["level_change_threshold"] = args.level_change_threshold
    if prior_overrides:
        base_prior = training_config.routing_prior or RoutingPriorConfig()
        overrides["routing_prior"] = replace(base_prior, **prior_overrides)
    if overrides:
        training_config = replace(training_config, **overrides)

    summary = train_capb(data_config, training_config)
    payload = {
        "best_checkpoint": str(summary["best_checkpoint"]),
        "last_checkpoint": str(summary["last_checkpoint"]),
        "best_val_total": summary["best_val_total"],
        "initial_checkpoint": (
            str(training_config.initial_checkpoint)
            if training_config.initial_checkpoint is not None
            else None
        ),
        "history": summary["history"],
        "precision": summary["precision"],
        "data_config": _jsonable(asdict(data_config)),
        "training_config": _jsonable(asdict(training_config)),
        "config_sha256": {
            "data": _sha256(args.data_config),
            "training": _sha256(args.config),
        },
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("best_checkpoint", "best_val_total")}))


def _override_seed(
    data_config: CAPBDataConfig,
    training_config: CAPBTrainingConfig,
    seed: int | None,
) -> tuple[CAPBDataConfig, CAPBTrainingConfig]:
    """Apply one reproducible seed to data generation and controller training."""
    if seed is None:
        return data_config, training_config
    if not 0 <= seed < 2**32:
        raise ValueError("--seed must be in [0, 2**32).")
    return replace(data_config, seed=seed), replace(training_config, seed=seed)


def _sha256(path: Path) -> str:
    """Return a configuration-file SHA-256 digest."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError(f"Failed to hash configuration {path}: {error}") from error


def _jsonable(value: object) -> object:
    """Convert dataclass payload values to JSON-compatible structures."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
