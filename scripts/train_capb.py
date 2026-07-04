"""Train the CAPB Stage 1 upsampler.

Usage:
    uv run python scripts/train_capb.py \
        --data-config configs/data_generation_capb.yaml \
        --config configs/training_stage1_capb.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from totton_audio_de_mirroring.data.capb_dataset import load_capb_data_config
from totton_audio_de_mirroring.training.capb_trainer import (
    load_capb_training_config,
    train_capb,
)


def main() -> None:
    """Run CAPB training from YAML configurations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    data_config = load_capb_data_config(args.data_config)
    training_config = load_capb_training_config(args.config)

    if args.num_samples is not None:
        data_config = replace(data_config, num_samples=args.num_samples)
    overrides: dict[str, object] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.device is not None:
        overrides["device"] = args.device
    if args.checkpoint_dir is not None:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    if overrides:
        training_config = replace(training_config, **overrides)

    summary = train_capb(data_config, training_config)
    payload = {
        "best_checkpoint": str(summary["best_checkpoint"]),
        "last_checkpoint": str(summary["last_checkpoint"]),
        "best_val_total": summary["best_val_total"],
        "history": summary["history"],
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("best_checkpoint", "best_val_total")}))


if __name__ == "__main__":
    main()
