"""Training loop for the CAPB upsampler.

The model has roughly 10^5 trainable parameters (controller only), uses fp32
throughout, and keeps the training loop intentionally compact.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, random_split

from totton_audio_de_mirroring.data.capb_dataset import (
    CAPBDataConfig,
    CAPBUpsampleDataset,
)
from totton_audio_de_mirroring.models.capb import CAPB
from totton_audio_de_mirroring.models.proto_bank import (
    build_prototype_bank,
    prototype_specs_for_target_rate,
)
from totton_audio_de_mirroring.training.capb_losses import (
    CAPBLossWeights,
    compute_capb_losses,
)
from totton_audio_de_mirroring.training.stft_loss import STFTLossConfig

logger = logging.getLogger(__name__)

DEFAULT_STFT_CONFIGS = (
    STFTLossConfig(n_fft=1024, hop_length=256, win_length=1024),
    STFTLossConfig(n_fft=2048, hop_length=512, win_length=2048),
)


@dataclass(frozen=True)
class CAPBTrainingConfig:
    """Configuration for CAPB training.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Gradient-norm clip value.
        val_fraction: Fraction of the dataset used for validation.
        num_workers: DataLoader worker count.
        device: Torch device string.
        seed: Torch RNG seed.
        loss_weights: Composite loss weights.
        border_trim: Samples excluded at chunk borders in the losses.
        checkpoint_dir: Directory for checkpoints.
        log_interval: Steps between train-loss log lines.

    Physical Basis:
        Only the controller trains; the DSP prototype bank is frozen, so a
        small learning problem with strong losses converges in few epochs.
    """

    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1234
    loss_weights: CAPBLossWeights = field(default_factory=CAPBLossWeights)
    border_trim: int = 512
    checkpoint_dir: Path = Path("data/checkpoints/capb")
    log_interval: int = 20


def load_capb_training_config(path: Path) -> CAPBTrainingConfig:
    """Load a CAPB training configuration from YAML.

    Args:
        path: YAML file path.

    Returns:
        Parsed configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    weights_raw = raw.get("loss_weights", {})
    loss_weights = CAPBLossWeights(
        wave=float(weights_raw.get("wave", 1.0)),
        stft=float(weights_raw.get("stft", 1.0)),
        plateau=float(weights_raw.get("plateau", 100.0)),
        quiet=float(weights_raw.get("quiet", 100.0)),
        tv=float(weights_raw.get("tv", 0.1)),
        entropy_floor=float(weights_raw.get("entropy_floor", 10.0)),
        edge_fidelity_relax=float(weights_raw.get("edge_fidelity_relax", 0.9)),
        edge_ring=float(weights_raw.get("edge_ring", 300.0)),
        min_entropy=float(weights_raw.get("min_entropy", 0.05)),
    )
    return CAPBTrainingConfig(
        epochs=int(raw.get("epochs", 50)),
        batch_size=int(raw.get("batch_size", 16)),
        learning_rate=float(raw.get("learning_rate", 1.0e-4)),
        weight_decay=float(raw.get("weight_decay", 0.0)),
        grad_clip=float(raw.get("grad_clip", 1.0)),
        val_fraction=float(raw.get("val_fraction", 0.1)),
        num_workers=int(raw.get("num_workers", 4)),
        device=str(raw.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
        seed=int(raw.get("seed", 1234)),
        loss_weights=loss_weights,
        border_trim=int(raw.get("border_trim", 512)),
        checkpoint_dir=Path(raw.get("checkpoint_dir", "data/checkpoints/capb")),
        log_interval=int(raw.get("log_interval", 20)),
    )


def train_capb(
    data_config: CAPBDataConfig,
    training_config: CAPBTrainingConfig,
    model: CAPB | None = None,
) -> dict[str, Any]:
    """Train a CAPB model.

    Args:
        data_config: Dataset configuration.
        training_config: Training configuration.
        model: Optional pre-built model (a fresh one is built otherwise).

    Returns:
        Summary dictionary with best/last checkpoint paths and history.

    Raises:
        ValueError: If configurations are inconsistent.

    Physical Basis:
        The controller starts biased to the sharp prototype (epoch 0 is
        already the FIR baseline that passes the fidelity gates); training
        moves weights toward gentle only where the ringing losses demand.
    """
    torch.manual_seed(training_config.seed)
    device = torch.device(training_config.device)

    dataset = CAPBUpsampleDataset(data_config)
    val_len = max(1, int(len(dataset) * training_config.val_fraction))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(training_config.seed),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    if model is None:
        bank = build_prototype_bank(
            prototype_specs_for_target_rate(data_config.target_sample_rate),
            sample_rate=data_config.target_sample_rate,
        )
        model = CAPB(bank=bank)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.controller.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    stft_configs = list(DEFAULT_STFT_CONFIGS)
    checkpoint_dir = training_config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_path = checkpoint_dir / "capb_best.pt"
    last_path = checkpoint_dir / "capb_last.pt"

    for epoch in range(training_config.epochs):
        start = time.monotonic()
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            stft_configs,
            training_config,
            train=True,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            stft_configs,
            training_config,
            train=False,
        )
        record = {
            "epoch": float(epoch),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        logger.info(
            "epoch %d train_total=%.5f val_total=%.5f weights=%s (%.1fs)",
            epoch,
            train_metrics["total"],
            val_metrics["total"],
            [round(train_metrics[f"w_{i}"], 3) for i in range(model.num_prototypes)],
            time.monotonic() - start,
        )

        _save_checkpoint(model, training_config, data_config, record, last_path)
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            _save_checkpoint(model, training_config, data_config, record, best_path)

    return {
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "best_val_total": best_val,
        "history": history,
    }


def _run_epoch(
    model: CAPB,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stft_configs: list[STFTLossConfig],
    config: CAPBTrainingConfig,
    train: bool,
) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    weight_sums = torch.zeros(model.num_prototypes)
    steps = 0

    for step, batch in enumerate(loader):
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        flat_mask = batch["flat_mask"].to(device)
        quiet_mask = batch["quiet_mask"].to(device)
        edge_mask = batch["edge_mask"].to(device)

        with torch.set_grad_enabled(train):
            output, weights, prototypes = model.forward_with_details(source)
            gentle_output = prototypes[
                :, model.prototype_names.index("gentle")
            ].detach()
            losses = compute_capb_losses(
                output=output,
                target=target,
                weights_frames=weights,
                flat_mask=flat_mask,
                quiet_mask=quiet_mask,
                stft_configs=stft_configs,
                loss_weights=config.loss_weights,
                trim=config.border_trim,
                edge_mask=edge_mask,
                gentle_output=gentle_output,
            )

        if train:
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.controller.parameters(), config.grad_clip
            )
            optimizer.step()
            if step % config.log_interval == 0:
                logger.debug(
                    "step %d total=%.5f", step, float(losses["total"].detach())
                )

        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        weight_sums += weights.detach().mean(dim=(0, 2)).cpu()
        steps += 1

    if steps == 0:
        raise ValueError("Loader produced no batches.")
    metrics = {key: value / steps for key, value in totals.items()}
    for index in range(model.num_prototypes):
        metrics[f"w_{index}"] = float(weight_sums[index] / steps)
    return metrics


def _save_checkpoint(
    model: CAPB,
    config: CAPBTrainingConfig,
    data_config: CAPBDataConfig,
    record: dict[str, float],
    path: Path,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "prototype_names": list(model.prototype_names),
            "expected_input_rate": data_config.source_sample_rate,
            "target_sample_rate": data_config.target_sample_rate,
            "training_config": {
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "loss_weights": vars(config.loss_weights),
            },
            "record": record,
        },
        path,
    )
