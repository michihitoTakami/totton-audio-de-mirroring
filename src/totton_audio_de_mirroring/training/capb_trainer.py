"""Training loop for the CAPB upsampler.

The model has roughly 10^5 trainable parameters (controller only), uses fp32
throughout, and keeps the training loop intentionally compact.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import yaml  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, random_split

from totton_audio_de_mirroring.data.capb_dataset import (
    CAPBDataConfig,
    CAPBUpsampleDataset,
)
from totton_audio_de_mirroring.models.capb import (
    CAPB,
    SUPPORTED_CONTROLLER_FEATURE_MODES,
    SUPPORTED_FIR_COMPUTE_DTYPES,
    FIRComputeDType,
    RoutingPriorConfig,
    capb_candidate_from_checkpoint,
    capb_from_checkpoint,
)
from totton_audio_de_mirroring.models.proto_bank import (
    RELEASE_PROTOTYPE_PROFILE,
    build_prototype_bank_for_profile,
    supported_prototype_profiles,
)
from totton_audio_de_mirroring.torch_precision import configure_torch_precision
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
        allow_tf32: Permit reduced-mantissa CUDA TF32 execution.
        deterministic: Require deterministic Torch algorithms.
        prototype_profile: Fixed FIR prototype design profile.
        controller_dilation: Controller convolution dilation; two supplies
            sufficient two-sided context for the complete echo masks.
        controller_feature_mode: Versioned controller input features.
        routing_prior: Physics routing prior constants. None inherits the
            initial checkpoint's prior (or the legacy default for a fresh
            model); an explicit value overrides the checkpoint and is saved.
        fir_compute_dtype: Arithmetic dtype for the fixed FIR path.
        initial_head_scale: Multiplicative scale applied once to loaded
            controller-head weights before fine-tuning.
        initial_controller_only: Transfer only controller weights into the
            configured prototype profile.
        loss_weights: Composite loss weights.
        border_trim: Samples excluded at chunk borders in the losses.
        checkpoint_dir: Directory for checkpoints.
        initial_checkpoint: Optional controller checkpoint used for fine-tuning.
        checkpoint_interval_epochs: Save numbered checkpoints at this interval;
            zero disables numbered snapshots.
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
    allow_tf32: bool = False
    deterministic: bool = False
    prototype_profile: str = RELEASE_PROTOTYPE_PROFILE
    controller_dilation: int = 1
    controller_feature_mode: str = "waveform"
    routing_prior: RoutingPriorConfig | None = None
    fir_compute_dtype: FIRComputeDType = "float32"
    initial_head_scale: float = 1.0
    initial_controller_only: bool = False
    loss_weights: CAPBLossWeights = field(default_factory=CAPBLossWeights)
    border_trim: int = 512
    checkpoint_dir: Path = Path("data/checkpoints/capb")
    initial_checkpoint: Path | None = None
    checkpoint_interval_epochs: int = 0
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
        stationary_modulation=float(weights_raw.get("stationary_modulation", 0.0)),
        edge_fidelity_relax=float(weights_raw.get("edge_fidelity_relax", 0.9)),
        edge_ring=float(weights_raw.get("edge_ring", 300.0)),
        pre_echo_excess=float(weights_raw.get("pre_echo_excess", 0.0)),
        post_echo_excess=float(weights_raw.get("post_echo_excess", 0.0)),
        prototype_routing=float(weights_raw.get("prototype_routing", 0.0)),
        min_entropy=float(weights_raw.get("min_entropy", 0.05)),
    )
    checkpoint_interval = int(raw.get("checkpoint_interval_epochs", 0))
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval_epochs must be non-negative.")
    initial_head_scale = float(raw.get("initial_head_scale", 1.0))
    if not 0.0 < initial_head_scale <= 1.0:
        raise ValueError("initial_head_scale must satisfy 0 < scale <= 1.")
    prototype_profile = str(raw.get("prototype_profile", RELEASE_PROTOTYPE_PROFILE))
    if prototype_profile not in supported_prototype_profiles():
        raise ValueError(f"Unknown prototype_profile: {prototype_profile!r}.")
    fir_compute_dtype = str(raw.get("fir_compute_dtype", "float32"))
    if fir_compute_dtype not in SUPPORTED_FIR_COMPUTE_DTYPES:
        raise ValueError(f"Unknown fir_compute_dtype: {fir_compute_dtype!r}.")
    controller_dilation = int(raw.get("controller_dilation", 1))
    if controller_dilation <= 0:
        raise ValueError("controller_dilation must be positive.")
    controller_feature_mode = str(raw.get("controller_feature_mode", "waveform"))
    if controller_feature_mode not in SUPPORTED_CONTROLLER_FEATURE_MODES:
        raise ValueError(
            f"Unknown controller_feature_mode: {controller_feature_mode!r}."
        )
    routing_prior_raw = raw.get("routing_prior")
    routing_prior = (
        RoutingPriorConfig.from_mapping(routing_prior_raw)
        if routing_prior_raw is not None
        else None
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
        allow_tf32=bool(raw.get("allow_tf32", False)),
        deterministic=bool(raw.get("deterministic", False)),
        prototype_profile=prototype_profile,
        controller_dilation=controller_dilation,
        controller_feature_mode=controller_feature_mode,
        routing_prior=routing_prior,
        fir_compute_dtype=cast(FIRComputeDType, fir_compute_dtype),
        initial_head_scale=initial_head_scale,
        initial_controller_only=bool(raw.get("initial_controller_only", False)),
        loss_weights=loss_weights,
        border_trim=int(raw.get("border_trim", 512)),
        checkpoint_dir=Path(raw.get("checkpoint_dir", "data/checkpoints/capb")),
        initial_checkpoint=(
            Path(raw["initial_checkpoint"])
            if raw.get("initial_checkpoint") is not None
            else None
        ),
        checkpoint_interval_epochs=checkpoint_interval,
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
    device = torch.device(training_config.device)
    precision = configure_torch_precision(
        device,
        allow_tf32=training_config.allow_tf32,
        deterministic=training_config.deterministic,
    )
    torch.manual_seed(training_config.seed)

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

    if model is not None and training_config.initial_checkpoint is not None:
        raise ValueError("model and initial_checkpoint cannot both be provided.")
    if model is None and training_config.initial_checkpoint is not None:
        model = _load_initial_checkpoint(training_config, data_config)
    if model is None:
        bank = build_prototype_bank_for_profile(
            data_config.target_sample_rate,
            training_config.prototype_profile,
        )
        model = CAPB(
            bank=bank,
            fir_compute_dtype=training_config.fir_compute_dtype,
            controller_dilation=training_config.controller_dilation,
            controller_feature_mode=training_config.controller_feature_mode,
            routing_prior=training_config.routing_prior,
        )
    _scale_controller_head(model, training_config.initial_head_scale)
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
        interval = training_config.checkpoint_interval_epochs
        if interval > 0 and (epoch + 1) % interval == 0:
            epoch_path = checkpoint_dir / f"capb_epoch_{epoch + 1:03d}.pt"
            _save_checkpoint(model, training_config, data_config, record, epoch_path)
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            _save_checkpoint(model, training_config, data_config, record, best_path)

    return {
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "best_val_total": best_val,
        "history": history,
        "precision": precision.to_dict(),
    }


def _scale_controller_head(model: CAPB, scale: float) -> None:
    """Scale learned controller-head weights without changing fixed bias.

    Args:
        model: CAPB model whose controller will be fine-tuned.
        scale: Multiplicative weight scale in ``(0, 1]``.

    Raises:
        ValueError: If scale is outside the supported interval.

    Physical Basis:
        A saturated softmax has vanishing waveform gradients. Scaling only
        the learned head weights preserves the fixed prior and routing order
        while restoring a reproducible gradient margin for fine-tuning.
    """
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must satisfy 0 < scale <= 1.")
    if scale == 1.0:
        return
    with torch.no_grad():
        model.controller.head.weight.mul_(scale)


def _load_initial_checkpoint(
    training_config: CAPBTrainingConfig,
    data_config: CAPBDataConfig,
) -> CAPB:
    """Load and rate-check a controller for fine-tuning.

    Args:
        training_config: Training settings containing the checkpoint path.
        data_config: Dataset rates that the checkpoint must match.

    Returns:
        Rate-correct CAPB initialized from the checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If checkpoint and dataset rates differ.

    Physical Basis:
        Fine-tuning preserves the validated transient response learned by the
        baseline while the stationary loss removes signal-synchronous weight
        modulation. Rate validation prevents pairing a controller with the
        wrong fixed prototype bank.
    """
    path = training_config.initial_checkpoint
    if path is None:
        raise ValueError("initial_checkpoint is required.")
    if not path.is_file():
        raise FileNotFoundError(f"Initial checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = (
        capb_candidate_from_checkpoint(
            checkpoint,
            prototype_profile=training_config.prototype_profile,
            fir_compute_dtype=training_config.fir_compute_dtype,
            controller_dilation=training_config.controller_dilation,
            controller_feature_mode=training_config.controller_feature_mode,
        )
        if training_config.initial_controller_only
        else capb_from_checkpoint(checkpoint)
    )
    if model.prototype_profile != training_config.prototype_profile:
        raise ValueError(
            "Initial checkpoint prototype profile "
            f"{model.prototype_profile!r} does not match training profile "
            f"{training_config.prototype_profile!r}."
        )
    if model.fir_compute_dtype != training_config.fir_compute_dtype:
        raise ValueError(
            "Initial checkpoint FIR dtype "
            f"{model.fir_compute_dtype!r} does not match training dtype "
            f"{training_config.fir_compute_dtype!r}."
        )
    if model.controller_dilation != training_config.controller_dilation:
        raise ValueError(
            "Initial checkpoint controller dilation "
            f"{model.controller_dilation} does not match training dilation "
            f"{training_config.controller_dilation}."
        )
    if model.controller_feature_mode != training_config.controller_feature_mode:
        raise ValueError(
            "Initial checkpoint controller feature mode "
            f"{model.controller_feature_mode!r} does not match training mode "
            f"{training_config.controller_feature_mode!r}."
        )
    target_sample_rate = int(checkpoint.get("target_sample_rate", 88_200))
    if target_sample_rate != data_config.target_sample_rate:
        raise ValueError(
            "Initial checkpoint target rate "
            f"{target_sample_rate} Hz does not match dataset target rate "
            f"{data_config.target_sample_rate} Hz."
        )
    expected_input_rate = checkpoint.get("expected_input_rate")
    if (
        expected_input_rate is not None
        and int(expected_input_rate) != data_config.source_sample_rate
    ):
        raise ValueError(
            "Initial checkpoint input rate "
            f"{expected_input_rate} Hz does not match dataset source rate "
            f"{data_config.source_sample_rate} Hz."
        )
    configured_prior = training_config.routing_prior
    if configured_prior is not None and configured_prior != model.routing_prior:
        # The prior holds routing constants only (no tensors), so a
        # continuation may change policy without invalidating the controller.
        logger.info(
            "Overriding checkpoint routing prior %s with %s",
            model.routing_prior.to_dict(),
            configured_prior.to_dict(),
        )
        model.routing_prior = configured_prior
    return model


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
        pre_echo_mask = batch["pre_echo_mask"].to(device)
        post_echo_mask = batch["post_echo_mask"].to(device)
        far_pre_echo_mask = batch["far_pre_echo_mask"].to(device)
        safe_active_mask = batch["safe_active_mask"].to(device)
        stationary = batch["stationary"].to(device)
        focused_transient = batch["focused_event"].to(device)

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
                prototype_outputs=prototypes,
                stationary=stationary,
                pre_echo_mask=pre_echo_mask,
                post_echo_mask=post_echo_mask,
                far_pre_echo_mask=far_pre_echo_mask,
                safe_active_mask=safe_active_mask,
                focused_transient=focused_transient,
                sharp_index=model.prototype_names.index("sharp"),
                gentle_index=model.prototype_names.index("gentle"),
                focused_gentle_fraction=model.focused_gentle_fraction_frames(
                    source, weights.shape[-1]
                ),
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
            "prototype_profile": model.prototype_profile,
            "prototype_hash": model.prototype_hash,
            "fir_compute_dtype": model.fir_compute_dtype,
            "controller_dilation": model.controller_dilation,
            "controller_feature_mode": model.controller_feature_mode,
            "routing_prior": model.routing_prior.to_dict(),
            "expected_input_rate": data_config.source_sample_rate,
            "target_sample_rate": data_config.target_sample_rate,
            "training_config": {
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "loss_weights": vars(config.loss_weights),
                "initial_checkpoint": (
                    str(config.initial_checkpoint)
                    if config.initial_checkpoint is not None
                    else None
                ),
                "checkpoint_interval_epochs": config.checkpoint_interval_epochs,
                "allow_tf32": config.allow_tf32,
                "deterministic": config.deterministic,
                "prototype_profile": config.prototype_profile,
                "fir_compute_dtype": config.fir_compute_dtype,
                "controller_dilation": config.controller_dilation,
                "controller_feature_mode": config.controller_feature_mode,
                "routing_prior": model.routing_prior.to_dict(),
                "initial_head_scale": config.initial_head_scale,
                "initial_controller_only": config.initial_controller_only,
            },
            "data_config": asdict(data_config),
            "record": record,
        },
        path,
    )
