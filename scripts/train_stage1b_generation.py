"""Train the Stage 1b Neural Bandwidth Extension Engine (HB generation).

Opt-in high-band GENERATION (not suppression). Trains on the synthetic raw_88k2
native teacher (abundant, generative target) and, optionally, a real hi-res
corpus mixed in for realism. The 0-20kHz band stays a structural bypass and the
shared envelope/energy-cap safety constraints bound the generated high band.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader

from totton_audio_de_mirroring.data.dataloader import collate_samples
from totton_audio_de_mirroring.data.dataset import MirrorSuppressionDataset
from totton_audio_de_mirroring.data.filters import design_band_split_filters
from totton_audio_de_mirroring.data.pipeline_config import load_data_config
from totton_audio_de_mirroring.models.nbee import NBEEConfig, NeuralBandwidthExtension
from totton_audio_de_mirroring.training.losses import (
    LossWeights,
    RingingLossConfig,
    STFTLossConfig,
    compute_losses,
)


def main() -> None:
    """Train the NBEE generative model and save a labelled checkpoint."""
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_data_config(args.data_config)
    cfg = replace(
        cfg,
        num_samples=args.num_samples,
        seed=args.seed,
        hb_target=replace(cfg.hb_target, energy_cap=args.energy_cap),
    )

    model = _build_model(cfg, args.energy_cap, device)
    loader = _build_loader(cfg, args)
    print(
        f"NBEE params={sum(p.numel() for p in model.parameters())} "
        f"samples/epoch={len(loader.dataset)} device={device}",  # type: ignore[arg-type]
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights = LossWeights(
        mask=1.0,
        stft=1.0,
        preserve=1.0,
        energy=1.0,
        subtract=0.0,
        cap_strict=4.0,
        edge=0.05,
        step=0.05,
    )
    mask_cfg = STFTLossConfig(n_fft=2048, hop_length=512, win_length=2048)
    stft_cfgs = (
        STFTLossConfig(n_fft=1024, hop_length=256, win_length=1024),
        STFTLossConfig(n_fft=2048, hop_length=512, win_length=2048),
    )
    ring = RingingLossConfig()

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        steps = 0
        for batch in loader:
            x_full = batch["x_full"].to(device)
            hb_in = batch["high_band"].to(device)
            hb_target = batch["hb_target"].to(device)
            mirror_mask = batch["mirror_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                hb_pred = model.generate_highband(x_full)
            terms = compute_losses(
                hb_in=hb_in,
                hb_target=hb_target,
                hb_pred=hb_pred,
                mirror_mask=mirror_mask,
                mask_config=mask_cfg,
                stft_configs=stft_cfgs,
                weights=weights,
                energy_cap=args.energy_cap,
                ringing_config=ring,
                mask_mode="l1",
            )
            scaler.scale(terms.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(terms.total.detach().item())
            steps += 1
        print(
            f"epoch={epoch + 1}/{args.epochs} train_total={total / max(steps, 1):.6f}",
            flush=True,
        )

    _save(model, args, cfg)


def _build_model(
    cfg: Any, energy_cap: float, device: torch.device
) -> NeuralBandwidthExtension:
    lowpass, highpass = design_band_split_filters(
        cutoff_hz=cfg.band_split.cutoff_hz,
        sample_rate=cfg.band_split.sample_rate,
        num_taps=cfg.band_split.num_taps,
        window=cfg.band_split.window,
    )
    model = NeuralBandwidthExtension(
        sample_rate=cfg.target_sample_rate,
        cutoff_hz=cfg.band_split.cutoff_hz,
        energy_cap=energy_cap,
        envelope_floor=cfg.hb_target.envelope_min,
        lowpass_taps=lowpass,
        highpass_taps=highpass,
        model_config=NBEEConfig(),
    )
    return model.to(device)


def _build_loader(cfg: Any, args: argparse.Namespace) -> DataLoader[dict[str, Any]]:
    synth = MirrorSuppressionDataset(cfg, target_mode="generate")
    datasets: list[Any] = [synth]
    if args.hires_root is not None:
        from totton_audio_de_mirroring.data.hires_corpus import HiResCorpusConfig
        from totton_audio_de_mirroring.data.hires_dataset import HiResTeacherDataset

        hires = HiResTeacherDataset(
            replace(cfg, num_samples=max(1, args.num_samples // 2)),
            HiResCorpusConfig(
                root=args.hires_root, min_sample_rate=cfg.target_sample_rate
            ),
            target_mode="generate",
        )
        datasets.append(hires)
    dataset: Any = synth if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_samples,
        persistent_workers=args.num_workers > 0,
    )


def _save(model: NeuralBandwidthExtension, args: argparse.Namespace, cfg: Any) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "training_config": {
            "teacher_type": cfg.teacher_type,
            "energy_cap": args.energy_cap,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "hires_root": str(args.hires_root) if args.hires_root else None,
            "target_mode": "generate",
        },
        "model_config": model.model_config.to_checkpoint_dict(),
        "epoch": args.epochs,
        "device": str(next(model.parameters()).device),
    }
    torch.save(state, args.output)
    print(f"saved NBEE -> {args.output}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1b NBEE (HB generation).")
    parser.add_argument(
        "--data-config", type=Path, default=Path("configs/data_generation_gen88k2.yaml")
    )
    parser.add_argument("--hires-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--energy-cap", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output", type=Path, default=Path("data/checkpoints/stage1b/stage1b_nbee.pt")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
