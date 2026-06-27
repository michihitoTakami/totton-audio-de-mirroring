"""Evaluate Stage 1b transient de-ringing: transparent FIR only vs FIR + NBEE.

On square-wave probes, reports overshoot and plateau-ripple RMS for the
mirror-free transparent-FIR upsample (baseline) and the NBEE-processed output.
Lower is better; the NBEE should reduce ringing without breaking 0-20kHz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from totton_audio_de_mirroring.data.filters import (
    design_band_split_filters,
    upsample_transparent_reference,
)
from totton_audio_de_mirroring.data.pipeline_config import load_data_config
from totton_audio_de_mirroring.models.nbee import NeuralBandwidthExtension

SRC = 44_100
TARGET = 88_200


def _square(freq: float, seconds: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return np.asarray(np.sign(np.sin(2 * np.pi * freq * t)), dtype=np.float64)


def _ringing_metrics(signal: np.ndarray, edge_guard: int) -> tuple[float, float]:
    """Return (overshoot, plateau_ripple_rms) for a +/-1 square output."""
    overshoot = float(np.max(np.abs(signal)) - 1.0)
    # plateau = samples far from sign transitions
    sign = np.sign(signal)
    transitions = np.where(np.abs(np.diff(sign)) > 0)[0]
    mask = np.ones(signal.shape[0], dtype=bool)
    for idx in transitions:
        lo = max(0, idx - edge_guard)
        hi = min(signal.shape[0], idx + edge_guard)
        mask[lo:hi] = False
    plateau = signal[mask]
    target = np.sign(plateau)
    ripple = float(np.sqrt(np.mean((plateau - target) ** 2))) if plateau.size else 0.0
    return overshoot, ripple


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_data_config(args.data_config)
    lowpass, highpass = design_band_split_filters(
        cutoff_hz=cfg.band_split.cutoff_hz,
        sample_rate=TARGET,
        num_taps=cfg.band_split.num_taps,
        window=cfg.band_split.window,
    )
    model = NeuralBandwidthExtension(
        sample_rate=TARGET,
        cutoff_hz=cfg.band_split.cutoff_hz,
        energy_cap=args.energy_cap,
        envelope_floor=cfg.hb_target.envelope_min,
        lowpass_taps=lowpass,
        highpass_taps=highpass,
    )
    ck = torch.load(args.nbee_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()

    edge_guard = int(0.0005 * TARGET)  # 0.5 ms around each edge
    print(
        f"{'freq':>7} {'overshoot_fir':>14} {'overshoot_nbee':>15} "
        f"{'ripple_fir':>12} {'ripple_nbee':>12}"
    )
    for freq in (500.0, 1000.0, 5000.0):
        sq = _square(freq, 0.2, SRC)
        fir = upsample_transparent_reference(signal=sq, source_sr=SRC, target_sr=TARGET)
        with torch.no_grad():
            nbee = (
                model.forward(
                    torch.tensor(fir, dtype=torch.float32, device=device).unsqueeze(0)
                )[0]
                .cpu()
                .numpy()
            )
        guard = 4097  # skip filter edge transient
        o_f, r_f = _ringing_metrics(fir[guard:-guard], edge_guard)
        o_n, r_n = _ringing_metrics(nbee[guard:-guard], edge_guard)
        print(f"{freq:>7.0f} {o_f:>14.4f} {o_n:>15.4f} {r_f:>12.4f} {r_n:>12.4f}")
    print(
        "\nLower = better. NBEE should reduce overshoot/ripple vs FIR-only "
        "while keeping 0-20kHz intact (structural bypass)."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 1b de-ringing.")
    parser.add_argument("--nbee-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data_generation_antiring88k2.yaml"),
    )
    parser.add_argument("--energy-cap", type=float, default=2.0e-2)
    return parser.parse_args()


if __name__ == "__main__":
    main()
