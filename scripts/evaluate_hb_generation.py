"""Evaluate Stage 1b HB generation against the suppression baseline.

Measures, on real hi-res clips, the high-band (20-44kHz) log-magnitude L1 error
versus the TRUE native high band for: the degraded input, the suppression NMSE
baseline, and the NBEE generator. Also reports the NBEE high-band energy-cap
compliance and 0-20kHz (low-band) preservation.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from totton_audio_de_mirroring.data.degradation import upsample_bessel_reference
from totton_audio_de_mirroring.data.filters import design_band_split_filters
from totton_audio_de_mirroring.data.hires_corpus import resample_signal
from totton_audio_de_mirroring.data.pipeline_config import load_data_config
from totton_audio_de_mirroring.models.nbee import NeuralBandwidthExtension
from totton_audio_de_mirroring.models.nmse import NMSE

SR = 88_200
SRC = 44_100
DUR = 1.0


def main() -> None:
    """Run the high-band generation ceiling evaluation."""
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_data_config(args.data_config)
    lowpass, highpass = design_band_split_filters(
        cutoff_hz=cfg.band_split.cutoff_hz,
        sample_rate=SR,
        num_taps=cfg.band_split.num_taps,
        window=cfg.band_split.window,
    )

    nbee = _load(
        NeuralBandwidthExtension,
        args.nbee_checkpoint,
        cfg,
        lowpass,
        highpass,
        device,
        energy_cap=args.energy_cap,
    )
    baseline = (
        _load(
            NMSE,
            args.baseline_checkpoint,
            cfg,
            lowpass,
            highpass,
            device,
            energy_cap=1.0e-3,
        )
        if args.baseline_checkpoint
        else None
    )

    freqs = torch.fft.rfftfreq(2048, d=1.0 / SR).to(device)
    hb_bins = (freqs >= 20000.0) & (freqs <= 44100.0)

    errs: dict[str, list[float]] = {"degraded": [], "baseline": [], "nbee": []}
    cap_viol = 0
    lb_errs: list[float] = []
    n = 0
    for path in sorted(glob.glob(str(args.hires_root) + "/*.wav"))[:: args.stride]:
        info = sf.info(path)
        if int(info.samplerate) < SR:
            continue
        frames = min(int(DUR * info.samplerate * 2), int(info.frames))
        block, _ = sf.read(path, frames=frames, dtype="float64", always_2d=True)
        mono = block[:, 0] if block.shape[1] == 1 else block.mean(axis=1)
        native = resample_signal(mono, int(info.samplerate), SR)[: int(DUR * SR)]
        if native.shape[0] < int(DUR * SR):
            continue
        src = resample_signal(native, SR, SRC)
        x_full = upsample_bessel_reference(
            signal=src, source_sr=SRC, target_sr=SR, cutoff_hz=20000.0, order=6
        )[: native.shape[0]]
        true_hb = _hb_logmag(native, device, hb_bins)
        errs["degraded"].append(_l1(_hb_logmag(x_full, device, hb_bins), true_hb))
        out_nbee = _process(nbee, x_full, device)
        errs["nbee"].append(_l1(_hb_logmag(out_nbee, device, hb_bins), true_hb))
        cap_viol += int(_hb_energy(out_nbee, device, hb_bins) > args.energy_cap * 1.05)
        lb_errs.append(_lb_error(x_full, out_nbee, device))
        if baseline is not None:
            out_b = _process(baseline, x_full, device)
            errs["baseline"].append(_l1(_hb_logmag(out_b, device, hb_bins), true_hb))
        n += 1

    print(f"evaluated {n} real hi-res clips (energy_cap={args.energy_cap})")
    print("Mean HB(20-44kHz) log-mag L1 vs TRUE native HB (lower=better):")
    for k in ("degraded", "baseline", "nbee"):
        if errs[k]:
            a = np.array(errs[k])
            print(f"  {k:9s}: {a.mean():.4f} (median {np.median(a):.4f})")
    if errs["baseline"] and errs["nbee"]:
        b, g = np.array(errs["baseline"]), np.array(errs["nbee"])
        print(
            f"NBEE vs baseline: delta={g.mean() - b.mean():+.4f} "
            f"({'IMPROVED' if g.mean() < b.mean() else 'worse'}), "
            f"win={np.mean(g < b) * 100:.0f}%"
        )
    print(f"HB energy-cap violations (NBEE): {cap_viol}/{n}")
    print(
        f"LB(0-20kHz) preservation error (NBEE): mean={np.mean(lb_errs):.2e} "
        f"(should be ~0 by structure)"
    )


def _load(
    cls: type,
    path: Path,
    cfg: Any,
    lowpass: np.ndarray,
    highpass: np.ndarray,
    device: torch.device,
    *,
    energy_cap: float,
) -> Any:
    model = cls(
        sample_rate=SR,
        cutoff_hz=cfg.band_split.cutoff_hz,
        energy_cap=energy_cap,
        envelope_floor=cfg.hb_target.envelope_min,
        lowpass_taps=lowpass,
        highpass_taps=highpass,
    )
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval()


def _process(model: Any, x_full: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        xt = torch.tensor(x_full, dtype=torch.float32, device=device).unsqueeze(0)
        return np.asarray(model.forward(xt)[0].cpu().numpy())


def _hb_logmag(
    sig: np.ndarray, device: torch.device, hb_bins: torch.Tensor
) -> torch.Tensor:
    x = torch.tensor(sig, dtype=torch.float32, device=device).unsqueeze(0)
    st = torch.stft(
        x,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        window=torch.hann_window(2048, device=device),
        center=True,
        return_complex=True,
    )
    return torch.log(st.abs()[0][hb_bins] + 1e-6)


def _hb_energy(sig: np.ndarray, device: torch.device, hb_bins: torch.Tensor) -> float:
    x = torch.tensor(sig, dtype=torch.float32, device=device).unsqueeze(0)
    st = torch.stft(
        x,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        window=torch.hann_window(2048, device=device),
        center=True,
        return_complex=True,
        normalized=True,
    )
    return float((st.abs()[0][hb_bins] ** 2).mean().item())


def _lb_error(x_full: np.ndarray, out: np.ndarray, device: torch.device) -> float:
    n = min(x_full.shape[0], out.shape[0])
    a = torch.tensor(x_full[:n], dtype=torch.float32, device=device)
    b = torch.tensor(out[:n], dtype=torch.float32, device=device)

    def lb(sig: torch.Tensor) -> torch.Tensor:
        st = torch.stft(
            sig.unsqueeze(0),
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window=torch.hann_window(2048, device=device),
            center=True,
            return_complex=True,
        )
        freqs = torch.fft.rfftfreq(2048, d=1.0 / SR).to(device)
        return st.abs()[0][freqs < 20000.0]

    return float((lb(a) - lb(b)).abs().mean().item())


def _l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 1b HB generation.")
    parser.add_argument("--nbee-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--hires-root", type=Path, default=Path("data/hires_corpus"))
    parser.add_argument(
        "--data-config", type=Path, default=Path("configs/data_generation_gen88k2.yaml")
    )
    parser.add_argument("--energy-cap", type=float, default=1.0e-2)
    parser.add_argument("--stride", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    main()
