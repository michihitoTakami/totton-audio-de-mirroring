"""Derive and empirically verify a 32-bit-transparent Kaiser FIR 2x upsampler.

Reports, for candidate tap counts: passband ripple (0-20kHz), stopband
attenuation (>=22.05kHz), float32 round-trip error vs the analytic upsample,
and image rejection vs the legacy Bessel IIR. Picks the minimal transparent N.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.degradation import upsample_bessel_reference
from totton_audio_de_mirroring.data.filters import (
    design_transparent_upsampler_fir,
    kaiser_params_for_stopband,
    upsample_fir,
)

SRC = 44_100
RATIO = 2
TARGET = SRC * RATIO
PASS_HZ = 20_000.0
STOP_HZ = 22_050.0


def _response(taps: np.ndarray, ratio: int) -> tuple[np.ndarray, np.ndarray]:
    w, h = sp_signal.freqz(taps / ratio, worN=16384, fs=TARGET)
    return w, 20.0 * np.log10(np.abs(h) + 1e-30)


def _passband_ripple_db(w: np.ndarray, mag_db: np.ndarray) -> float:
    band = mag_db[w <= PASS_HZ]
    return float(np.max(np.abs(band)))


def _stopband_db(w: np.ndarray, mag_db: np.ndarray) -> float:
    band = mag_db[w >= STOP_HZ]
    return float(np.max(band))


def _roundtrip_error(taps: np.ndarray) -> float:
    """Max abs error of float32 upsampling a 0-20kHz signal vs analytic."""
    n = max(16384, taps.size * 2)
    t_src = np.arange(n) / SRC
    freqs = [1000.0, 5000.0, 10000.0, 18000.0]
    src = np.sum([0.2 * np.sin(2 * np.pi * f * t_src) for f in freqs], axis=0)
    up = upsample_fir(src.astype(np.float32).astype(np.float64), RATIO, taps)
    t_tgt = np.arange(n * RATIO) / TARGET
    ideal = np.sum([0.2 * np.sin(2 * np.pi * f * t_tgt) for f in freqs], axis=0)
    guard = taps.size
    return float(np.max(np.abs(up[guard:-guard] - ideal[guard:-guard])))


def _image_level_db(
    upsampler: Callable[[np.ndarray], np.ndarray], tone_hz: float
) -> float:
    """Image (mirror) level in dB for a near-Nyquist source tone after 2x."""
    n = 8192
    t = np.arange(n) / SRC
    src = 0.5 * np.sin(2 * np.pi * tone_hz * t)
    up = upsampler(src)
    spec = np.abs(np.fft.rfft(up * np.hanning(up.shape[0])))
    freqs = np.fft.rfftfreq(up.shape[0], d=1.0 / TARGET)
    fundamental = float(np.max(spec[freqs <= STOP_HZ]) + 1e-30)
    image = float(np.max(spec[freqs >= STOP_HZ]) + 1e-30)
    return float(20.0 * np.log10(image / fundamental))


def main() -> None:
    args = _parse_args()
    transition = STOP_HZ - PASS_HZ
    derived_n, beta = kaiser_params_for_stopband(args.stopband_db, transition, TARGET)
    print(
        f"target_sr={TARGET} transition={transition:.0f}Hz "
        f"stopband_db={args.stopband_db} -> beta={beta:.3f} derived_taps={derived_n}"
    )
    print()
    candidates = sorted({derived_n, 1025, 2049, 4097, 8193})

    def bessel_up(s: np.ndarray) -> np.ndarray:
        return upsample_bessel_reference(
            signal=s, source_sr=SRC, target_sr=TARGET, cutoff_hz=PASS_HZ, order=6
        )

    print(
        f"{'taps':>6} {'passripple_dB':>14} {'stopband_dB':>12} "
        f"{'roundtrip_err':>14} {'image@21k_dB':>13}"
    )
    bessel_img = _image_level_db(bessel_up, 21_000.0)
    for n in candidates:
        taps, _ = design_transparent_upsampler_fir(
            source_sr=SRC,
            ratio=RATIO,
            passband_hz=PASS_HZ,
            stopband_hz=STOP_HZ,
            stopband_db=args.stopband_db,
            num_taps=n,
        )
        w, mag = _response(taps, RATIO)
        rip = _passband_ripple_db(w, mag)
        stop = _stopband_db(w, mag)
        rte = _roundtrip_error(taps)

        def fir_up(s: np.ndarray, tp: np.ndarray = taps) -> np.ndarray:
            return upsample_fir(s, RATIO, tp)

        img = _image_level_db(fir_up, 21_000.0)
        print(f"{n:>6} {rip:>14.2e} {stop:>12.1f} {rte:>14.2e} {img:>13.1f}")
    print(
        f"\nlegacy Bessel IIR image@21k = {bessel_img:.1f} dB (higher = more leakage)"
    )
    print(
        "float32 floor ~ -144 dB; pick the smallest N with passripple<1e-7, "
        "stopband<-144dB, roundtrip~float32 floor."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive transparent Kaiser FIR.")
    parser.add_argument("--stopband-db", type=float, default=180.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
