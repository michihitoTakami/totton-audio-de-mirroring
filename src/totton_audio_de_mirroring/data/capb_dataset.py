"""Dataset for CAPB Stage 1 training (alias-free teacher/input pairs).

Teacher policy ``capb_bl_88k2``: the teacher is synthesized natively at
88.2 kHz, band-limited with a near-brickwall linear-phase FIR below the
input Nyquist, and the 44.1 kHz input is its exact 2:1 decimation. Input and
target are therefore perfectly consistent (x == target[::2]) and no Bessel
degradation path enters training. Per-sample masks derived from the target
waveform carry the plateau/silence structure the probe losses need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from scipy import signal as sp_signal
from torch.utils.data import Dataset

from totton_audio_de_mirroring.data.dataset import apply_augmentations
from totton_audio_de_mirroring.data.generator import generate_signal
from totton_audio_de_mirroring.data.pipeline_config import AugmentationConfig

SOURCE_SAMPLE_RATE = 44_100
TARGET_SAMPLE_RATE = 88_200
UPSAMPLE_RATIO = 2
FLAT_MASK_WINDOW_MS = 0.5
FLAT_MASK_SLOPE_REL = 3.0e-4
QUIET_MASK_LEVEL_REL = 1.0e-3


@dataclass(frozen=True)
class BrickwallConfig:
    """Teacher band-limiting filter specification.

    Args:
        passband_edge_hz: Last fully passed frequency in Hz.
        stopband_edge_hz: First fully attenuated frequency in Hz.
        attenuation_db: Kaiser stopband attenuation in dB.

    Physical Basis:
        Limiting the teacher strictly below the input Nyquist (22.05 kHz)
        makes plain 2:1 decimation alias-free, so the training input is an
        exact, phase-true observation of the target.
    """

    passband_edge_hz: float = 21_800.0
    stopband_edge_hz: float = 22_050.0
    attenuation_db: float = 120.0

    def __post_init__(self) -> None:
        """Validate the specification."""
        if not 0.0 < self.passband_edge_hz < self.stopband_edge_hz:
            raise ValueError("Require 0 < passband_edge_hz < stopband_edge_hz.")
        if self.stopband_edge_hz > TARGET_SAMPLE_RATE / 4:
            raise ValueError(
                "stopband_edge_hz must not exceed the input Nyquist "
                f"({TARGET_SAMPLE_RATE / 4} Hz)."
            )
        if self.attenuation_db <= 0.0:
            raise ValueError("attenuation_db must be positive.")


@dataclass(frozen=True)
class CAPBDataConfig:
    """Configuration for the CAPB training dataset.

    Args:
        num_samples: Dataset length.
        source_duration_sec: Full synthesized signal duration in seconds.
        chunk_duration_sec: Training chunk duration in seconds.
        random_chunk: Sample the chunk position randomly per item.
        seed: Base RNG seed.
        signal_mix: Mapping of signal type to sampling weight.
        brickwall: Teacher band-limiting specification.
        augmentation: Augmentation configuration (applied to the teacher).

    Physical Basis:
        The signal mix intentionally over-weights edge-rich families
        (squares, plateaus, clicks) that were absent from the previous
        training distribution and are exactly what the ringing gates probe.
    """

    num_samples: int = 10_000
    source_duration_sec: float = 1.0
    chunk_duration_sec: float = 0.25
    random_chunk: bool = True
    seed: int | None = 1234
    signal_mix: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SIGNAL_MIX)
    )
    brickwall: BrickwallConfig = field(default_factory=BrickwallConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self.chunk_duration_sec > self.source_duration_sec:
            raise ValueError("chunk_duration_sec must not exceed source duration.")
        if not self.signal_mix:
            raise ValueError("signal_mix must not be empty.")
        if any(weight < 0.0 for weight in self.signal_mix.values()):
            raise ValueError("signal_mix weights must be non-negative.")
        if sum(self.signal_mix.values()) <= 0.0:
            raise ValueError("signal_mix weights must sum to a positive value.")


DEFAULT_SIGNAL_MIX: dict[str, float] = {
    "square_wave": 0.10,
    "step_plateau": 0.10,
    "isolated_click": 0.05,
    "sawtooth_wave": 0.05,
    "tone_burst": 0.10,
    "music_like_mixture": 0.15,
    "multitone": 0.10,
    "sweep_log": 0.05,
    "sweep_linear": 0.03,
    "am_tone": 0.05,
    "fm_tone": 0.05,
    "percussive": 0.05,
    "pink_noise": 0.04,
    "band_limited_noise": 0.04,
    "near_nyquist_noise": 0.04,
}


class CAPBUpsampleDataset(Dataset[dict[str, Any]]):
    """Alias-free (input, target) pairs for the CAPB upsampler.

    Args:
        config: Dataset configuration.

    Physical Basis:
        Because the input is the exact decimation of the band-limited
        target, the only information the model must supply is the choice of
        interpolation behavior - which is precisely the controller's job.
    """

    def __init__(self, config: CAPBDataConfig) -> None:
        self._config = config
        self._brickwall_taps = _design_brickwall(config.brickwall)
        self._base_seed = (
            config.seed
            if config.seed is not None
            else int(np.random.SeedSequence().generate_state(1)[0])
        )
        names = sorted(config.signal_mix)
        weights = np.asarray([config.signal_mix[name] for name in names])
        self._mix_names = names
        self._mix_probs = weights / weights.sum()

    def __len__(self) -> int:
        """Return dataset length."""
        return self._config.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one training sample.

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= self._config.num_samples:
            raise IndexError("index out of range")

        rng = self._rng_for_index(index)
        signal_type, params = self._sample_request(rng)

        clean_full = generate_signal(
            signal_type
            if signal_type != "near_nyquist_noise"
            else ("band_limited_noise"),
            sample_rate=TARGET_SAMPLE_RATE,
            duration_sec=self._config.source_duration_sec,
            seed=int(rng.integers(0, 2**32 - 1)),
            **params,
        ).astype(np.float64)
        augmented = apply_augmentations(clean_full, self._config.augmentation, rng)

        target_full = _apply_brickwall(augmented, self._brickwall_taps)
        target_chunk, chunk_start = self._extract_chunk(target_full, rng)
        source_chunk = target_chunk[::UPSAMPLE_RATIO].copy()

        # Masks come from the CLEAN pre-brickwall signal: the band-limited
        # target itself carries Gibbs ripple on plateaus and augmentation
        # noise in silences, but ringing losses need "where the underlying
        # signal is flat/quiet", not where the training target happens to be.
        clean_chunk = clean_full[chunk_start : chunk_start + target_chunk.size]
        flat_mask = compute_flat_mask(clean_chunk)
        quiet_mask = compute_quiet_mask(clean_chunk)

        return {
            "source": torch.from_numpy(source_chunk.astype(np.float32)),
            "target": torch.from_numpy(target_chunk.astype(np.float32)),
            "flat_mask": torch.from_numpy(flat_mask.astype(np.float32)),
            "quiet_mask": torch.from_numpy(quiet_mask.astype(np.float32)),
            "signal_type": signal_type,
            "chunk_start": int(chunk_start),
        }

    def _extract_chunk(
        self, target_full: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, int]:
        chunk_len = int(round(self._config.chunk_duration_sec * TARGET_SAMPLE_RATE))
        margin = self._brickwall_taps.size // 2
        max_start = target_full.size - chunk_len - margin
        if max_start < margin:
            raise ValueError(
                "source_duration_sec too short for chunk plus filter margins."
            )
        if self._config.random_chunk:
            start = int(rng.integers(margin, max_start + 1))
        else:
            start = margin
        start -= start % UPSAMPLE_RATIO
        return target_full[start : start + chunk_len], start

    def _sample_request(self, rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
        signal_type = str(rng.choice(self._mix_names, p=self._mix_probs))
        params: dict[str, Any] = {}
        if signal_type == "square_wave":
            params = {
                "frequency_hz": _log_uniform(rng, 40.0, 5_000.0),
                "duty": float(rng.uniform(0.3, 0.7)),
            }
        elif signal_type == "sawtooth_wave":
            params = {
                "frequency_hz": _log_uniform(rng, 40.0, 2_000.0),
                "width": float(rng.choice([1.0, 0.5])),
            }
        elif signal_type == "tone_burst":
            params = {
                "frequency_hz": _log_uniform(rng, 100.0, 19_000.0),
                "burst_ms": float(rng.uniform(5.0, 50.0)),
            }
        elif signal_type == "multitone":
            count = int(rng.integers(2, 9))
            params = {
                "frequencies_hz": np.sort(
                    rng.uniform(40.0, 18_000.0, size=count)
                ).tolist()
            }
        elif signal_type in {"sweep_log", "sweep_linear"}:
            start = float(rng.uniform(20.0, 2_000.0))
            params = {
                "start_hz": start,
                "end_hz": float(rng.uniform(start + 1_000.0, 20_000.0)),
            }
        elif signal_type == "am_tone":
            params = {
                "carrier_hz": _log_uniform(rng, 200.0, 15_000.0),
                "mod_hz": float(rng.uniform(2.0, 200.0)),
                "modulation_index": float(rng.uniform(0.2, 0.9)),
            }
        elif signal_type == "fm_tone":
            params = {
                "carrier_hz": _log_uniform(rng, 200.0, 10_000.0),
                "mod_hz": float(rng.uniform(2.0, 200.0)),
                "modulation_index": float(rng.uniform(0.5, 4.0)),
            }
        elif signal_type == "band_limited_noise":
            low = float(rng.uniform(40.0, 4_000.0))
            params = {
                "low_hz": low,
                "high_hz": float(rng.uniform(low + 1_000.0, 18_000.0)),
            }
        elif signal_type == "near_nyquist_noise":
            low = float(rng.uniform(15_000.0, 18_000.0))
            params = {"low_hz": low, "high_hz": float(rng.uniform(20_000.0, 21_500.0))}
        return signal_type, params

    def _rng_for_index(self, index: int) -> np.random.Generator:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed = int(self._base_seed + index + worker_id * 1_000_000)
        return np.random.default_rng(seed)


def load_capb_data_config(path: Path) -> CAPBDataConfig:
    """Load a CAPB dataset configuration from YAML.

    Args:
        path: Configuration file path.

    Returns:
        Parsed CAPBDataConfig.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the configuration is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}

    brickwall = BrickwallConfig(**raw.get("brickwall", {}))
    aug_raw = raw.get("augmentation", {})
    augmentation = AugmentationConfig(
        gain_range=tuple(aug_raw.get("gain_range", (0.7, 1.0))),
        polarity_flip_prob=float(aug_raw.get("polarity_flip_prob", 0.3)),
        noise_std_range=tuple(aug_raw.get("noise_std_range", (0.0, 0.002))),
        soft_clip_prob=float(aug_raw.get("soft_clip_prob", 0.2)),
        soft_clip_drive_range=tuple(aug_raw.get("soft_clip_drive_range", (1.2, 2.5))),
    )
    return CAPBDataConfig(
        num_samples=int(raw.get("num_samples", 10_000)),
        source_duration_sec=float(raw.get("source_duration_sec", 1.0)),
        chunk_duration_sec=float(raw.get("chunk_duration_sec", 0.25)),
        random_chunk=bool(raw.get("random_chunk", True)),
        seed=raw.get("seed", 1234),
        signal_mix=dict(raw.get("signal_mix", DEFAULT_SIGNAL_MIX)),
        brickwall=brickwall,
        augmentation=augmentation,
    )


def _design_brickwall(config: BrickwallConfig) -> np.ndarray:
    width_hz = config.stopband_edge_hz - config.passband_edge_hz
    num_taps, beta = sp_signal.kaiserord(
        config.attenuation_db, width_hz / (TARGET_SAMPLE_RATE / 2)
    )
    if num_taps % 2 == 0:
        num_taps += 1
    cutoff = 0.5 * (config.passband_edge_hz + config.stopband_edge_hz)
    return np.asarray(
        sp_signal.firwin(
            num_taps, cutoff, window=("kaiser", beta), fs=TARGET_SAMPLE_RATE
        ),
        dtype=np.float64,
    )


def _apply_brickwall(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    return np.asarray(
        sp_signal.fftconvolve(signal, taps, mode="same"), dtype=np.float64
    )


def compute_flat_mask(clean_signal: np.ndarray) -> np.ndarray:
    """Mark samples on locally flat regions (plateaus) of the clean signal.

    Args:
        clean_signal: Pre-brickwall, pre-augmentation waveform.

    Returns:
        Float mask (1.0 on plateaus, 0.0 elsewhere).

    Physical Basis:
        On a plateau of the underlying clean signal, any high-frequency
        content the system emits is interpolation ringing by definition, so
        the probe losses can penalize it without an explicit edge detector.
    """
    peak = max(float(np.max(np.abs(clean_signal))), 1e-12)
    slope = np.abs(np.diff(clean_signal, prepend=clean_signal[:1]))
    window = max(1, int(round(FLAT_MASK_WINDOW_MS * TARGET_SAMPLE_RATE / 1_000.0)))
    local_max = _moving_max(slope, window)
    return (local_max < FLAT_MASK_SLOPE_REL * peak).astype(np.float64)


def compute_quiet_mask(clean_signal: np.ndarray) -> np.ndarray:
    """Mark samples where the clean signal is silent.

    Args:
        clean_signal: Pre-brickwall, pre-augmentation waveform.

    Returns:
        Float mask (1.0 in silence, 0.0 elsewhere).

    Physical Basis:
        Energy the system emits where the clean signal is silent is
        pre/post-echo or hallucination; penalizing it needs no edge
        detection at all.
    """
    peak = max(float(np.max(np.abs(clean_signal))), 1e-12)
    window = max(1, int(round(FLAT_MASK_WINDOW_MS * TARGET_SAMPLE_RATE / 1_000.0)))
    envelope = _moving_max(np.abs(clean_signal), window)
    return (envelope < QUIET_MASK_LEVEL_REL * peak).astype(np.float64)


def _moving_max(values: np.ndarray, half_window: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d

    return np.asarray(
        maximum_filter1d(values, size=2 * half_window + 1, mode="nearest"),
        dtype=np.float64,
    )


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))
