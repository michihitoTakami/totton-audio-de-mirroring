"""Dataset for CAPB Stage 1 training (alias-free teacher/input pairs).

Teacher policy ``capb_bl_<target>`` (default ``capb_bl_88k2``; 48k family
uses 96 kHz): the teacher is synthesized natively at the target rate,
band-limited with a near-brickwall linear-phase FIR below the input Nyquist,
and the source-rate input is its exact 2:1 decimation. Input and target are
therefore perfectly consistent (x == target[::2]) and no Bessel degradation
path enters training. Per-sample masks derived from the target waveform
carry the plateau/silence structure the probe losses need. All mask windows
are millisecond-based, so they transfer unchanged across rate families.
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

from totton_audio_de_mirroring.data.generator import apply_soft_clip, generate_signal
from totton_audio_de_mirroring.data.transient_supervision import (
    TransientSupervisionConfig,
    cardinal_upsample,
    compute_pre_echo_mask,
    find_event_bounds,
)

SOURCE_SAMPLE_RATE = 44_100
TARGET_SAMPLE_RATE = 88_200
UPSAMPLE_RATIO = 2
# Near-Nyquist noise band edges: low edge is audible-band absolute; the high
# edge tops out just below the input Nyquist (97.5% at 44.1k) so the teacher
# brickwall does not remove the band entirely.
DEFAULT_NEAR_NYQUIST_HIGH_RANGE_HZ = (20_000.0, 21_500.0)
# The exclusion window must match the ringing gate, which starts measuring
# plateau ripple 0.1 ms after the edge: a wider window would let the mid
# prototype's ~0.3 ms settling tail escape the training loss while still
# failing the gate (observed in run5 as mid-instead-of-gentle at edges).
FLAT_MASK_WINDOW_MS = 0.15
FLAT_MASK_SLOPE_REL = 3.0e-4
QUIET_MASK_LEVEL_REL = 1.0e-3
EDGE_MASK_DILATION_MS = 3.0
EDGE_SLOPE_SPIKE_REL = 0.25


@dataclass(frozen=True)
class AugmentationConfig:
    """Configure CAPB training augmentation.

    Args:
        gain_range: Linear gain range applied to the teacher.
        polarity_flip_prob: Probability of flipping signal polarity.
        noise_std_range: Standard deviation range for additive noise.
        soft_clip_prob: Probability of applying soft clipping.
        soft_clip_drive_range: Drive range for soft clipping.

    Physical Basis:
        Mild level, polarity, noise, and saturation variation improves
        controller robustness without changing the interpolation contract.
    """

    gain_range: tuple[float, float] = (0.7, 1.0)
    polarity_flip_prob: float = 0.3
    noise_std_range: tuple[float, float] = (0.0, 0.002)
    soft_clip_prob: float = 0.2
    soft_clip_drive_range: tuple[float, float] = (1.2, 2.5)

    def __post_init__(self) -> None:
        """Validate augmentation ranges."""
        _validate_range(self.gain_range, "gain_range", positive=True)
        _validate_probability(self.polarity_flip_prob, "polarity_flip_prob")
        _validate_range(self.noise_std_range, "noise_std_range", non_negative=True)
        _validate_probability(self.soft_clip_prob, "soft_clip_prob")
        _validate_range(
            self.soft_clip_drive_range,
            "soft_clip_drive_range",
            positive=True,
        )


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
        """Validate the specification.

        The input-Nyquist bound depends on the configured rates, so it is
        checked by CAPBDataConfig where both are known.
        """
        if not 0.0 < self.passband_edge_hz < self.stopband_edge_hz:
            raise ValueError("Require 0 < passband_edge_hz < stopband_edge_hz.")
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
        transient_supervision: Event-focused transient training settings.
        source_sample_rate: Input sample rate in Hz (44.1k or 48k family).
        target_sample_rate: Teacher sample rate in Hz (2x the source rate).
        near_nyquist_high_range_hz: Range the near-Nyquist noise family's
            high edge is drawn from (rate-family dependent).
        flat_mask_window_ms: Edge-exclusion window of the plateau mask.
            The ringing gate measures the plateau from 0.1 ms after the
            edge, so any excess over 0.1 ms is a training-blind zone the
            gate still sees (48k run2 failed exactly there: the mid
            prototype's settling tail at 0.10-0.15 ms).

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
    transient_supervision: TransientSupervisionConfig = field(
        default_factory=TransientSupervisionConfig
    )
    source_sample_rate: int = SOURCE_SAMPLE_RATE
    target_sample_rate: int = TARGET_SAMPLE_RATE
    near_nyquist_high_range_hz: tuple[float, float] = DEFAULT_NEAR_NYQUIST_HIGH_RANGE_HZ
    flat_mask_window_ms: float = FLAT_MASK_WINDOW_MS

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
        if self.source_sample_rate <= 0:
            raise ValueError("source_sample_rate must be positive.")
        if self.target_sample_rate != self.source_sample_rate * UPSAMPLE_RATIO:
            raise ValueError(
                "target_sample_rate must equal source_sample_rate * "
                f"{UPSAMPLE_RATIO}, got {self.target_sample_rate} vs "
                f"{self.source_sample_rate}."
            )
        input_nyquist = self.source_sample_rate / 2.0
        if self.brickwall.stopband_edge_hz > input_nyquist:
            raise ValueError(
                "brickwall stopband_edge_hz must not exceed the input "
                f"Nyquist ({input_nyquist} Hz)."
            )
        low, high = self.near_nyquist_high_range_hz
        if not 0.0 < low < high <= input_nyquist:
            raise ValueError(
                "near_nyquist_high_range_hz must satisfy 0 < low < high <= "
                f"input Nyquist ({input_nyquist} Hz), got {low}/{high}."
            )
        if self.flat_mask_window_ms <= 0.0:
            raise ValueError("flat_mask_window_ms must be positive.")


DEFAULT_SIGNAL_MIX: dict[str, float] = {
    "square_wave": 0.10,
    "step_plateau": 0.10,
    "isolated_click": 0.05,
    "sawtooth_wave": 0.05,
    "tone_burst": 0.10,
    "music_like_mixture": 0.10,
    "multitone": 0.10,
    "imd_two_tone": 0.05,
    "sweep_log": 0.05,
    "sweep_linear": 0.03,
    "am_tone": 0.05,
    "fm_tone": 0.05,
    "percussive": 0.05,
    "pink_noise": 0.04,
    "band_limited_noise": 0.04,
    "near_nyquist_noise": 0.04,
}

STATIONARY_SIGNAL_TYPES = frozenset(
    {
        "square_wave",
        "sawtooth_wave",
        "multitone",
        "imd_two_tone",
        "am_tone",
        "fm_tone",
        "pink_noise",
        "band_limited_noise",
        "near_nyquist_noise",
    }
)


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
        self._brickwall_taps = _design_brickwall(
            config.brickwall, config.target_sample_rate
        )
        self._base_seed = (
            config.seed
            if config.seed is not None
            else int(np.random.SeedSequence().generate_state(1)[0])
        )
        names = sorted(config.signal_mix)
        weights = np.asarray([config.signal_mix[name] for name in names])
        self._mix_names = names
        self._mix_probs = weights / weights.sum()
        augmentation = config.augmentation
        self._clean_transient_augmentation = AugmentationConfig(
            gain_range=augmentation.gain_range,
            polarity_flip_prob=augmentation.polarity_flip_prob,
            noise_std_range=(0.0, 0.0),
            soft_clip_prob=0.0,
            soft_clip_drive_range=augmentation.soft_clip_drive_range,
        )

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
        focused = self._is_focused_transient(signal_type)
        transient_clean = focused and (
            rng.random() < self._config.transient_supervision.clean_probability
        )
        augmentation = (
            self._clean_transient_augmentation
            if transient_clean
            else self._config.augmentation
        )
        generator_seed = int(rng.integers(0, 2**32 - 1))
        if focused:
            clean_source = generate_signal(
                signal_type,
                sample_rate=self._config.source_sample_rate,
                duration_sec=self._config.source_duration_sec,
                seed=generator_seed,
                **params,
            ).astype(np.float64)
            source_event_bounds = find_event_bounds(clean_source)
            augmented_source = apply_augmentations(clean_source, augmentation, rng)
            clean_full = cardinal_upsample(clean_source, UPSAMPLE_RATIO)
            target_full = cardinal_upsample(augmented_source, UPSAMPLE_RATIO)
            event_bounds = (
                source_event_bounds[0] * UPSAMPLE_RATIO,
                source_event_bounds[1] * UPSAMPLE_RATIO,
            )
        else:
            clean_full = generate_signal(
                signal_type
                if signal_type != "near_nyquist_noise"
                else ("band_limited_noise"),
                sample_rate=self._config.target_sample_rate,
                duration_sec=self._config.source_duration_sec,
                seed=generator_seed,
                **params,
            ).astype(np.float64)
            augmented = apply_augmentations(clean_full, augmentation, rng)
            target_full = _apply_brickwall(augmented, self._brickwall_taps)
            event_bounds = None
        target_chunk, chunk_start = self._extract_chunk(target_full, rng, event_bounds)
        source_chunk = target_chunk[::UPSAMPLE_RATIO].copy()

        # Masks come from the CLEAN pre-brickwall signal: the band-limited
        # target itself carries Gibbs ripple on plateaus and augmentation
        # noise in silences, but ringing losses need "where the underlying
        # signal is flat/quiet", not where the training target happens to be.
        clean_chunk = clean_full[chunk_start : chunk_start + target_chunk.size]
        flat_mask = compute_flat_mask(
            clean_chunk,
            self._config.target_sample_rate,
            window_ms=self._config.flat_mask_window_ms,
        )
        quiet_mask = compute_quiet_mask(clean_chunk, self._config.target_sample_rate)
        # Slope spikes are enabled only for generator-labelled edge families;
        # stationary broadband noise never enters that path.
        edge_mask = compute_edge_mask(
            flat_mask,
            quiet_mask,
            clean_signal=(
                clean_chunk
                if signal_type
                in self._config.transient_supervision.edge_supervision_signal_types
                else None
            ),
            sample_rate=self._config.target_sample_rate,
        )
        pre_echo_mask = np.zeros_like(clean_chunk)
        if event_bounds is not None:
            pre_echo_mask = compute_pre_echo_mask(
                clean_chunk.size,
                event_start=event_bounds[0] - chunk_start,
                sample_rate=self._config.target_sample_rate,
                guard_ms=self._config.transient_supervision.pre_echo_guard_ms,
                window_ms=self._config.transient_supervision.pre_echo_window_ms,
            )
        if focused and not transient_clean:
            flat_mask = np.zeros_like(flat_mask)
            quiet_mask = np.zeros_like(quiet_mask)

        return {
            "source": torch.from_numpy(source_chunk.astype(np.float32)),
            "target": torch.from_numpy(target_chunk.astype(np.float32)),
            "flat_mask": torch.from_numpy(flat_mask.astype(np.float32)),
            "quiet_mask": torch.from_numpy(quiet_mask.astype(np.float32)),
            "edge_mask": torch.from_numpy(edge_mask.astype(np.float32)),
            "pre_echo_mask": torch.from_numpy(pre_echo_mask.astype(np.float32)),
            "stationary": torch.tensor(
                signal_type in STATIONARY_SIGNAL_TYPES, dtype=torch.bool
            ),
            "signal_type": signal_type,
            "chunk_start": int(chunk_start),
            "focused_event": torch.tensor(focused, dtype=torch.bool),
            "transient_clean": torch.tensor(transient_clean, dtype=torch.bool),
        }

    def _extract_chunk(
        self,
        target_full: np.ndarray,
        rng: np.random.Generator,
        event_bounds: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, int]:
        chunk_len = int(
            round(self._config.chunk_duration_sec * self._config.target_sample_rate)
        )
        margin = self._brickwall_taps.size // 2
        max_start = target_full.size - chunk_len - margin
        if max_start < margin:
            raise ValueError(
                "source_duration_sec too short for chunk plus filter margins."
            )
        low_start, high_start = margin, max_start
        if event_bounds is not None:
            context = int(
                round(
                    self._config.transient_supervision.context_ms
                    * self._config.target_sample_rate
                    / 1_000.0
                )
            )
            event_start, event_stop = event_bounds
            low_start = max(low_start, event_stop + context - chunk_len)
            high_start = min(high_start, event_start - context)
        low_start += (-low_start) % UPSAMPLE_RATIO
        high_start -= high_start % UPSAMPLE_RATIO
        if low_start > high_start:
            raise ValueError("Event and required context do not fit a valid chunk.")
        if self._config.random_chunk:
            choices = (high_start - low_start) // UPSAMPLE_RATIO + 1
            start = low_start + UPSAMPLE_RATIO * int(rng.integers(0, choices))
        elif event_bounds is not None:
            start = low_start + (high_start - low_start) // 2
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
        elif signal_type == "imd_two_tone":
            params = {
                "low_tone_hz": float(rng.uniform(40.0, 120.0)),
                "high_tone_hz": float(rng.uniform(4_000.0, 12_000.0)),
                "amplitude_ratio": float(rng.uniform(3.0, 5.0)),
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
        elif signal_type == "isolated_click":
            params = {"click_width_samples": int(rng.integers(1, 6))}
        elif signal_type == "band_limited_noise":
            low = float(rng.uniform(40.0, 4_000.0))
            params = {
                "low_hz": low,
                "high_hz": float(rng.uniform(low + 1_000.0, 18_000.0)),
            }
        elif signal_type == "near_nyquist_noise":
            low = float(rng.uniform(15_000.0, 18_000.0))
            high_low, high_high = self._config.near_nyquist_high_range_hz
            params = {"low_hz": low, "high_hz": float(rng.uniform(high_low, high_high))}
        if self._is_focused_transient(signal_type):
            params["center_fraction"] = float(
                rng.uniform(*self._config.transient_supervision.center_fraction_range)
            )
        return signal_type, params

    def _is_focused_transient(self, signal_type: str) -> bool:
        return self._config.transient_supervision.enabled and (
            signal_type in self._config.transient_supervision.focus_signal_types
        )

    def _rng_for_index(self, index: int) -> np.random.Generator:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed = int(self._base_seed + index + worker_id * 1_000_000)
        return np.random.default_rng(seed)


def apply_augmentations(
    signal: np.ndarray,
    config: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply immutable on-the-fly CAPB training augmentation.

    Args:
        signal: Input waveform. It is not modified.
        config: Augmentation settings.
        rng: Random generator used for reproducibility.

    Returns:
        Augmented float32 waveform.

    Physical Basis:
        Gain, polarity, low-level noise, and optional soft clipping expose the
        controller to level variation without altering the fixed FIR bank.
    """
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("signal must be a non-empty 1D array.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must contain only finite values.")

    augmented = signal * rng.uniform(*config.gain_range)
    if rng.random() < config.polarity_flip_prob:
        augmented = -augmented
    noise_std = rng.uniform(*config.noise_std_range)
    if noise_std > 0.0:
        augmented = augmented + rng.normal(0.0, noise_std, size=signal.shape)
    if rng.random() < config.soft_clip_prob:
        drive = rng.uniform(*config.soft_clip_drive_range)
        augmented = apply_soft_clip(augmented, drive=float(drive))
    return np.asarray(augmented, dtype=np.float32)


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
    transient_raw = raw.get("transient_supervision", {})
    transient_supervision = TransientSupervisionConfig(
        enabled=bool(transient_raw.get("enabled", False)),
        focus_signal_types=tuple(
            transient_raw.get("focus_signal_types", ("isolated_click", "tone_burst"))
        ),
        clean_probability=float(transient_raw.get("clean_probability", 0.7)),
        center_fraction_range=_parse_range(
            transient_raw.get("center_fraction_range", (0.2, 0.8))
        ),
        context_ms=float(transient_raw.get("context_ms", 5.0)),
        pre_echo_guard_ms=float(transient_raw.get("pre_echo_guard_ms", 0.5)),
        pre_echo_window_ms=float(transient_raw.get("pre_echo_window_ms", 3.5)),
        edge_supervision_signal_types=tuple(
            transient_raw.get(
                "edge_supervision_signal_types",
                ("square_wave", "step_plateau", "isolated_click"),
            )
        ),
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
        transient_supervision=transient_supervision,
        source_sample_rate=int(raw.get("source_sample_rate", SOURCE_SAMPLE_RATE)),
        target_sample_rate=int(raw.get("target_sample_rate", TARGET_SAMPLE_RATE)),
        near_nyquist_high_range_hz=_parse_range(
            raw.get("near_nyquist_high_range_hz", DEFAULT_NEAR_NYQUIST_HIGH_RANGE_HZ)
        ),
        flat_mask_window_ms=float(raw.get("flat_mask_window_ms", FLAT_MASK_WINDOW_MS)),
    )


def _parse_range(raw_range: object) -> tuple[float, float]:
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        raise ValueError(f"Expected a [low, high] pair, got {raw_range!r}.")
    return float(raw_range[0]), float(raw_range[1])


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")


def _validate_range(
    value: tuple[float, float],
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> None:
    if len(value) != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite (low, high) pair.")
    low, high = value
    if low > high:
        raise ValueError(f"{name} lower bound must not exceed upper bound.")
    if positive and low <= 0.0:
        raise ValueError(f"{name} values must be positive.")
    if non_negative and low < 0.0:
        raise ValueError(f"{name} values must be non-negative.")


def _design_brickwall(
    config: BrickwallConfig, target_sample_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    width_hz = config.stopband_edge_hz - config.passband_edge_hz
    num_taps, beta = sp_signal.kaiserord(
        config.attenuation_db, width_hz / (target_sample_rate / 2)
    )
    if num_taps % 2 == 0:
        num_taps += 1
    cutoff = 0.5 * (config.passband_edge_hz + config.stopband_edge_hz)
    return np.asarray(
        sp_signal.firwin(
            num_taps, cutoff, window=("kaiser", beta), fs=target_sample_rate
        ),
        dtype=np.float64,
    )


def _apply_brickwall(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    return np.asarray(
        sp_signal.fftconvolve(signal, taps, mode="same"), dtype=np.float64
    )


def compute_flat_mask(
    clean_signal: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    window_ms: float = FLAT_MASK_WINDOW_MS,
) -> np.ndarray:
    """Mark samples on locally flat regions (plateaus) of the clean signal.

    Args:
        clean_signal: Pre-brickwall, pre-augmentation waveform.
        sample_rate: Sample rate of the clean signal in Hz.
        window_ms: Edge-exclusion half-window in milliseconds.

    Returns:
        Float mask (1.0 on plateaus, 0.0 elsewhere).

    Physical Basis:
        On a plateau of the underlying clean signal, any high-frequency
        content the system emits is interpolation ringing by definition, so
        the probe losses can penalize it without an explicit edge detector.
        The window is defined in milliseconds (gate-aligned), so it holds
        across rate families; shrinking it toward the gate's 0.1 ms plateau
        start closes the training-blind zone next to each edge.
    """
    peak = max(float(np.max(np.abs(clean_signal))), 1e-12)
    slope = np.abs(np.diff(clean_signal, prepend=clean_signal[:1]))
    window = max(1, int(round(window_ms * sample_rate / 1_000.0)))
    local_max = _moving_max(slope, window)
    return (local_max < FLAT_MASK_SLOPE_REL * peak).astype(np.float64)


def compute_quiet_mask(
    clean_signal: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """Mark samples where the clean signal is silent.

    Args:
        clean_signal: Pre-brickwall, pre-augmentation waveform.
        sample_rate: Sample rate of the clean signal in Hz.

    Returns:
        Float mask (1.0 in silence, 0.0 elsewhere).

    Physical Basis:
        Energy the system emits where the clean signal is silent is
        pre/post-echo or hallucination; penalizing it needs no edge
        detection at all.
    """
    peak = max(float(np.max(np.abs(clean_signal))), 1e-12)
    window = max(1, int(round(FLAT_MASK_WINDOW_MS * sample_rate / 1_000.0)))
    envelope = _moving_max(np.abs(clean_signal), window)
    return (envelope < QUIET_MASK_LEVEL_REL * peak).astype(np.float64)


def compute_edge_mask(
    flat_mask: np.ndarray,
    quiet_mask: np.ndarray,
    clean_signal: np.ndarray | None = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Mark broadband-transient neighborhoods.

    Args:
        flat_mask: Plateau mask from compute_flat_mask.
        quiet_mask: Silence mask from compute_quiet_mask.
        clean_signal: Optional clean waveform; when given, large slope
            spikes also count as edges (covers dense-edge signals like
            5 kHz squares whose plateaus are too short for flat_mask).
        sample_rate: Sample rate of the masks in Hz.

    Returns:
        Float mask (1.0 near broadband transients, 0.0 elsewhere).

    Physical Basis:
        Edges of plateaus, onsets out of silence, and sample-scale jumps
        are the points where "ring like the brickwall teacher" and "stay
        clean like the Bessel reference" genuinely conflict; the fidelity
        losses are relaxed there so the controller can choose the ring-free
        behavior without fighting the teacher. A slope spike threshold of
        a quarter of the peak cannot be reached by any band-limited-to-10k
        oscillation at that amplitude, so steady tonal content is never
        marked.
    """
    if flat_mask.shape != quiet_mask.shape:
        raise ValueError("flat_mask and quiet_mask must share a shape.")
    transitions = np.zeros_like(flat_mask)
    transitions[1:] = np.maximum(
        np.abs(np.diff(flat_mask)), np.abs(np.diff(quiet_mask))
    )
    if clean_signal is not None:
        if clean_signal.shape != flat_mask.shape:
            raise ValueError("clean_signal must share the mask shape.")
        peak = max(float(np.max(np.abs(clean_signal))), 1e-12)
        spikes = np.zeros_like(flat_mask)
        spikes[1:] = (
            np.abs(np.diff(clean_signal)) > EDGE_SLOPE_SPIKE_REL * peak
        ).astype(np.float64)
        transitions = np.maximum(transitions, spikes)
    half_window = max(1, int(round(EDGE_MASK_DILATION_MS * sample_rate / 1_000.0)))
    return (_moving_max(transitions, half_window) > 0.0).astype(np.float64)


def _moving_max(values: np.ndarray, half_window: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d

    return np.asarray(
        maximum_filter1d(values, size=2 * half_window + 1, mode="nearest"),
        dtype=np.float64,
    )


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))
