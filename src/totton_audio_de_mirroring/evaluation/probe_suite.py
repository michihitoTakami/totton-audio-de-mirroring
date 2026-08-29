"""Deterministic probe-signal suite for Stage 1 gate evaluation.

Probes come in two tiers: the canonical tier may also be used by training
probe losses, while the held-out tier is gate-only (never referenced by any
loss or data-generation config) to prevent training from overfitting the
gates (Goodhart). Both tiers gate identically; a checkpoint must pass both.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

DEFAULT_SOURCE_SAMPLE_RATE = 44_100
DEFAULT_DURATION_SEC = 1.0
MANIFEST_VERSION = 2

TIER_CANONICAL = "canonical"
TIER_HELD_OUT = "held_out"

KIND_SQUARE = "square"
KIND_DC_STEP = "dc_step"
KIND_IMPULSE = "impulse"
KIND_IMPULSE_TRAIN = "impulse_train"
KIND_TONE_BURST = "tone_burst"
KIND_SWEEP_LOG = "sweep_log"
KIND_PINK_NOISE = "pink_noise"
KIND_MULTITONE = "multitone"
KIND_IMD_TWO_TONE = "imd_two_tone"

_VALID_KINDS = frozenset(
    {
        KIND_SQUARE,
        KIND_DC_STEP,
        KIND_IMPULSE,
        KIND_IMPULSE_TRAIN,
        KIND_TONE_BURST,
        KIND_SWEEP_LOG,
        KIND_PINK_NOISE,
        KIND_MULTITONE,
        KIND_IMD_TWO_TONE,
    }
)
_VALID_TIERS = frozenset({TIER_CANONICAL, TIER_HELD_OUT})


@dataclass(frozen=True)
class ProbeSpec:
    """Specification of one deterministic probe signal.

    Args:
        probe_id: Unique identifier (encodes kind/frequency/tier).
        kind: Signal kind (one of the KIND_* constants).
        tier: "canonical" (train + gate) or "held_out" (gate-only).
        amplitude: Peak amplitude of the probe.
        duration_sec: Probe duration in seconds.
        frequency_hz: Primary frequency (square/tone/burst), if applicable.
        frequency_end_hz: Sweep end frequency, if applicable.
        secondary_frequency_hz: High primary of an IMD two-tone probe.
        amplitude_ratio: Low-to-high amplitude ratio of an IMD probe.
        burst_ms: Tone-burst Hann gate length in milliseconds.
        period_ms: Impulse-train period in milliseconds.
        step_sign: DC-step direction (+1 rise, -1 fall).
        seed: RNG seed for stochastic probes (noise, multitone phases).

    Physical Basis:
        Probes are fully determined by this spec so gate results are
        reproducible and the manifest hash can certify which suite a
        checkpoint was gated against.
    """

    probe_id: str
    kind: str
    tier: str
    amplitude: float = 0.5
    duration_sec: float = DEFAULT_DURATION_SEC
    frequency_hz: float | None = None
    frequency_end_hz: float | None = None
    secondary_frequency_hz: float | None = None
    amplitude_ratio: float | None = None
    burst_ms: float | None = None
    period_ms: float | None = None
    step_sign: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the specification at construction."""
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"Unknown probe kind: {self.kind}")
        if self.tier not in _VALID_TIERS:
            raise ValueError(f"Unknown probe tier: {self.tier}")
        if self.amplitude <= 0.0 or self.duration_sec <= 0.0:
            raise ValueError("amplitude and duration_sec must be positive.")


def build_default_probe_suite() -> tuple[ProbeSpec, ...]:
    """Build the frozen default probe suite (canonical + held-out tiers).

    Returns:
        Tuple of probe specifications.

    Physical Basis:
        Low-frequency squares and DC steps expose plateau ripple (the known
        failure mode); bursts and impulses expose pre/post-echo; sweeps,
        noise, and multitones expose image leakage and passband notches.
        Held-out frequencies are deliberately non-round so no synthetic
        training family aligns with them exactly.
    """
    specs: list[ProbeSpec] = []

    for freq in (50.0, 100.0, 500.0, 1_000.0, 2_000.0, 5_000.0):
        specs.append(
            ProbeSpec(
                probe_id=f"square_{int(freq)}hz",
                kind=KIND_SQUARE,
                tier=TIER_CANONICAL,
                frequency_hz=freq,
            )
        )
    specs.append(
        ProbeSpec(
            probe_id="square_500hz_a005",
            kind=KIND_SQUARE,
            tier=TIER_CANONICAL,
            frequency_hz=500.0,
            amplitude=0.05,
        )
    )
    specs.append(
        ProbeSpec(
            probe_id="dc_step_up",
            kind=KIND_DC_STEP,
            tier=TIER_CANONICAL,
            step_sign=1,
        )
    )
    specs.append(
        ProbeSpec(
            probe_id="dc_step_down",
            kind=KIND_DC_STEP,
            tier=TIER_CANONICAL,
            step_sign=-1,
        )
    )
    specs.append(ProbeSpec(probe_id="impulse", kind=KIND_IMPULSE, tier=TIER_CANONICAL))
    specs.append(
        ProbeSpec(
            probe_id="impulse_train_10ms",
            kind=KIND_IMPULSE_TRAIN,
            tier=TIER_CANONICAL,
            period_ms=10.0,
        )
    )
    for freq in (1_000.0, 10_000.0, 19_000.0):
        specs.append(
            ProbeSpec(
                probe_id=f"tone_burst_{int(freq)}hz",
                kind=KIND_TONE_BURST,
                tier=TIER_CANONICAL,
                frequency_hz=freq,
                burst_ms=10.0,
            )
        )
    specs.append(
        ProbeSpec(
            probe_id="sweep_log_20_20k",
            kind=KIND_SWEEP_LOG,
            tier=TIER_CANONICAL,
            frequency_hz=20.0,
            frequency_end_hz=20_000.0,
            duration_sec=2.0,
        )
    )
    specs.append(
        ProbeSpec(
            probe_id="pink_noise_s1234",
            kind=KIND_PINK_NOISE,
            tier=TIER_CANONICAL,
            seed=1234,
        )
    )
    specs.append(
        ProbeSpec(
            probe_id="multitone_60_s20260704",
            kind=KIND_MULTITONE,
            tier=TIER_CANONICAL,
            seed=20260704,
        )
    )
    specs.append(
        ProbeSpec(
            probe_id="imd_60hz_7000hz",
            kind=KIND_IMD_TWO_TONE,
            tier=TIER_CANONICAL,
            frequency_hz=60.0,
            secondary_frequency_hz=7_000.0,
            amplitude_ratio=4.0,
            duration_sec=3.0,
        )
    )

    for freq in (73.0, 331.0, 1_730.0, 4_400.0):
        specs.append(
            ProbeSpec(
                probe_id=f"square_{int(freq)}hz_held",
                kind=KIND_SQUARE,
                tier=TIER_HELD_OUT,
                frequency_hz=freq,
            )
        )
    for freq in (3_700.0, 14_300.0):
        specs.append(
            ProbeSpec(
                probe_id=f"tone_burst_{int(freq)}hz_held",
                kind=KIND_TONE_BURST,
                tier=TIER_HELD_OUT,
                frequency_hz=freq,
                burst_ms=10.0,
            )
        )
    specs.append(
        ProbeSpec(
            probe_id="sweep_log_30_19k_held",
            kind=KIND_SWEEP_LOG,
            tier=TIER_HELD_OUT,
            frequency_hz=30.0,
            frequency_end_hz=19_000.0,
            duration_sec=2.0,
        )
    )
    specs.append(
        ProbeSpec(
            probe_id="pink_noise_s5678_held",
            kind=KIND_PINK_NOISE,
            tier=TIER_HELD_OUT,
            seed=5678,
        )
    )
    specs.append(
        ProbeSpec(
            probe_id="imd_83hz_6311hz_held",
            kind=KIND_IMD_TWO_TONE,
            tier=TIER_HELD_OUT,
            frequency_hz=83.0,
            secondary_frequency_hz=6_311.0,
            amplitude_ratio=3.7,
            duration_sec=3.0,
        )
    )
    return tuple(specs)


def generate_probe(
    spec: ProbeSpec, sample_rate: int = DEFAULT_SOURCE_SAMPLE_RATE
) -> np.ndarray:
    """Generate the waveform for one probe specification.

    Args:
        spec: Probe specification.
        sample_rate: Source sample rate in Hz.

    Returns:
        1D float64 waveform of length duration_sec * sample_rate.

    Raises:
        ValueError: If the spec is inconsistent with its kind.

    Physical Basis:
        All probes are generated at the 44.1 kHz source rate so they follow
        the exact input path of the system under test.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    num_samples = int(round(spec.duration_sec * sample_rate))
    time_axis = np.arange(num_samples, dtype=np.float64) / sample_rate

    if spec.kind == KIND_SQUARE:
        frequency = _require(spec.frequency_hz, "frequency_hz", spec)
        wave = sp_signal.square(2.0 * np.pi * frequency * time_axis)
        return spec.amplitude * np.asarray(wave, dtype=np.float64)

    if spec.kind == KIND_DC_STEP:
        sign = float(_require(spec.step_sign, "step_sign", spec))
        wave = np.full(num_samples, -sign, dtype=np.float64)
        wave[num_samples // 2 :] = sign
        return np.asarray(spec.amplitude * wave, dtype=np.float64)

    if spec.kind == KIND_IMPULSE:
        wave = np.zeros(num_samples, dtype=np.float64)
        wave[num_samples // 2] = spec.amplitude
        return np.asarray(wave, dtype=np.float64)

    if spec.kind == KIND_IMPULSE_TRAIN:
        period_ms = _require(spec.period_ms, "period_ms", spec)
        period = max(1, int(round(period_ms * sample_rate / 1_000.0)))
        wave = np.zeros(num_samples, dtype=np.float64)
        wave[::period] = spec.amplitude
        return np.asarray(wave, dtype=np.float64)

    if spec.kind == KIND_TONE_BURST:
        frequency = _require(spec.frequency_hz, "frequency_hz", spec)
        burst_ms = _require(spec.burst_ms, "burst_ms", spec)
        burst_len = max(3, int(round(burst_ms * sample_rate / 1_000.0)))
        wave = np.zeros(num_samples, dtype=np.float64)
        start = (num_samples - burst_len) // 2
        local_time = np.arange(burst_len, dtype=np.float64) / sample_rate
        burst = np.sin(2.0 * np.pi * frequency * local_time) * np.hanning(burst_len)
        wave[start : start + burst_len] = spec.amplitude * burst
        return np.asarray(wave, dtype=np.float64)

    if spec.kind == KIND_SWEEP_LOG:
        start_hz = _require(spec.frequency_hz, "frequency_hz", spec)
        end_hz = _require(spec.frequency_end_hz, "frequency_end_hz", spec)
        wave = sp_signal.chirp(
            time_axis,
            f0=start_hz,
            f1=end_hz,
            t1=spec.duration_sec,
            method="logarithmic",
        )
        return spec.amplitude * np.asarray(wave, dtype=np.float64)

    if spec.kind == KIND_PINK_NOISE:
        seed = int(_require(spec.seed, "seed", spec))
        rng = np.random.default_rng(seed)
        white = rng.standard_normal(num_samples)
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
        weights = np.ones_like(freqs)
        weights[1:] = 1.0 / np.sqrt(freqs[1:])
        weights[0] = 0.0
        pink = np.asarray(
            np.fft.irfft(spectrum * weights, n=num_samples), dtype=np.float64
        )
        return np.asarray(
            spec.amplitude * pink / np.max(np.abs(pink)), dtype=np.float64
        )

    if spec.kind == KIND_MULTITONE:
        seed = int(_require(spec.seed, "seed", spec))
        rng = np.random.default_rng(seed)
        tones = np.zeros(num_samples, dtype=np.float64)
        for frequency in np.geomspace(100.0, 20_000.0, 60):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            tones += np.sin(2.0 * np.pi * frequency * time_axis + phase)
        return np.asarray(
            spec.amplitude * tones / np.max(np.abs(tones)), dtype=np.float64
        )

    if spec.kind == KIND_IMD_TWO_TONE:
        low_hz = _require(spec.frequency_hz, "frequency_hz", spec)
        high_hz = _require(spec.secondary_frequency_hz, "secondary_frequency_hz", spec)
        ratio = _require(spec.amplitude_ratio, "amplitude_ratio", spec)
        if low_hz >= high_hz:
            raise ValueError("IMD low frequency must be below the high frequency.")
        if ratio <= 0.0:
            raise ValueError("IMD amplitude_ratio must be positive.")
        low_amplitude = spec.amplitude * ratio / (ratio + 1.0)
        high_amplitude = spec.amplitude / (ratio + 1.0)
        return np.asarray(
            low_amplitude * np.sin(2.0 * np.pi * low_hz * time_axis)
            + high_amplitude * np.sin(2.0 * np.pi * high_hz * time_axis),
            dtype=np.float64,
        )

    raise ValueError(f"Unhandled probe kind: {spec.kind}")


def suite_manifest(
    suite: tuple[ProbeSpec, ...],
    source_sample_rate: int = DEFAULT_SOURCE_SAMPLE_RATE,
) -> dict[str, object]:
    """Serialize a probe suite into a versioned manifest dictionary.

    The probe frequency list is rate-family independent (frequencies are
    absolute), so a 48 kHz manifest reuses the same specs and differs only
    in source_sample_rate (and therefore in manifest hash).
    """
    return {
        "version": MANIFEST_VERSION,
        "source_sample_rate": source_sample_rate,
        "probes": [asdict(spec) for spec in suite],
    }


def suite_from_manifest(manifest: dict[str, object]) -> tuple[ProbeSpec, ...]:
    """Reconstruct a probe suite from a manifest dictionary.

    Raises:
        ValueError: If the manifest version is unsupported.
    """
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")
    probes = manifest.get("probes")
    if not isinstance(probes, list):
        raise ValueError("Manifest is missing the probes list.")
    return tuple(ProbeSpec(**entry) for entry in probes)


def manifest_hash(manifest: dict[str, object]) -> str:
    """Return a stable short hash identifying the manifest contents."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def save_manifest(
    suite: tuple[ProbeSpec, ...],
    path: Path,
    source_sample_rate: int = DEFAULT_SOURCE_SAMPLE_RATE,
) -> None:
    """Write the suite manifest as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(suite_manifest(suite, source_sample_rate), indent=2) + "\n"
    )


def load_manifest(path: Path) -> tuple[ProbeSpec, ...]:
    """Load a probe suite from a manifest file.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: If the manifest is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Probe manifest not found: {path}")
    return suite_from_manifest(json.loads(path.read_text()))


def _require(value: float | int | None, name: str, spec: ProbeSpec) -> float | int:
    if value is None:
        raise ValueError(f"Probe '{spec.probe_id}' requires {name}.")
    return value
