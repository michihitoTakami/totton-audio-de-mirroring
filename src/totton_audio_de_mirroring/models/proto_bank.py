"""Fixed linear-phase FIR prototype bank for the CAPB Stage 1 upsampler.

CAPB (Constrained Adaptive Prototype-Blend) replaces the free-form STFT mask
with a convex, time-varying blend of a small number of fixed 2x interpolation
FIR prototypes. All prototypes are linear phase with one shared group delay,
so blending mixes their magnitudes coherently and the network's only freedom
is choosing, over time, a point on the fixed sharp-vs-gentle trade-off curve.

Phase 0 findings that shaped this design (reports/capb_phase0/):
- Any sharp spectral cut (including a shared-passband projection at 20 kHz)
  injects a Gibbs plateau-ripple floor on square probes; projections are out.
- Widening a Kaiser transition barely reduces square-edge overshoot (~4.5e-2
  at amplitude 0.5 regardless of width): removing the 19-22 kHz harmonics of
  a discontinuity always leaves that ripple. The only fixed filter with
  reference-level ringing is one that, like the Bessel reference SRC, keeps
  a gradual magnitude rolloff - hence the Bessel-magnitude-matched prototype.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

DEFAULT_SAMPLE_RATE = 88_200
DEFAULT_UPSAMPLE_RATIO = 2
DEFAULT_NORMALIZATION_FREQ_HZ = 1_000.0
DEFAULT_MATCH_BAND_HIGH_HZ = 19_000.0
DEFAULT_BESSEL_CUTOFF_HZ = 20_000.0
DEFAULT_BESSEL_ORDER = 6
_RESPONSE_FFT_SIZE = 1 << 17
_MAGNITUDE_FIT_GRID = 2049


@dataclass(frozen=True)
class KaiserPrototypeSpec:
    """Kaiser windowed-sinc prototype defined by its transition band.

    Args:
        name: Prototype identifier (e.g. "sharp").
        passband_edge_hz: Last frequency with full response, in Hz.
        stopband_edge_hz: First frequency at full attenuation, in Hz.
        attenuation_db: Kaiser design stopband attenuation in dB.

    Physical Basis:
        Transition width sets how fast square-edge Gibbs ripple decays away
        from the edge, while attenuation sets mirror-image suppression above
        the input Nyquist. Both trade against kernel support (pre/post-echo
        extent), which is why several prototypes are blended.
    """

    name: str
    passband_edge_hz: float
    stopband_edge_hz: float
    attenuation_db: float


@dataclass(frozen=True)
class BesselMagnitudePrototypeSpec:
    """Linear-phase prototype matched to a Bessel IIR magnitude response.

    Args:
        name: Prototype identifier (e.g. "gentle").
        num_taps: FIR length (odd).
        cutoff_hz: Bessel design cutoff in Hz.
        order: Bessel filter order.

    Physical Basis:
        The project's reference SRC is a Bessel IIR whose gradual magnitude
        rolloff is what keeps square plateaus ripple-free. Matching that
        magnitude with linear phase reproduces the reference's time-domain
        behavior (ripple split symmetrically pre/post edge) while remaining
        phase-coherent with the other linear-phase prototypes in the blend.
    """

    name: str
    num_taps: int
    cutoff_hz: float
    order: int


PrototypeSpecType = KaiserPrototypeSpec | BesselMagnitudePrototypeSpec

DEFAULT_PROTOTYPE_SPECS: tuple[PrototypeSpecType, ...] = (
    KaiserPrototypeSpec(
        name="sharp",
        passband_edge_hz=21_000.0,
        stopband_edge_hz=22_050.0,
        attenuation_db=90.0,
    ),
    KaiserPrototypeSpec(
        name="mid",
        passband_edge_hz=19_500.0,
        stopband_edge_hz=23_000.0,
        attenuation_db=80.0,
    ),
    BesselMagnitudePrototypeSpec(
        name="gentle",
        num_taps=101,
        cutoff_hz=DEFAULT_BESSEL_CUTOFF_HZ,
        order=DEFAULT_BESSEL_ORDER,
    ),
)


@dataclass(frozen=True)
class PrototypeBank:
    """Bank of centered, gain-matched linear-phase interpolation kernels.

    Args:
        sample_rate: Target sample rate in Hz (after upsampling).
        upsample_ratio: Integer upsampling ratio the bank interpolates.
        names: Prototype names, index-aligned with kernels.
        kernels: Kernels padded to one common odd length, shape (K, L).
        group_delay_samples: Constant group delay shared by all kernels.

    Physical Basis:
        Common length and centering give every kernel the same constant
        group delay, so a convex blend has flat group delay too, and the
        blend response magnitude is the weight-average of the prototype
        magnitudes (no phase cancellation between prototypes).
    """

    sample_rate: int
    upsample_ratio: int
    names: tuple[str, ...]
    kernels: np.ndarray
    group_delay_samples: int


def design_kaiser_prototype(
    spec: KaiserPrototypeSpec,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    upsample_ratio: int = DEFAULT_UPSAMPLE_RATIO,
    normalization_freq_hz: float = DEFAULT_NORMALIZATION_FREQ_HZ,
) -> np.ndarray:
    """Design a Kaiser windowed-sinc interpolation prototype.

    Args:
        spec: Transition-band specification.
        sample_rate: Target sample rate in Hz.
        upsample_ratio: Integer upsampling ratio (gain compensation).
        normalization_freq_hz: Frequency at which passband gain is pinned.

    Returns:
        FIR taps with passband gain of exactly upsample_ratio at
        normalization_freq_hz.

    Raises:
        ValueError: If the specification is invalid.

    Physical Basis:
        Zero-stuffing by R attenuates the baseband by 1/R, so interpolation
        kernels carry a gain of R. Pinning the gain at a low reference
        frequency removes residual design-gain error and rules out the
        systematic level loss seen in earlier zero-stuff experiments.
    """
    _validate_kaiser_spec(spec, sample_rate)
    width_hz = spec.stopband_edge_hz - spec.passband_edge_hz
    num_taps, beta = sp_signal.kaiserord(
        spec.attenuation_db, width_hz / (sample_rate / 2)
    )
    if num_taps % 2 == 0:
        num_taps += 1
    cutoff_hz = 0.5 * (spec.passband_edge_hz + spec.stopband_edge_hz)
    taps = sp_signal.firwin(
        num_taps, cutoff_hz, window=("kaiser", beta), fs=sample_rate
    ).astype(np.float64)
    return _normalize_gain(taps, sample_rate, upsample_ratio, normalization_freq_hz)


def design_bessel_magnitude_prototype(
    spec: BesselMagnitudePrototypeSpec,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    upsample_ratio: int = DEFAULT_UPSAMPLE_RATIO,
    normalization_freq_hz: float = DEFAULT_NORMALIZATION_FREQ_HZ,
) -> np.ndarray:
    """Design a linear-phase FIR matching a Bessel IIR magnitude response.

    Args:
        spec: Bessel magnitude specification.
        sample_rate: Target sample rate in Hz.
        upsample_ratio: Integer upsampling ratio (gain compensation).
        normalization_freq_hz: Frequency at which passband gain is pinned.

    Returns:
        FIR taps with passband gain of exactly upsample_ratio at
        normalization_freq_hz.

    Raises:
        ValueError: If the specification is invalid.

    Physical Basis:
        Phase 0 measurements show this kernel's square-plateau ripple is
        below the Bessel reference itself (ripple splits pre/post edge),
        making it the "no added ringing" end of the blend trade-off.
    """
    if spec.num_taps <= 0 or spec.num_taps % 2 == 0:
        raise ValueError(
            f"num_taps must be a positive odd integer, got {spec.num_taps}."
        )
    if not 0.0 < spec.cutoff_hz < sample_rate / 2:
        raise ValueError(f"cutoff_hz must be in (0, Nyquist), got {spec.cutoff_hz}.")
    if spec.order <= 0:
        raise ValueError(f"order must be positive, got {spec.order}.")

    b, a = sp_signal.bessel(
        spec.order,
        spec.cutoff_hz,
        btype="lowpass",
        analog=False,
        output="ba",
        norm="phase",
        fs=sample_rate,
    )
    freq_grid = np.linspace(0.0, sample_rate / 2, _MAGNITUDE_FIT_GRID)
    _, response = sp_signal.freqz(b, a, worN=freq_grid, fs=sample_rate)
    taps = sp_signal.firwin2(
        spec.num_taps, freq_grid, np.abs(response), fs=sample_rate
    ).astype(np.float64)
    return _normalize_gain(taps, sample_rate, upsample_ratio, normalization_freq_hz)


def build_prototype_bank(
    specs: tuple[PrototypeSpecType, ...] = DEFAULT_PROTOTYPE_SPECS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    upsample_ratio: int = DEFAULT_UPSAMPLE_RATIO,
) -> PrototypeBank:
    """Build the prototype bank with a common length and group delay.

    Args:
        specs: Prototype specifications.
        sample_rate: Target sample rate in Hz.
        upsample_ratio: Integer upsampling ratio.

    Returns:
        PrototypeBank with centered, gain-matched kernels.

    Raises:
        ValueError: If specs are empty or produce incompatible lengths.

    Physical Basis:
        Zero-padding shorter kernels symmetrically preserves their linear
        phase while aligning every prototype to one shared group delay.
    """
    if len(specs) == 0:
        raise ValueError("specs must contain at least one prototype.")

    kernels = [_design_prototype(spec, sample_rate, upsample_ratio) for spec in specs]
    common_len = max(kernel.size for kernel in kernels)
    stacked = np.stack(
        [_pad_centered(kernel, common_len) for kernel in kernels], axis=0
    )
    return PrototypeBank(
        sample_rate=sample_rate,
        upsample_ratio=upsample_ratio,
        names=tuple(spec.name for spec in specs),
        kernels=stacked,
        group_delay_samples=(common_len - 1) // 2,
    )


def blend_modulation_bounds(
    bank: PrototypeBank,
    band_edges_hz: tuple[float, ...] = (
        5_000.0,
        10_000.0,
        15_000.0,
        17_000.0,
        19_000.0,
    ),
) -> dict[str, float]:
    """Report the structural bound on blend-induced modulation per band.

    Args:
        bank: Prototype bank.
        band_edges_hz: Upper band edges to report bounds below.

    Returns:
        Mapping of "below_<hz>" to the worst pairwise response deviation in
        dB relative to the passband gain.

    Physical Basis:
        With convex weights, the blend response at any instant lies between
        the prototype responses, so time-varying weights can modulate content
        at frequency f by at most the pairwise response spread at f. This is
        the structural guarantee replacing the old hard low-band bypass.
    """
    freqs, responses = _bank_responses(bank)
    gain = float(bank.upsample_ratio)
    spread = (np.max(responses, axis=0) - np.min(responses, axis=0)) / gain

    bounds: dict[str, float] = {}
    for edge in band_edges_hz:
        in_band = freqs <= edge
        bounds[f"below_{int(edge)}"] = _db(np.max(spread[in_band]))
    return bounds


def validate_bank(
    bank: PrototypeBank,
    match_band_high_hz: float = DEFAULT_MATCH_BAND_HIGH_HZ,
    kaiser_match_tolerance_db: float = -70.0,
) -> dict[str, float]:
    """Validate structural properties of the bank.

    Args:
        bank: Prototype bank to validate.
        match_band_high_hz: Band top for the Kaiser passband-match check.
        kaiser_match_tolerance_db: Max allowed deviation between Kaiser-flat
            prototypes below match_band_high_hz.

    Returns:
        Mapping of check name to measured value in dB.

    Raises:
        ValueError: If symmetry or passband matching fails.

    Physical Basis:
        Kernel symmetry certifies linear phase (flat group delay); the
        passband match between the flat prototypes bounds what the blend can
        do to audible content when it moves between them.
    """
    results: dict[str, float] = {}

    symmetry = float(
        np.max(np.abs(bank.kernels - bank.kernels[:, ::-1]))
        / np.max(np.abs(bank.kernels))
    )
    results["kernel_symmetry_rel"] = symmetry
    if symmetry > 1e-12:
        raise ValueError(f"Kernels are not symmetric (rel err {symmetry:.2e}).")

    flat_indices = [index for index, name in enumerate(bank.names) if name != "gentle"]
    if len(flat_indices) >= 2:
        freqs, responses = _bank_responses(bank)
        in_band = freqs <= match_band_high_hz
        gain = float(bank.upsample_ratio)
        worst = 0.0
        for offset, index_a in enumerate(flat_indices):
            for index_b in flat_indices[offset + 1 :]:
                deviation = (
                    np.abs(responses[index_a][in_band] - responses[index_b][in_band])
                    / gain
                )
                worst = max(worst, float(np.max(deviation)))
        worst_db = _db(worst)
        results["kaiser_passband_match_db"] = worst_db
        if worst_db > kaiser_match_tolerance_db:
            raise ValueError(
                "Kaiser prototype passband mismatch "
                f"{worst_db:.1f} dB exceeds {kaiser_match_tolerance_db:.1f} dB."
            )
    return results


def upsample_with_kernel(
    signal: np.ndarray,
    kernel: np.ndarray,
    upsample_ratio: int = DEFAULT_UPSAMPLE_RATIO,
    compensate_delay: bool = True,
) -> np.ndarray:
    """Upsample a signal with one fixed prototype kernel.

    Args:
        signal: Input signal (1D), at the source rate.
        kernel: Prototype taps (1D, odd length).
        upsample_ratio: Integer upsampling ratio.
        compensate_delay: Trim the kernel group delay so the output aligns
            with the zero-stuffed input timeline.

    Returns:
        Upsampled signal of length len(signal) * upsample_ratio.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Zero-stuffing creates spectral images of the baseband; convolving
        with the interpolation kernel attenuates the images according to the
        kernel's stopband. Linear phase makes the delay an exact constant.
    """
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("signal must be a non-empty 1D array.")
    if kernel.ndim != 1 or kernel.size % 2 == 0:
        raise ValueError("kernel must be a 1D array of odd length.")
    if upsample_ratio <= 0:
        raise ValueError(f"upsample_ratio must be positive, got {upsample_ratio}.")

    stuffed = np.zeros(signal.size * upsample_ratio, dtype=np.float64)
    stuffed[::upsample_ratio] = np.asarray(signal, dtype=np.float64)
    full = np.asarray(
        sp_signal.fftconvolve(stuffed, np.asarray(kernel, dtype=np.float64)),
        dtype=np.float64,
    )
    if not compensate_delay:
        return full[: stuffed.size]
    delay = (kernel.size - 1) // 2
    return full[delay : delay + stuffed.size]


def summarize_bank(bank: PrototypeBank) -> dict[str, dict[str, float]]:
    """Summarize per-prototype frequency-domain properties.

    Args:
        bank: Prototype bank to summarize.

    Returns:
        Mapping of prototype name to response statistics in dB.

    Physical Basis:
        The image band of a 2x upsampler from 44.1 kHz starts at 22.05 kHz;
        response depth there predicts mirror suppression, while passband
        deviation below 19 kHz predicts audible-band flatness.
    """
    freqs, responses = _bank_responses(bank)
    gain = float(bank.upsample_ratio)
    passband = freqs <= DEFAULT_MATCH_BAND_HIGH_HZ
    image_band = freqs >= 22_050.0 + 500.0
    deep_image = freqs >= 24_000.0

    summary: dict[str, dict[str, float]] = {}
    for name, response in zip(bank.names, responses, strict=True):
        rel = response / gain
        summary[name] = {
            "passband_dev_db": _db(np.max(np.abs(rel[passband] - 1.0))),
            "image_band_max_db": _db(np.max(rel[image_band])),
            "deep_image_max_db": _db(np.max(rel[deep_image])),
            "response_20k_db": _db(_response_at_freq(freqs, rel, 20_000.0)),
        }
    return summary


def _design_prototype(
    spec: PrototypeSpecType, sample_rate: int, upsample_ratio: int
) -> np.ndarray:
    if isinstance(spec, KaiserPrototypeSpec):
        return design_kaiser_prototype(spec, sample_rate, upsample_ratio)
    return design_bessel_magnitude_prototype(spec, sample_rate, upsample_ratio)


def _bank_responses(bank: PrototypeBank) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(_RESPONSE_FFT_SIZE, d=1.0 / bank.sample_rate)
    responses = np.abs(np.fft.rfft(bank.kernels, n=_RESPONSE_FFT_SIZE, axis=-1))
    return freqs, responses


def _normalize_gain(
    taps: np.ndarray, sample_rate: int, upsample_ratio: int, freq_hz: float
) -> np.ndarray:
    _, response = sp_signal.freqz(taps, worN=[freq_hz], fs=sample_rate)
    gain = float(np.abs(response[0]))
    if gain <= 0.0:
        raise ValueError("Prototype has non-positive passband gain.")
    return taps * (float(upsample_ratio) / gain)


def _response_at_freq(freqs: np.ndarray, response: np.ndarray, freq_hz: float) -> float:
    index = int(np.argmin(np.abs(freqs - freq_hz)))
    return float(response[index])


def _pad_centered(taps: np.ndarray, length: int) -> np.ndarray:
    if taps.size > length:
        raise ValueError(f"taps ({taps.size}) longer than target length {length}.")
    if (length - taps.size) % 2 != 0:
        raise ValueError("Length difference must be even to keep taps centered.")
    pad = (length - taps.size) // 2
    return np.pad(taps, (pad, pad))


def _db(value: float | np.floating) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-300)))


def _validate_kaiser_spec(spec: KaiserPrototypeSpec, sample_rate: int) -> None:
    nyquist = sample_rate / 2
    if not 0.0 < spec.passband_edge_hz < spec.stopband_edge_hz:
        raise ValueError(
            f"Prototype '{spec.name}' requires 0 < passband_edge_hz < "
            f"stopband_edge_hz, got {spec.passband_edge_hz}/{spec.stopband_edge_hz}."
        )
    if spec.stopband_edge_hz >= nyquist:
        raise ValueError(
            f"Prototype '{spec.name}' stopband_edge_hz must be below Nyquist "
            f"({nyquist} Hz), got {spec.stopband_edge_hz}."
        )
    if spec.attenuation_db <= 0.0:
        raise ValueError(f"Prototype '{spec.name}' attenuation_db must be positive.")
