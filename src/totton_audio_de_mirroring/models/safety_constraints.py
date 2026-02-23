"""Safety constraints for Stage 1 high-band suppression."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as torch_f


def apply_energy_cap(
    magnitude: torch.Tensor,
    energy_cap: float,
    *,
    highband_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a fixed energy cap to high-band STFT magnitude.

    Args:
        magnitude: STFT magnitude tensor with shape (batch, freq, time).
        energy_cap: Maximum mean high-band energy allowed per batch sample.
        highband_mask: Optional (freq,) mask indicating high-band bins.

    Returns:
        Magnitude scaled to satisfy the energy cap.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Limiting high-band total energy reduces intermodulation risk in
        nonlinear analog stages while preserving relative spectral shape.
    """
    if not math.isfinite(energy_cap) or energy_cap <= 0:
        raise ValueError(
            f"energy_cap must be a finite positive value, got {energy_cap}."
        )
    if magnitude.ndim != 3:
        raise ValueError("magnitude must be 3D (batch, freq, time).")

    mag_sq = magnitude**2
    if highband_mask is not None:
        if highband_mask.ndim != 1:
            raise ValueError("highband_mask must be 1D (freq,).")
        if highband_mask.shape[0] != magnitude.shape[1]:
            raise ValueError(
                "highband_mask length must match magnitude frequency bins. "
                f"Expected {magnitude.shape[1]}, got {highband_mask.shape[0]}."
            )
        mask = highband_mask.to(device=magnitude.device, dtype=torch.bool)
        if torch.count_nonzero(mask) == 0:
            return magnitude
        mag_sq = mag_sq[:, mask, :]

    # Mean energy keeps the cap stable across STFT grid sizes.
    energy = torch.mean(mag_sq, dim=(-2, -1), keepdim=True)
    scale = torch.sqrt(energy_cap / (energy + 1.0e-8))
    limited_scale = torch.clamp(scale, max=1.0)
    return magnitude * limited_scale


def apply_frequency_dependent_energy_cap(
    magnitude: torch.Tensor,
    energy_cap_profile: torch.Tensor,
    *,
    highband_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply per-frequency mean-energy cap in STFT magnitude domain.

    Args:
        magnitude: STFT magnitude tensor with shape (batch, freq, time).
        energy_cap_profile: Per-frequency cap for mean(mag^2), shape (freq,).
        highband_mask: Optional (freq,) mask indicating high-band bins.

    Returns:
        Magnitude scaled per frequency bin to satisfy the cap profile.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Frequency-dependent caps suppress ultrasonic peak build-up above
        20kHz while allowing smoother decay toward Nyquist.
    """
    if magnitude.ndim != 3:
        raise ValueError("magnitude must be 3D (batch, freq, time).")
    if energy_cap_profile.ndim != 1:
        raise ValueError("energy_cap_profile must be 1D (freq,).")
    if energy_cap_profile.shape[0] != magnitude.shape[1]:
        raise ValueError(
            "energy_cap_profile length must match magnitude frequency bins. "
            f"Expected {magnitude.shape[1]}, got {energy_cap_profile.shape[0]}."
        )
    if torch.any(~torch.isfinite(energy_cap_profile)):
        raise ValueError("energy_cap_profile must be finite.")
    if torch.any(energy_cap_profile <= 0):
        raise ValueError("energy_cap_profile must be positive.")

    if highband_mask is not None:
        if highband_mask.ndim != 1:
            raise ValueError("highband_mask must be 1D (freq,).")
        if highband_mask.shape[0] != magnitude.shape[1]:
            raise ValueError(
                "highband_mask length must match magnitude frequency bins. "
                f"Expected {magnitude.shape[1]}, got {highband_mask.shape[0]}."
            )
        mask = highband_mask.to(device=magnitude.device, dtype=torch.bool)
    else:
        mask = torch.ones(
            magnitude.shape[1],
            dtype=torch.bool,
            device=magnitude.device,
        )

    if torch.count_nonzero(mask) == 0:
        return magnitude

    cap = energy_cap_profile.to(device=magnitude.device, dtype=magnitude.dtype)
    mag_sq = magnitude**2
    freq_energy = torch.mean(mag_sq, dim=-1)
    safe_cap = torch.clamp(cap, min=1.0e-12)
    scale = torch.sqrt(safe_cap.unsqueeze(0) / (freq_energy + 1.0e-8))
    limited_scale = torch.clamp(scale, max=1.0)
    if torch.count_nonzero(~mask) > 0:
        limited_scale[:, ~mask] = 1.0
    return magnitude * limited_scale.unsqueeze(-1)


def apply_envelope_target(
    magnitude: torch.Tensor,
    envelope_target: torch.Tensor,
    highband_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply fixed envelope shaping only in the high band.

    Args:
        magnitude: STFT magnitude tensor with shape (batch, freq, time).
        envelope_target: Per-frequency decay target with shape (freq,).
        highband_mask: Binary mask for high band with shape (freq,).

    Returns:
        Envelope-shaped magnitude while preserving low-band bins.

    Raises:
        ValueError: If tensor dimensions are invalid.

    Physical Basis:
        A fixed post-network envelope suppresses sharp ultrasonic peaks
        while leaving the 0-20kHz band unchanged by construction.
    """
    _validate_freq_domain_inputs(magnitude, envelope_target, highband_mask)
    lowband_mask = 1.0 - highband_mask
    gain = lowband_mask + (highband_mask * envelope_target)
    return magnitude * gain[None, :, None]


def apply_highband_mask(
    magnitude: torch.Tensor,
    highband_mask: torch.Tensor,
) -> torch.Tensor:
    """Zero low-band bins to prevent leakage from HB processing path.

    Args:
        magnitude: STFT magnitude tensor with shape (batch, freq, time).
        highband_mask: Binary mask for high band with shape (freq,).

    Returns:
        Magnitude with low-band bins suppressed.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Explicitly masking low-band bins enforces structural separation:
        high-band branch cannot inject content below the crossover.
    """
    if magnitude.ndim != 3:
        raise ValueError("magnitude must be 3D (batch, freq, time).")
    if highband_mask.ndim != 1:
        raise ValueError("highband_mask must be 1D (freq,).")
    if highband_mask.shape[0] != magnitude.shape[1]:
        raise ValueError(
            "highband_mask length must match magnitude frequency bins. "
            f"Expected {magnitude.shape[1]}, got {highband_mask.shape[0]}."
        )
    return magnitude * highband_mask[None, :, None]


def apply_safety_constraints(
    magnitude: torch.Tensor,
    envelope_target: torch.Tensor,
    highband_mask: torch.Tensor,
    energy_cap: float,
    energy_cap_profile: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply Stage 1 post-network safety constraints in fixed order.

    Args:
        magnitude: STFT magnitude tensor with shape (batch, freq, time).
        envelope_target: Per-frequency decay target with shape (freq,).
        highband_mask: Binary mask for high band with shape (freq,).
        energy_cap: Maximum total energy allowed per batch sample.
        energy_cap_profile: Optional per-frequency mean-energy caps.

    Returns:
        Magnitude after envelope shaping, high-band masking, and cap.

    Raises:
        ValueError: If any input is invalid.

    Physical Basis:
        Fixed post-processing enforces anti-hallucination constraints and
        IMD-safe high-band energy limits independent of model behavior.
    """
    shaped = apply_envelope_target(magnitude, envelope_target, highband_mask)
    highband_only = apply_highband_mask(shaped, highband_mask)
    if energy_cap_profile is not None:
        highband_only = apply_frequency_dependent_energy_cap(
            highband_only,
            energy_cap_profile=energy_cap_profile,
            highband_mask=highband_mask,
        )
    return apply_energy_cap(highband_only, energy_cap, highband_mask=highband_mask)


def build_envelope_target(
    num_freqs: int,
    sample_rate: int,
    cutoff_hz: float,
    floor: float,
) -> torch.Tensor:
    """Build a monotonic high-band decay envelope.

    Args:
        num_freqs: Number of STFT frequency bins.
        sample_rate: Sample rate in Hz.
        cutoff_hz: Crossover frequency in Hz.
        floor: Minimum gain at Nyquist in [0.0, 1.0].

    Returns:
        Envelope vector with shape (num_freqs,).

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Gradual decay above cutoff avoids excessive ultrasonic emphasis
        and keeps shaping aligned with anti-hallucination constraints.
    """
    if num_freqs <= 0:
        raise ValueError(f"num_freqs must be positive, got {num_freqs}.")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}.")
    nyquist = sample_rate / 2
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz must be less than Nyquist ({nyquist}).")
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}.")

    freqs = torch.linspace(0.0, nyquist, num_freqs)
    envelope = torch.ones_like(freqs)
    high = freqs >= cutoff_hz
    if torch.any(high):
        decay = (freqs[high] - cutoff_hz) / (nyquist - cutoff_hz)
        envelope[high] = torch.clamp(1.0 - decay, min=floor)
    return envelope


def build_highband_mask(
    num_freqs: int,
    sample_rate: int,
    cutoff_hz: float,
) -> torch.Tensor:
    """Build a binary mask for high-band frequency bins.

    Args:
        num_freqs: Number of STFT frequency bins.
        sample_rate: Sample rate in Hz.
        cutoff_hz: Crossover frequency in Hz.

    Returns:
        Float32 mask with shape (num_freqs,), values in {0, 1}.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        High-band-only masking preserves low-band identity by preventing
        any post-network shaping below the 20kHz crossover.
    """
    if num_freqs <= 0:
        raise ValueError(f"num_freqs must be positive, got {num_freqs}.")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}.")
    nyquist = sample_rate / 2
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz must be less than Nyquist ({nyquist}).")

    freqs = torch.linspace(0.0, nyquist, num_freqs)
    return (freqs >= cutoff_hz).to(dtype=torch.float32)


def build_energy_cap_profile(
    num_freqs: int,
    sample_rate: int,
    *,
    start_hz: float,
    cap_start: float,
    cap_floor_ratio: float,
) -> torch.Tensor:
    """Build monotonic per-frequency high-band energy-cap profile.

    Args:
        num_freqs: Number of STFT frequency bins.
        sample_rate: Sample rate in Hz.
        start_hz: Frequency where decay starts.
        cap_start: Cap value at start_hz.
        cap_floor_ratio: Floor ratio at Nyquist in (0, 1].

    Returns:
        Per-frequency cap profile with shape (num_freqs,).

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        A smooth cap profile limits excessive ultrasonic energy growth
        while preserving gradual decay characteristics from 20kHz upward.
    """
    if num_freqs <= 0:
        raise ValueError(f"num_freqs must be positive, got {num_freqs}.")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if start_hz <= 0:
        raise ValueError(f"start_hz must be positive, got {start_hz}.")
    if cap_start <= 0:
        raise ValueError(f"cap_start must be positive, got {cap_start}.")
    if not 0.0 < cap_floor_ratio <= 1.0:
        raise ValueError(f"cap_floor_ratio must be in (0, 1], got {cap_floor_ratio}.")
    nyquist = sample_rate / 2
    if start_hz >= nyquist:
        raise ValueError(f"start_hz must be less than Nyquist ({nyquist}).")

    freqs = torch.linspace(0.0, nyquist, num_freqs)
    profile = torch.full((num_freqs,), cap_start, dtype=torch.float32)
    high = freqs >= start_hz
    if torch.any(high):
        decay = (freqs[high] - start_hz) / max(nyquist - start_hz, 1.0e-6)
        floor_cap = cap_start * cap_floor_ratio
        profile[high] = torch.clamp(
            cap_start - (cap_start - floor_cap) * decay,
            min=floor_cap,
            max=cap_start,
        )
    return profile


def enforce_highpass_dc_block(
    signal: torch.Tensor,
    highpass_taps: torch.Tensor,
) -> torch.Tensor:
    """Re-apply high-pass filtering to suppress DC/leakage in HB output.

    Args:
        signal: Time-domain high-band signal (batch, time) or
            (batch, channels, time).
        highpass_taps: 1D high-pass FIR taps with odd length.

    Returns:
        High-pass filtered signal with the same shape as input.

    Raises:
        ValueError: If signal or taps are invalid.

    Physical Basis:
        A final HPF pass removes residual low-frequency leakage and DC
        components introduced by finite-window STFT reconstruction.
    """
    if signal.ndim not in (2, 3):
        raise ValueError(f"signal must be 2D or 3D, got {signal.ndim}D.")
    if signal.numel() == 0:
        raise ValueError("signal cannot be empty.")
    if highpass_taps.ndim != 1:
        raise ValueError("highpass_taps must be 1D.")
    if highpass_taps.numel() == 0:
        raise ValueError("highpass_taps cannot be empty.")
    if highpass_taps.numel() % 2 == 0:
        raise ValueError("highpass_taps length must be odd.")

    if signal.ndim == 2:
        signal_for_conv = signal.unsqueeze(1)
        squeeze = True
    else:
        signal_for_conv = signal
        squeeze = False

    taps = highpass_taps.flip(0).to(device=signal.device, dtype=signal.dtype)
    channels = signal_for_conv.shape[1]
    weight = taps.view(1, 1, -1).repeat(channels, 1, 1)
    padding = taps.numel() // 2
    filtered = torch_f.conv1d(
        signal_for_conv,
        weight,
        padding=padding,
        groups=channels,
    )
    return filtered.squeeze(1) if squeeze else filtered


def _validate_freq_domain_inputs(
    magnitude: torch.Tensor,
    envelope_target: torch.Tensor,
    highband_mask: torch.Tensor,
) -> None:
    if magnitude.ndim != 3:
        raise ValueError("magnitude must be 3D (batch, freq, time).")
    if envelope_target.ndim != 1:
        raise ValueError("envelope_target must be 1D (freq,).")
    if highband_mask.ndim != 1:
        raise ValueError("highband_mask must be 1D (freq,).")
    freq_bins = magnitude.shape[1]
    if envelope_target.shape[0] != freq_bins:
        raise ValueError(
            "envelope_target length must match magnitude frequency bins. "
            f"Expected {freq_bins}, got {envelope_target.shape[0]}."
        )
    if highband_mask.shape[0] != freq_bins:
        raise ValueError(
            "highband_mask length must match magnitude frequency bins. "
            f"Expected {freq_bins}, got {highband_mask.shape[0]}."
        )
