"""Safety constraints for Stage 1 high-band suppression."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as torch_f


def apply_energy_cap(magnitude: torch.Tensor, energy_cap: float) -> torch.Tensor:
    """Apply a fixed energy cap to high-band STFT magnitude.

    Args:
        magnitude: STFT magnitude tensor with shape (batch, freq, time).
        energy_cap: Maximum total energy allowed per batch sample.

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

    energy = torch.sum(magnitude**2, dim=(-2, -1), keepdim=True)
    scale = torch.sqrt(energy_cap / (energy + 1.0e-8))
    limited_scale = torch.clamp(scale, max=1.0)
    return magnitude * limited_scale


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
) -> torch.Tensor:
    """Apply Stage 1 post-network safety constraints in fixed order.

    Args:
        magnitude: STFT magnitude tensor with shape (batch, freq, time).
        envelope_target: Per-frequency decay target with shape (freq,).
        highband_mask: Binary mask for high band with shape (freq,).
        energy_cap: Maximum total energy allowed per batch sample.

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
    return apply_energy_cap(highband_only, energy_cap)


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
