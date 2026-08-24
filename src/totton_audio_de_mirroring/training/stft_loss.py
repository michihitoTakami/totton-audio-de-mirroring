"""Multi-resolution spectral fidelity loss used by CAPB training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class STFTLossConfig:
    """Configure one STFT resolution.

    Args:
        n_fft: FFT size.
        hop_length: Frame hop in samples.
        win_length: Hann window length in samples.
        center: Whether to center frames with padding.

    Physical Basis:
        Multiple resolutions constrain both short transients and sustained
        spectral error while the fixed FIR bank retains the safety contract.
    """

    n_fft: int
    hop_length: int
    win_length: int
    center: bool = True

    def __post_init__(self) -> None:
        if min(self.n_fft, self.hop_length, self.win_length) <= 0:
            raise ValueError("STFT sizes must be positive.")
        if self.win_length > self.n_fft:
            raise ValueError("win_length must be <= n_fft.")
        if self.hop_length > self.win_length:
            raise ValueError("hop_length must be <= win_length.")


def multi_resolution_stft_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    configs: Sequence[STFTLossConfig],
) -> torch.Tensor:
    """Return the mean magnitude error over several STFT resolutions.

    Args:
        prediction: Predicted waveform with shape ``(batch, time)``.
        target: Target waveform with the same shape.
        configs: Non-empty STFT configuration sequence.

    Returns:
        Scalar spectral fidelity loss.

    Physical Basis:
        Magnitude error at multiple time-frequency resolutions discourages
        narrow image residue and broad response drift simultaneously.
    """
    _validate_waveform(prediction, "prediction")
    _validate_waveform(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share shape.")
    if not configs:
        raise ValueError("configs must be non-empty.")
    losses = [
        torch.mean(
            torch.abs(_stft_magnitude(prediction, item) - _stft_magnitude(target, item))
        )
        for item in configs
    ]
    return torch.stack(losses).mean()


def _stft_magnitude(signal: torch.Tensor, config: STFTLossConfig) -> torch.Tensor:
    window = torch.hann_window(
        config.win_length,
        periodic=True,
        device=signal.device,
        dtype=signal.dtype,
    )
    spectrum = torch.stft(
        signal,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=config.center,
        normalized=True,
        return_complex=True,
    )
    return torch.abs(spectrum)


def _validate_waveform(signal: torch.Tensor, name: str) -> None:
    if signal.ndim != 2 or signal.shape[-1] == 0:
        raise ValueError(f"{name} must be a non-empty (batch, time) tensor.")
