"""Neural Mirror Suppression Engine (NMSE) implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn.functional as torch_f
from torch import nn

from totton_audio_de_mirroring.models.unet import UNet2D


@dataclass(frozen=True)
class STFTConfig:
    """Configuration for STFT/iSTFT operations.

    Args:
        n_fft: FFT size.
        hop_length: Hop length between frames.
        win_length: Window length.
        center: Whether to pad input so frames are centered.

    Physical Basis:
        STFT settings define the time-frequency resolution used to
        detect mirror patterns while preserving time-domain fidelity.
    """

    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    center: bool = True


class NMSE(nn.Module):
    """Neural Mirror Suppression Engine for HB processing.

    Args:
        sample_rate: Sample rate of the input signal in Hz.
        cutoff_hz: Crossover frequency between LB and HB in Hz.
        stft_config: STFT configuration.
        unet: Optional UNet2D instance. If None, a default UNet2D is created.
        energy_cap: Maximum energy allowed in 20–44kHz band.
        envelope_floor: Minimum envelope value at Nyquist (0.0 to 1.0).
        lowpass_taps: FIR taps for low-band extraction.
        highpass_taps: FIR taps for high-band extraction.

    Physical Basis:
        NMSE suppresses mirror/aliasing artifacts in the 20–44kHz band by
        estimating a magnitude mask in STFT space. Low-band content is
        bypassed to guarantee 0–20kHz preservation by structure.
    """

    def __init__(
        self,
        sample_rate: int,
        cutoff_hz: float = 20_000.0,
        stft_config: STFTConfig | None = None,
        unet: UNet2D | None = None,
        envelope_floor: float = 0.0,
        *,
        energy_cap: float,
        lowpass_taps: np.ndarray,
        highpass_taps: np.ndarray,
    ) -> None:
        super().__init__()
        self._validate_sample_rate(sample_rate)
        self._validate_cutoff(cutoff_hz, sample_rate)
        self._validate_envelope_floor(envelope_floor)
        self._validate_energy_cap(energy_cap)

        self.sample_rate = sample_rate
        self.cutoff_hz = float(cutoff_hz)
        self.stft_config = stft_config or STFTConfig()
        self._validate_stft_config(self.stft_config)
        self.energy_cap = energy_cap
        self.lowpass_taps: torch.Tensor
        self.highpass_taps: torch.Tensor
        self.envelope_target: torch.Tensor
        self.highband_mask: torch.Tensor
        self.window: torch.Tensor

        self.unet = unet or UNet2D()
        self.register_buffer(
            "window",
            torch.hann_window(self.stft_config.win_length, periodic=True),
        )

        freq_bins = self.stft_config.n_fft // 2 + 1
        envelope = _build_envelope_target(
            num_freqs=freq_bins,
            sample_rate=sample_rate,
            cutoff_hz=self.cutoff_hz,
            floor=envelope_floor,
        )
        hb_mask = _build_highband_mask(
            num_freqs=freq_bins,
            sample_rate=sample_rate,
            cutoff_hz=self.cutoff_hz,
        )
        self.register_buffer("envelope_target", envelope)
        self.register_buffer("highband_mask", hb_mask)

        self._validate_fir_taps(lowpass_taps, "lowpass_taps")
        self._validate_fir_taps(highpass_taps, "highpass_taps")
        self.register_buffer(
            "lowpass_taps",
            torch.tensor(lowpass_taps, dtype=torch.float32),
        )
        self.register_buffer(
            "highpass_taps",
            torch.tensor(highpass_taps, dtype=torch.float32),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Process input signal with NMSE.

        Args:
            signal: Input signal (batch, time) or (batch, channels, time).

        Returns:
            Output signal with HB mirror artifacts suppressed.

        Raises:
            ValueError: If input is invalid or band-split taps are missing.

        Physical Basis:
            Low-band is preserved by bypass, while high-band is suppressed
            through STFT mask estimation and safety constraints.
        """
        self._validate_signal(signal)
        lb = _apply_fir_filter(signal, self.lowpass_taps)
        hb = _apply_fir_filter(signal, self.highpass_taps)
        hb_out = self._process_highband(hb)
        return lb + hb_out

    def forward_highband(self, high_band: torch.Tensor) -> torch.Tensor:
        """Process only the high-band signal.

        Args:
            high_band: High-band signal (batch, time) or (batch, channels, time).

        Returns:
            Processed high-band signal.

        Physical Basis:
            Masking in STFT space suppresses mirror artifacts without
            introducing new frequencies.
        """
        self._validate_signal(high_band)
        return self._process_highband(high_band)

    def _process_highband(self, high_band: torch.Tensor) -> torch.Tensor:
        """Run STFT masking and reconstruction on high-band signal.

        Args:
            high_band: High-band signal (batch, time) or (batch, channels, time).

        Returns:
            Time-domain high-band signal after suppression.

        Physical Basis:
            Applying a bounded mask in STFT magnitude space suppresses
            mirror artifacts without creating new frequency content.
        """
        flattened, restore_shape = _flatten_signal(high_band)
        stft = self._stft(flattened)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)

        magnitude_4d = magnitude.unsqueeze(1)
        padded, pad_f, pad_t = _pad_to_multiple(
            magnitude_4d,
            multiple=2**self._num_downsamples,
        )
        mask = self.unet(padded)
        if mask.shape[1] != 1:
            raise ValueError("Mask output must have a single channel.")
        mask = torch.clamp(mask, 0.0, 1.0)
        mask = _crop_to_shape(mask, pad_f, pad_t)

        masked_mag = magnitude * mask.squeeze(1)
        envelope = cast(torch.Tensor, self.envelope_target)
        highband = cast(torch.Tensor, self.highband_mask)
        masked_mag = masked_mag * envelope[:, None]
        masked_mag = masked_mag * highband[:, None]
        if self.energy_cap is not None:
            masked_mag = apply_energy_cap(masked_mag, self.energy_cap)

        complex_spec = masked_mag * torch.exp(1j * phase)
        time_signal = self._istft(complex_spec, length=high_band.shape[-1])
        time_signal = _apply_fir_filter(time_signal, self.highpass_taps)
        if restore_shape is None:
            return time_signal
        batch, channels, time = restore_shape
        return time_signal.reshape(batch, channels, time)

    @property
    def _num_downsamples(self) -> int:
        return len(self.unet.down_blocks)

    def _stft(self, signal: torch.Tensor) -> torch.Tensor:
        """Compute STFT for 2D input signal.

        Args:
            signal: Input signal (batch, time).

        Returns:
            Complex STFT tensor (batch, freq, time).

        Physical Basis:
            STFT provides localized time-frequency representation to
            detect structured mirror patterns while preserving phase.
        """
        signal_2d = _ensure_2d(signal)
        window = cast(torch.Tensor, self.window).to(
            device=signal_2d.device,
            dtype=signal_2d.dtype,
        )
        return torch.stft(
            signal_2d,
            n_fft=self.stft_config.n_fft,
            hop_length=self.stft_config.hop_length,
            win_length=self.stft_config.win_length,
            window=window,
            center=self.stft_config.center,
            return_complex=True,
        )

    def _istft(self, stft: torch.Tensor, length: int) -> torch.Tensor:
        """Compute inverse STFT for complex spectrogram.

        Args:
            stft: Complex STFT tensor (batch, freq, time).
            length: Output signal length.

        Returns:
            Time-domain signal (batch, time).

        Physical Basis:
            iSTFT reconstructs time-domain waveform while retaining the
            original phase structure needed for transient preservation.
        """
        window = cast(torch.Tensor, self.window).to(
            device=stft.device,
            dtype=stft.real.dtype,
        )
        time_signal = torch.istft(
            stft,
            n_fft=self.stft_config.n_fft,
            hop_length=self.stft_config.hop_length,
            win_length=self.stft_config.win_length,
            window=window,
            center=self.stft_config.center,
            length=length,
        )
        if time_signal.ndim == 2:
            return time_signal
        raise RuntimeError("iSTFT output has unexpected dimensionality.")

    @staticmethod
    def _validate_sample_rate(sample_rate: int) -> None:
        """Validate sample rate.

        Args:
            sample_rate: Sample rate in Hz.

        Physical Basis:
            Valid sample rates define Nyquist limit for mirror suppression.
        """
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}.")

    @staticmethod
    def _validate_cutoff(cutoff_hz: float, sample_rate: int) -> None:
        """Validate cutoff frequency.

        Args:
            cutoff_hz: Cutoff frequency in Hz.
            sample_rate: Sample rate in Hz.

        Physical Basis:
            Cutoff must remain below Nyquist to avoid invalid band splits.
        """
        if cutoff_hz <= 0:
            raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}.")
        nyquist = sample_rate / 2
        if cutoff_hz >= nyquist:
            raise ValueError(
                f"cutoff_hz must be less than Nyquist ({nyquist} Hz), got {cutoff_hz}."
            )

    @staticmethod
    def _validate_envelope_floor(envelope_floor: float) -> None:
        """Validate envelope floor value.

        Args:
            envelope_floor: Minimum envelope value at Nyquist.

        Physical Basis:
            Ensuring floor within [0,1] maintains bounded magnitude shaping.
        """
        if not 0.0 <= envelope_floor <= 1.0:
            raise ValueError(
                f"envelope_floor must be within [0, 1], got {envelope_floor}."
            )

    @staticmethod
    def _validate_energy_cap(energy_cap: float) -> None:
        """Validate energy cap value.

        Args:
            energy_cap: Maximum allowed energy.

        Physical Basis:
            Energy cap must be positive to enforce high-band safety.
        """
        if energy_cap <= 0:
            raise ValueError(f"energy_cap must be positive, got {energy_cap}.")

    @staticmethod
    def _validate_stft_config(stft_config: STFTConfig) -> None:
        """Validate STFT configuration.

        Args:
            stft_config: STFT configuration.

        Physical Basis:
            STFT parameters must be positive and internally consistent
            to preserve time-frequency alignment.
        """
        if stft_config.n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {stft_config.n_fft}.")
        if stft_config.hop_length <= 0:
            raise ValueError(
                f"hop_length must be positive, got {stft_config.hop_length}."
            )
        if stft_config.win_length <= 0:
            raise ValueError(
                f"win_length must be positive, got {stft_config.win_length}."
            )
        if stft_config.win_length > stft_config.n_fft:
            raise ValueError(
                "win_length must be less than or equal to n_fft, got "
                f"{stft_config.win_length} > {stft_config.n_fft}."
            )
        if stft_config.hop_length > stft_config.win_length:
            raise ValueError(
                "hop_length must be less than or equal to win_length, got "
                f"{stft_config.hop_length} > {stft_config.win_length}."
            )

    @staticmethod
    def _validate_signal(signal: torch.Tensor) -> None:
        """Validate input signal tensor.

        Args:
            signal: Input tensor.

        Physical Basis:
            Valid tensor shapes are required for stable STFT and filtering.
        """
        if not torch.is_tensor(signal):
            raise ValueError("signal must be a torch.Tensor.")
        if signal.ndim not in (2, 3):
            raise ValueError(f"signal must be 2D or 3D, got {signal.ndim}D.")
        if signal.numel() == 0:
            raise ValueError("signal cannot be empty.")

    @staticmethod
    def _validate_fir_taps(taps: np.ndarray, name: str) -> None:
        """Validate FIR tap array.

        Args:
            taps: FIR taps.
            name: Parameter name for error messages.

        Physical Basis:
            Odd-length FIR taps maintain linear-phase band splitting.
        """
        if taps.ndim != 1:
            raise ValueError(f"{name} must be a 1D array.")
        if taps.size == 0:
            raise ValueError(f"{name} cannot be empty.")
        if taps.size % 2 == 0:
            raise ValueError(f"{name} must have odd length for linear-phase FIR.")


def apply_energy_cap(magnitude: torch.Tensor, energy_cap: float) -> torch.Tensor:
    """Apply energy cap to high-band magnitude.

    Args:
        magnitude: STFT magnitude (batch, freq, time).
        energy_cap: Maximum allowed energy.

    Returns:
        Magnitude with energy capped.

    Physical Basis:
        Limiting total high-band energy reduces IMD risk and enforces
        safe spectral shaping in the ultrasonic band.
    """
    if energy_cap <= 0:
        raise ValueError(f"energy_cap must be positive, got {energy_cap}.")
    if magnitude.ndim != 3:
        raise ValueError("magnitude must be 3D (batch, freq, time).")

    energy = torch.sum(magnitude**2, dim=(-2, -1), keepdim=True)
    scale = torch.sqrt(energy_cap / (energy + 1.0e-8))
    scale = torch.clamp(scale, max=1.0)
    return magnitude * scale


def _build_envelope_target(
    num_freqs: int,
    sample_rate: int,
    cutoff_hz: float,
    floor: float,
) -> torch.Tensor:
    """Build fixed envelope target for high-band shaping.

    Args:
        num_freqs: Number of frequency bins.
        sample_rate: Sample rate in Hz.
        cutoff_hz: Cutoff frequency in Hz.
        floor: Minimum envelope value at Nyquist.

    Returns:
        Envelope vector shaped (num_freqs,).

    Physical Basis:
        Enforcing a gentle decay above the cutoff prevents excessive
        ultrasonic energy while preserving time response below 20kHz.
    """
    freqs = torch.linspace(0.0, sample_rate / 2, num_freqs)
    envelope = torch.ones_like(freqs)
    if cutoff_hz < sample_rate / 2:
        high = freqs >= cutoff_hz
        if torch.any(high):
            decay = (freqs[high] - cutoff_hz) / ((sample_rate / 2) - cutoff_hz)
            envelope[high] = torch.clamp(1.0 - decay, min=floor)
    return envelope


def _build_highband_mask(
    num_freqs: int,
    sample_rate: int,
    cutoff_hz: float,
) -> torch.Tensor:
    """Build a binary high-band mask in frequency domain.

    Args:
        num_freqs: Number of frequency bins.
        sample_rate: Sample rate in Hz.
        cutoff_hz: Cutoff frequency in Hz.

    Returns:
        Binary mask with 1 for >= cutoff_hz bins, else 0.

    Physical Basis:
        Zeroing bins below cutoff enforces strict low-band preservation
        by preventing high-band leakage into the audible band.
    """
    freqs = torch.linspace(0.0, sample_rate / 2, num_freqs)
    mask = (freqs >= cutoff_hz).to(dtype=torch.float32)
    return mask


def _apply_fir_filter(signal: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """Apply linear-phase FIR filter with convolution.

    Args:
        signal: Input signal (batch, time) or (batch, channels, time).
        taps: FIR taps (1D tensor).

    Returns:
        Filtered signal with same shape as input.

    Physical Basis:
        FIR filtering isolates low/high bands without altering group delay
        in the preserved audible band.
    """
    if taps.ndim != 1:
        raise ValueError("taps must be 1D.")
    if taps.numel() == 0:
        raise ValueError("taps cannot be empty.")
    if taps.numel() % 2 == 0:
        raise ValueError("taps length must be odd for linear-phase filtering.")

    if signal.ndim == 2:
        signal = signal.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False

    taps_flipped = taps.flip(0).to(dtype=signal.dtype, device=signal.device)
    channels = signal.shape[1]
    weight = taps_flipped.view(1, 1, -1).repeat(channels, 1, 1)
    padding = taps_flipped.numel() // 2
    filtered = torch_f.conv1d(signal, weight, padding=padding, groups=channels)
    return filtered.squeeze(1) if squeeze else filtered


def _ensure_2d(signal: torch.Tensor) -> torch.Tensor:
    """Ensure signal is 2D (batch, time).

    Args:
        signal: Input signal (batch, time).

    Returns:
        Input signal unchanged.

    Physical Basis:
        STFT assumes 2D input; channel dimensions are flattened elsewhere.
    """
    if signal.ndim == 2:
        return signal
    raise ValueError("signal must be 2D.")


def _pad_to_multiple(
    features: torch.Tensor,
    multiple: int,
) -> tuple[torch.Tensor, int, int]:
    """Pad feature maps so freq/time are divisible by multiple.

    Args:
        features: Input features (batch, channels, freq, time).
        multiple: Required multiple for divisibility.

    Returns:
        Tuple of (padded_features, pad_freq, pad_time).

    Physical Basis:
        U-Net downsampling requires dimensions divisible by powers of two
        to preserve alignment across skip connections.
    """
    if features.ndim != 4:
        raise ValueError("features must be 4D.")
    if multiple <= 0:
        raise ValueError("multiple must be positive.")

    freq = features.shape[-2]
    time = features.shape[-1]
    pad_f = (multiple - freq % multiple) % multiple
    pad_t = (multiple - time % multiple) % multiple
    padded = torch_f.pad(features, (0, pad_t, 0, pad_f), mode="replicate")
    return padded, pad_f, pad_t


def _crop_to_shape(
    features: torch.Tensor,
    pad_f: int,
    pad_t: int,
) -> torch.Tensor:
    """Remove padding from feature maps.

    Args:
        features: Input features (batch, channels, freq, time).
        pad_f: Padding applied to frequency dimension.
        pad_t: Padding applied to time dimension.

    Returns:
        Cropped features matching original dimensions.

    Physical Basis:
        Removing padding restores original STFT resolution to avoid
        spectral distortion during reconstruction.
    """
    if features.ndim != 4:
        raise ValueError("features must be 4D.")
    freq = features.shape[-2] - pad_f
    time = features.shape[-1] - pad_t
    return features[..., :freq, :time]


def _flatten_signal(
    signal: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int, int] | None]:
    """Flatten channel dimension into batch if present.

    Args:
        signal: Input signal (batch, time) or (batch, channels, time).

    Returns:
        Tuple of (flattened_signal, restore_shape). restore_shape is None
        if no channel dimension was present.

    Physical Basis:
        STFT operates per-channel; flattening allows batch processing
        without mixing channel information.
    """
    if signal.ndim == 2:
        return signal, None
    if signal.ndim == 3:
        batch, channels, time = signal.shape
        return signal.reshape(batch * channels, time), (batch, channels, time)
    raise ValueError("signal must be 2D or 3D.")
