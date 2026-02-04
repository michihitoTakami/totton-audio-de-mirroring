# Bessel Filter-Based Upsampling

## Table of Contents

- [Theoretical Background](#theoretical-background)
- [FIR Filter Design](#fir-filter-design)
- [Implementation Strategy](#implementation-strategy)
- [Evaluation Metrics](#evaluation-metrics)
- [Comparison with Other Methods](#comparison-with-other-methods)

---

## Theoretical Background

### What is a Bessel Filter?

A **Bessel filter** (also called Thomson filter) is a type of analog filter designed to have a **maximally flat group delay** in the passband. This unique characteristic makes it ideal for applications where **phase linearity** and **transient preservation** are critical.

#### Key Properties

1. **Maximally Flat Group Delay**
   - Group delay is nearly constant across the passband (0-20kHz)
   - Minimizes phase distortion → preserves transient characteristics
   - Critical for audio: bass transients (kick drums, bass guitar) reproduced accurately

2. **Zero Overshoot in Step Response**
   - Unlike Butterworth or Chebyshev filters, Bessel has **no overshoot**
   - Step response rises monotonically without ringing
   - Ideal for digital audio upsampling (no pre-echo or post-echo)

3. **Near-Linear Phase Response**
   - Phase response is approximately linear in passband
   - Equivalent to constant time delay for all frequencies
   - Preserves waveform shape (no phase distortion)

4. **Gradual Rolloff (Trade-off)**
   - Frequency response rolls off more gradually than Butterworth/Chebyshev
   - **Insufficient stopband attenuation** for typical upsampling requirements
   - This is the key challenge that Neural-DeRinger addresses

### Comparison: Time Domain vs Frequency Domain

| Filter Type | Time Domain | Frequency Domain |
|-------------|-------------|------------------|
| **Sinc (Ideal)** | **Severe ringing** (Gibbs phenomenon) | **Perfect** (brick-wall) |
| **Butterworth** | Moderate ringing | Good rolloff, some ripple |
| **Chebyshev** | Heavy ringing | Sharp rolloff, passband ripple |
| **Bessel** | **Zero overshoot** | **Gradual rolloff** |

**Neural-DeRinger's Approach**: Use Bessel for time-domain excellence, then use NN to fix frequency-domain issues (aliasing).

### Why Bessel for Audio?

Traditional digital audio processing prioritizes **frequency-domain accuracy** (flat response to 20kHz, brick-wall cutoff). However, this comes at the cost of **time-domain distortion**:

- **Sinc interpolation**: Ideal in frequency but creates pre-echo/post-echo (ringing)
- **Minimum phase FIR**: Reduces pre-echo but has nonlinear group delay
- **Bessel FIR**: Preserves time-domain characteristics but allows some aliasing

For music playback, **time-domain accuracy** is arguably more important:
- Kick drum transients should be sharp, not smeared
- Bass guitar attacks should be punchy, not blurred
- Cymbal hits should be crisp, not ringing

Bessel filters excel at this, making them a natural choice for a "time-domain first" upsampler.

---

## FIR Filter Design

### Design Process

To create a Bessel FIR filter for upsampling:

1. **Design analog Bessel filter** using `scipy.signal.bessel`:
   ```python
   from scipy import signal

   # Design analog Bessel filter
   order = 10
   cutoff_freq = 20_000.0  # Hz
   b, a = signal.bessel(order, cutoff_freq, analog=True)
   ```

2. **Digitize using bilinear transform** (or impulse invariance):
   ```python
   sample_rate = 705_600  # Hz
   b_digital, a_digital = signal.bilinear(b, a, fs=sample_rate)
   ```

3. **Convert IIR to FIR** by truncating impulse response:
   ```python
   num_taps = 20_000
   impulse = signal.unit_impulse(num_taps)
   fir_coeffs = signal.lfilter(b_digital, a_digital, impulse)
   ```

4. **Normalize** to ensure unity gain at DC:
   ```python
   fir_coeffs /= np.sum(fir_coeffs)
   ```

### Tap Count vs Frequency Response

The number of FIR taps determines the quality of the approximation:

| Tap Count | Stopband Attenuation | Transition Bandwidth | Group Delay Flatness |
|-----------|---------------------|---------------------|---------------------|
| 1,000 | ~-30dB | ~4kHz | Good |
| 5,000 | ~-40dB | ~2kHz | Very Good |
| 10,000 | ~-50dB | ~1kHz | Excellent |
| 20,000 | ~-60dB | ~0.5kHz | Excellent |
| 50,000 | ~-70dB | ~0.2kHz | Excellent |

**Recommendation**:
- **Development**: 10k taps (good balance, fast iteration)
- **Production**: 20k taps (high quality, acceptable computational cost)
- **Research**: 50k taps (diminishing returns, mainly for analysis)

### Why Insufficient Stopband Attenuation?

For upsampling by 16× (44.1kHz → 705.6kHz), the Nyquist frequency of the input is **22.05kHz**. Ideally, the FIR filter should have:

- **Passband**: 0-20kHz (human hearing)
- **Transition band**: 20kHz-22.05kHz
- **Stopband**: >22.05kHz with **>60dB attenuation**

However, a Bessel FIR with 20k taps achieves only **-60dB** attenuation in the stopband. This means that frequencies near 22.05kHz (which should be fully suppressed) are only reduced by a factor of 1,000.

**Result**: Aliasing artifacts appear in the 20kHz-22.05kHz range.

**Solution**: Train a neural network to identify and suppress these artifacts.

### Phase Response and Group Delay

The defining characteristic of Bessel filters is their **maximally flat group delay**:

```
Group Delay = -dφ/dω
```

Where φ is phase and ω is angular frequency.

**Ideal**: Group delay is constant (e.g., 500 samples) across 0-20kHz
**Bessel**: Group delay deviation < 0.1ms across 0-20kHz
**Minimum Phase**: Group delay varies by >1ms across 0-20kHz

This flatness ensures that all frequencies experience the same time delay, preserving transient shape.

### Stopband Attenuation Goals

For Neural-DeRinger, the stopband attenuation target is:

- **Acceptable**: -40dB (aliasing artifacts are learnable by NN)
- **Good**: -50dB (minimal aliasing, easier for NN)
- **Excellent**: -60dB (very little aliasing, NN focuses on other artifacts)

**Note**: -60dB attenuation = 0.1% energy leakage, which is perceptually negligible after NN cleanup.

---

## Implementation Strategy

### CPU Implementation (Reference Only)

**Note**: The codebase no longer supports CPU-based Bessel FIR upsampling. The
GPU implementation is required for practical 10k-20k tap processing. If CUDA
is not available, use the SoX or minimum phase paths instead.

For initial development and testing:

```python
import numpy as np
from scipy import signal

def design_bessel_fir(
    target_sr: int,
    source_sr: int,
    num_taps: int,
    cutoff_hz: float,
    order: int = 10,
) -> np.ndarray:
    """Design Bessel FIR filter for upsampling.

    Args:
        target_sr: Target sample rate (e.g., 705,600 Hz)
        source_sr: Source sample rate (e.g., 44,100 Hz)
        num_taps: Number of FIR filter taps
        cutoff_hz: Cutoff frequency in Hz
        order: Bessel filter order

    Returns:
        FIR filter coefficients (num_taps,)

    Physical Basis:
        Bessel filter has maximally flat group delay but insufficient
        stopband attenuation, causing aliasing near Nyquist. This is
        intentional for training data generation.
    """
    # Design analog Bessel filter
    b, a = signal.bessel(order, cutoff_hz, analog=True)

    # Digitize using bilinear transform
    b_digital, a_digital = signal.bilinear(b, a, fs=target_sr)

    # Generate impulse response
    impulse = signal.unit_impulse(num_taps)
    fir_coeffs = signal.lfilter(b_digital, a_digital, impulse)

    # Normalize to unity gain at DC
    fir_coeffs /= np.sum(fir_coeffs)

    return fir_coeffs


def upsample_with_bessel_fir(
    signal_44k: np.ndarray,
    fir_coefficients: np.ndarray,
    upsample_ratio: int = 16,
) -> np.ndarray:
    """Upsample using Bessel FIR filter (CPU, slow).

    Args:
        signal_44k: Input signal at 44.1kHz
        fir_coefficients: Bessel FIR filter coefficients
        upsample_ratio: Upsampling ratio (16 for 44.1kHz → 705.6kHz)

    Returns:
        Upsampled signal at 705.6kHz

    Physical Basis:
        Zero-stuffing + FIR convolution creates aliasing artifacts
        near Nyquist that the neural network learns to suppress.
    """
    # Zero-stuffing: insert (ratio - 1) zeros between each sample
    upsampled_length = len(signal_44k) * upsample_ratio
    zero_stuffed = np.zeros(upsampled_length, dtype=signal_44k.dtype)
    zero_stuffed[::upsample_ratio] = signal_44k

    # Apply FIR filter
    filtered = signal.convolve(zero_stuffed, fir_coefficients, mode='same')

    # Scale by upsampling ratio (energy conservation)
    filtered *= upsample_ratio

    return filtered
```

**Performance**: ~1-10 seconds per 1-second audio file (too slow; not supported)

### GPU Implementation (Production)

For production use, implement GPU-accelerated FIR filtering with PyTorch:

```python
import torch
import torch.nn.functional as F

class BesselUpsampler:
    """GPU-accelerated Bessel FIR upsampler.

    Attributes:
        fir_coefficients: Bessel FIR filter coefficients (torch.Tensor)
        upsample_ratio: Upsampling ratio (16 for 44.1kHz → 705.6kHz)
        device: CUDA device
    """

    def __init__(self, fir_coefficients: np.ndarray, device: str = "cuda"):
        """Initialize Bessel upsampler.

        Args:
            fir_coefficients: Bessel FIR filter coefficients (NumPy array)
            device: CUDA device to run on (e.g., "cuda", "cuda:0")
        """
        self.device = torch.device(device)
        self.fir = torch.from_numpy(fir_coefficients).float().to(self.device)
        self.upsample_ratio = 16  # 44.1kHz → 705.6kHz

    def upsample(self, signal_44k: np.ndarray) -> np.ndarray:
        """GPU-accelerated upsampling.

        Args:
            signal_44k: Input signal at 44.1kHz (NumPy array)

        Returns:
            Upsampled signal at 705.6kHz (NumPy array)
        """
        # Convert to PyTorch tensor
        x = torch.from_numpy(signal_44k).float().to(self.device)

        # Add batch and channel dimensions (B, C, T) = (1, 1, T)
        x = x.unsqueeze(0).unsqueeze(0)

        # Zero-stuffing using F.interpolate
        x = F.interpolate(
            x,
            scale_factor=self.upsample_ratio,
            mode='nearest',
        )

        # Apply FIR filter using Conv1d
        # Reshape FIR to (out_channels, in_channels, kernel_size) = (1, 1, num_taps)
        fir_kernel = self.fir.unsqueeze(0).unsqueeze(0)
        x = F.conv1d(x, fir_kernel, padding=len(self.fir) // 2)

        # Scale by upsampling ratio
        x = x * self.upsample_ratio

        # Convert back to NumPy
        return x.squeeze().cpu().numpy()
```

**Performance**: ~10-100ms per 1-second audio file (**100× faster than CPU**)

### Memory Efficiency

When processing large datasets:

1. **Batch processing**: Process multiple files in parallel
2. **Streaming**: Process audio in chunks to avoid loading entire file into memory
3. **Mixed precision**: Use float16 for intermediate computations (requires careful handling)

### Comparison with Minimum Phase FIR

| Feature | Minimum Phase FIR (640k taps) | Bessel FIR (20k taps) |
|---------|-------------------------------|----------------------|
| **Group Delay** | Nonlinear (minimum phase) | **Maximally flat** |
| **Stopband Attenuation** | >60dB | -60dB (borderline) |
| **Computational Cost** | Very High (32× more taps) | Medium |
| **Memory Usage** | High (256MB for coefficients) | Low (80KB for coefficients) |
| **GPU Efficiency** | Poor (kernel too large) | **Good** |
| **Low-Frequency Fidelity** | Good | **Excellent** |

**Conclusion**: Bessel FIR is a more practical choice for real-time upsampling with GPU acceleration.

---

## Evaluation Metrics

### 1. Stopband Attenuation

**Definition**: Energy reduction in the stopband (>22.05kHz)

**Measurement**:
```python
def measure_stopband_attenuation(fir_coeffs: np.ndarray, sample_rate: int) -> float:
    """Measure stopband attenuation in dB.

    Returns:
        Stopband attenuation in dB (negative value, e.g., -60.0)
    """
    # Compute frequency response
    w, h = signal.freqz(fir_coeffs, worN=8192, fs=sample_rate)

    # Find stopband (>22.05kHz)
    stopband_mask = w > 22_050

    # Compute attenuation
    stopband_energy = np.mean(np.abs(h[stopband_mask]) ** 2)
    passband_energy = np.mean(np.abs(h[w < 20_000]) ** 2)

    attenuation_db = 10 * np.log10(stopband_energy / passband_energy)
    return attenuation_db
```

**Target**: < -50dB (good), < -60dB (excellent)

### 2. Group Delay Flatness

**Definition**: Deviation from constant group delay in passband (0-20kHz)

**Measurement**:
```python
def measure_group_delay_flatness(fir_coeffs: np.ndarray, sample_rate: int) -> float:
    """Measure group delay flatness in milliseconds.

    Returns:
        Maximum deviation from mean group delay in ms
    """
    # Compute group delay
    w, gd = signal.group_delay((fir_coeffs, 1), w=8192, fs=sample_rate)

    # Focus on passband (0-20kHz)
    passband_mask = w < 20_000
    gd_passband = gd[passband_mask]

    # Compute deviation
    mean_gd = np.mean(gd_passband)
    max_deviation_samples = np.max(np.abs(gd_passband - mean_gd))

    # Convert to milliseconds
    max_deviation_ms = max_deviation_samples / sample_rate * 1000
    return max_deviation_ms
```

**Target**: < 0.1ms (excellent), < 0.5ms (good)

### 3. Aliasing Energy Percentage

**Definition**: Percentage of total energy in the aliasing range (20kHz-22.05kHz)

**Measurement**:
```python
def measure_aliasing_percentage(
    upsampled_signal: np.ndarray,
    sample_rate: int,
) -> float:
    """Measure aliasing energy percentage.

    Returns:
        Percentage of total energy in 20kHz-22.05kHz range
    """
    # Compute power spectrum
    freqs = np.fft.rfftfreq(len(upsampled_signal), 1 / sample_rate)
    spectrum = np.abs(np.fft.rfft(upsampled_signal)) ** 2

    # Aliasing range
    aliasing_mask = (freqs >= 20_000) & (freqs <= 22_050)
    aliasing_energy = np.sum(spectrum[aliasing_mask])

    # Total energy
    total_energy = np.sum(spectrum)

    # Percentage
    aliasing_percentage = (aliasing_energy / total_energy) * 100
    return aliasing_percentage
```

**Target**: < 0.1% (excellent), < 1.0% (good)

### 4. Phase Linearity

**Definition**: Deviation from linear phase in passband

**Measurement**: Compute phase response and measure R² of linear fit

**Target**: R² > 0.99 (excellent)

---

## Comparison with Other Methods

### Sinc Interpolation

**Pros**:
- Perfect frequency response (brick-wall cutoff)
- Mathematically ideal

**Cons**:
- Severe ringing (Gibbs phenomenon)
- Pre-echo and post-echo
- Non-causal (requires lookahead)

**Use Case**: Traditional DSP, not suitable for music playback

### Minimum Phase FIR

**Pros**:
- Causal (no lookahead required)
- Good stopband attenuation (>60dB)

**Cons**:
- Nonlinear group delay (phase distortion)
- Very high computational cost (640k taps)
- Poor GPU efficiency

**Use Case**: Offline processing where quality trumps speed

### Bessel FIR (This Project)

**Pros**:
- Maximally flat group delay (best time-domain fidelity)
- Zero overshoot (no ringing)
- Moderate computational cost (10k-20k taps)
- Good GPU efficiency

**Cons**:
- Insufficient stopband attenuation (aliasing artifacts)
- Requires neural network for cleanup

**Use Case**: Real-time music upsampling with NN post-processing

### Standard AI Super-Resolution

**Pros**:
- Can generate >20kHz content (if desired)
- End-to-end learned

**Cons**:
- Risk of hallucination (generating non-existent content)
- Computationally expensive (generative models)
- Unpredictable on out-of-distribution inputs

**Use Case**: Music restoration, not transparent upsampling

---

## Summary

Bessel filter-based upsampling is a hybrid approach that combines the best of traditional DSP (physics-informed, predictable) and modern AI (artifact removal, adaptive). By using Bessel FIR filters, we achieve excellent time-domain characteristics (flat group delay, zero overshoot) at the cost of some aliasing artifacts, which are then removed by a neural network.

This "time-domain first, then fix frequency issues" philosophy is the core of the Neural-DeRinger project.

For implementation details, see:
- `src/neural_deringer/data/bessel_upsample.py` (implementation, EPIC 2)
- `scripts/evaluate_bessel_fir.py` (evaluation script, EPIC 2)
- `notebooks/02_bessel_fir_design.ipynb` (interactive design, EPIC 2)
