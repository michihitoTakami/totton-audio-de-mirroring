# Data Generation Pipeline

## Overview

The synthetic data pipeline builds paired audio examples for training the Neural-DeRinger upsampler:

- **Target**: Bessel low-pass filtered waveform (zero ringing, maximally flat group delay, 705.6kHz)
- **Input**: SoX downsample (705.6kHz → 44.1kHz) + **Bessel FIR upsample** (44.1kHz → 705.6kHz)

The new pipeline uses **Bessel FIR upsampling** instead of Sinc interpolation. This approach:
1. Preserves **time-domain characteristics** (flat group delay, zero overshoot)
2. Intentionally creates **aliasing artifacts near Nyquist (20kHz-22.05kHz)** due to insufficient stopband attenuation
3. Trains the neural network to **remove aliasing** while preserving Bessel characteristics

This keeps the dataset strictly band-limited while emphasizing both time-domain fidelity and frequency-domain cleanliness.

## Prerequisites

- SoX installed and available on `PATH`
- Python dependencies via `uv sync`

If SoX is missing, install it with your system package manager or run `scripts/setup_system_deps.sh`.

## Quickstart (Python API)

```python
from pathlib import Path
from neural_deringer.data import SyntheticDataGenerator

output_dir = Path("data/synthetic")

generator = SyntheticDataGenerator(
    high_sample_rate=705_600,
    low_sample_rate=44_100,
    duration_seconds=1.0,
    cutoff_hz=20_000.0,
    filter_order=10,
    seed=42,
)

next_index = generator.generate_to_directory(
    output_dir=output_dir,
    total_samples=100,
    batch_size=8,
    format="npy",  # Default (recommended). Use "wav" for compatibility.
)
print(f"Wrote 100 samples, next index: {next_index}")
```

Dataset layout (NPY, default):

```
data/synthetic/
  inputs.npy
  targets.npy
  metadata.jsonl
  config.json
```

Dataset layout (WAV, compatibility):

```
data/synthetic/
  inputs/input_00000000.wav
  targets/target_00000000.wav
  metadata.jsonl
```

Each metadata row includes waveform/envelope parameters plus the SoX downsample profile and Bessel FIR upsample profile used for that sample.

## Bessel FIR Upsampling

### Overview

The new data generation pipeline uses **Bessel FIR filters** for upsampling (44.1kHz → 705.6kHz) instead of Sinc interpolation. This creates a unique training scenario where aliasing artifacts are intentionally introduced and then removed by the neural network.

### Why Bessel FIR?

**Advantages:**
- **Maximally flat group delay**: Preserves transient characteristics and low-frequency fidelity
- **Zero overshoot**: No ringing in time domain (unlike Sinc)
- **Computational efficiency**: 10k-20k taps vs 640k taps for minimum phase FIR

**Trade-off:**
- **Insufficient stopband attenuation**: -40dB to -60dB with 10k-20k taps
- **Aliasing near Nyquist**: 20kHz-22.05kHz range has folded-back content

### Tap Count Selection

The number of FIR filter taps determines the quality/cost trade-off:

| Tap Count | Stopband Attenuation | Group Delay Flatness | Computational Cost | Recommendation |
|-----------|---------------------|----------------------|-------------------|----------------|
| 1k | ~-30dB | Good | Very Low | Too much aliasing |
| 5k | ~-40dB | Very Good | Low | Acceptable for prototyping |
| 10k | ~-50dB | Excellent | Medium | **Recommended minimum** |
| 20k | ~-60dB | Excellent | Medium-High | **Recommended for quality** |
| 50k | ~-70dB | Excellent | High | Overkill (NN can handle -60dB) |

**Recommendation**: Start with **10k taps** for development, use **20k taps** for production training.

### Physical Basis: Why Aliasing Occurs

When upsampling by a factor of 16 (44.1kHz → 705.6kHz):

1. **Zero-stuffing**: Insert 15 zeros between each sample
2. **FIR filtering**: Apply Bessel FIR to interpolate

The Bessel FIR has:
- **Passband**: 0-20kHz (clean, flat group delay)
- **Transition band**: 20kHz-22.05kHz (gradual rolloff)
- **Stopband**: >22.05kHz (insufficient attenuation)

As a result, frequencies near 22.05kHz (original Nyquist) fold back into the 20kHz-22.05kHz range, creating **aliasing artifacts**. The neural network learns to identify and suppress these artifacts.

### GPU Acceleration

Bessel FIR upsampling with 10k-20k taps requires significant computational power:

- **GPU (PyTorch)**: ~10-100ms per 1-second audio file (**100x faster**)
- **CPU**: Not supported (use CUDA for Bessel FIR, or use SoX/minimum phase paths)

**Implementation**: Use `torch.nn.functional.conv1d` for GPU-accelerated FIR filtering.

**Example**:
```python
import torch

# Design Bessel FIR (done once)
fir_coeffs = design_bessel_fir(
    target_sr=705_600,
    source_sr=44_100,
    num_taps=20_000,
    cutoff_hz=20_000.0,
    order=10,
)

# GPU-accelerated upsampling
upsampler = BesselUpsampler(fir_coeffs, device="cuda")
upsampled = upsampler.upsample(signal_44k)
```

### Low-Frequency Fidelity

One of the key benefits of Bessel filters is **low-frequency phase preservation**:

- **Group delay**: Maximally flat across 0-20kHz
- **Phase response**: Nearly linear (no phase distortion)
- **Bass reproduction**: Transients preserved without pre-ringing

This is critical for music upsampling where bass transients (kick drums, bass guitar) must be reproduced accurately.

### Evaluation Metrics

When evaluating Bessel FIR upsampling quality, measure:

1. **Stopband attenuation**: Energy in 20kHz-22.05kHz range (target: <-50dB)
2. **Group delay flatness**: Deviation from constant delay in 0-20kHz (target: <0.1ms)
3. **Aliasing percentage**: % of total energy in aliasing range (target: <0.1%)

**Tools**: `scripts/evaluate_bessel_fir.py`

## SoX Profile Distribution

To reduce overfitting to a single resampling profile, profiles are sampled from fixed distributions.
Downsample and upsample profiles are independently sampled per batch, so mixing profiles across
many batches is the intended policy. These are currently defined in
`src/neural_deringer/data/resample.py`:

- **Quality (`rate -l|-m|-h|-v`)**: l=0.15, m=0.35, h=0.35, v=0.15
- **Phase**: minimum=0.2, intermediate=0.2, linear=0.4, maximum=0.2
- **Bandwidth (%)**: 95.0=0.5, 97.5=0.3, 99.0=0.2
- **Steep filter**: true=0.3, false=0.7
- **Allow aliasing**: true=0.1, false=0.9

If you update these distributions, also update this document.

## Performance & Scale Checks

### Throughput Smoke Test

The generator tracks a throughput target (`throughput_target_audio_hours_per_hour`).
Run a smoke test to confirm runtime health; for example:

- **Goal**: 60 seconds of audio generated in under 12 seconds (with GPU acceleration)
- **Method**: time a single `generate_batch` call with `duration_seconds=60.0` and `batch_size=1`

Keep the threshold generous for development machines and CI environments.

**Note**: Bessel FIR upsampling requires CUDA and should achieve 10-100ms per 1-second audio file.

### Memory Usage Measurement

**Critical for 8GB GPU systems**: Measure memory usage when scaling up batch sizes or chunk sizes.

**Tool**: Use `scripts/profile_memory.py` (EPIC 3)

**Key Metrics**:
- **GPU Memory**: Track with `torch.cuda.memory_summary()`
- **CPU Memory**: Track with `psutil.Process().memory_info()`
- **Batch Size Impact**: Test batch_size=1, 2, 4, 8, 16
- **Chunk Size Impact**: Test 0.1s, 0.25s, 0.5s, 1.0s chunks

**Expected Results** (705.6kHz, float32):
- 1.0s audio: ~2.8MB per sample
- 0.25s audio: ~0.7MB per sample
- Batch size = 8, chunk = 0.25s: ~5.6MB audio data + ~2GB model/activations = ~2.6GB total

### Chunk Size Optimization

For memory-constrained training (6GB available GPU memory):

1. **Use chunked data loading**: Split 1-second audio into 0.25-second chunks
2. **Benefits**:
   - Larger batch sizes (8 vs 2)
   - Faster training (more samples per batch)
   - Fixed memory footprint (independent of original audio length)
3. **Trade-offs**:
   - Boundary artifacts (mitigated by Overlap-Add)
   - More complex data pipeline

**Recommended**: 0.25-second chunks with 50% overlap (Hann window)

**Implementation**: `ChunkedAudioDataset`

### Benchmarking Bessel FIR Performance

Benchmark Bessel FIR upsampling speed vs tap count:

**Tool**: `scripts/evaluate_bessel_fir.py`

**Test Matrix**:
- Tap counts: 1k, 5k, 10k, 20k, 50k
- Devices: CUDA
- Audio lengths: 1s, 10s, 60s

**Expected Results** (NVIDIA GTX 1660 SUPER, 1-second audio):
- GPU (PyTorch): 10-100ms per file

### CPU / Memory Notes

Record resource usage when scaling up:

- Use `time` or `hyperfine` to measure elapsed time
- Use `top` / `htop` / Activity Monitor to capture CPU and memory snapshots
- Use `nvidia-smi` to monitor GPU memory usage
- Include these metrics in experiment notes when adjusting SoX profiles, Bessel tap counts, or batch sizes

## Related Assets

- Integration test: `tests/integration/test_data_pipeline.py`
- Example notebook: `notebooks/01_data_generation_demo.ipynb`

## Data Source Policy (Training)

Training uses **pre-generated datasets on disk** (metadata + NPY/WAV files).
Streaming generation inside the training loop is intentionally out of scope
for now to keep I/O deterministic and reproducible.

If streaming generation becomes necessary in the future, define and document:

- fixed RNG seed policy per epoch
- epoch size policy (number of samples per epoch)
- input/target shapes aligned with `SyntheticBatch` (`(batch, time)`)
