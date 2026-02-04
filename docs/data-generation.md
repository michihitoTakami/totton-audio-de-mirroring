# Data Generation Guide

## Overview

Neural-DeRinger supports two data generation workflows:

1. **Pre-computed generation (Recommended)**: Generate all training data in advance and save to disk in efficient `.npy` format
2. **On-the-fly generation**: Generate data during training (legacy, slower due to I/O overhead)

## Pre-computed Generation (Recommended)

### Why Pre-compute?

Pre-computing training data eliminates the I/O bottleneck caused by:
- Disk writes/reads for temporary files
- External SoX process spawning
- File system overhead

**Benefits:**
- 10-100x faster training iteration
- Higher GPU utilization (80-95% vs 5-20%)
- Consistent data across epochs
- Easy data sharing and versioning

### Usage

Generate dataset using the provided script:

```bash
# Generate 10,000 samples with default settings
uv run python scripts/generate_dataset.py \
  --num-samples 10000 \
  --output-dir data/synthetic

# Custom configuration
uv run python scripts/generate_dataset.py \
  --num-samples 50000 \
  --batch-size 16 \
  --output-dir data/synthetic_50k \
  --high-sample-rate 705600 \
  --low-sample-rate 44100 \
  --duration 1.0 \
  --cutoff-hz 20000 \
  --filter-order 10 \
  --seed 42 \
  --format npy
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-samples` | 10000 | Total number of training samples |
| `--batch-size` | 8 | Batch size for generation |
| `--output-dir` | data/synthetic | Output directory |
| `--high-sample-rate` | 705600 | High-resolution sample rate (Hz) |
| `--low-sample-rate` | 44100 | Low-resolution sample rate (Hz) |
| `--duration` | 1.0 | Sample duration (seconds) |
| `--cutoff-hz` | 20000.0 | Bessel filter cutoff frequency (Hz) |
| `--filter-order` | 10 | Bessel filter order |
| `--seed` | None | Random seed for reproducibility |
| `--format` | npy | Storage format: `npy` (recommended) or `wav` |
| `--upsampler-type` | bessel | Upsampler type (`bessel`, `bessel_fir` (legacy), `minimum_phase`, `sox`) |
| `--bessel-taps` | 10000 | Bessel FIR tap count (for `bessel`, legacy alias: `--bessel-fir-taps`) |
| `--bessel-device` | None | CUDA device for Bessel FIR upsampling (`cuda` / `cuda:0`) |

### Output Structure

```
data/synthetic/
├── inputs.npy           # Input signals (Bessel FIR upsampled, with aliasing)
├── targets.npy          # Target signals (Bessel-filtered, no ringing)
├── metadata.jsonl       # Per-sample metadata
└── config.json          # Dataset configuration
```

### Loading Pre-computed Data

```python
from neural_deringer.data import PrecomputedDataset
from torch.utils.data import DataLoader
from pathlib import Path

# Create dataset
dataset = PrecomputedDataset(
    data_dir=Path("data/synthetic"),
    device="cuda"
)

# Create data loader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        # inputs, targets are already on GPU
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # ...
```

## On-the-Fly Generation (Legacy)

### When to Use

Only use on-the-fly generation for:
- Prototyping and experimentation
- Limited disk space scenarios
- Testing new data augmentation strategies

### Usage

```python
from neural_deringer.data import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    high_sample_rate=705600,
    low_sample_rate=44100,
    duration_seconds=1.0,
    seed=42
)

# Generate single batch
batch = generator.generate_batch(batch_size=8)

# Iterate over batches
for batch in generator.iter_batches(total_samples=1000, batch_size=8):
    inputs = batch.inputs
    targets = batch.targets
    # ...
```

### Performance Comparison

| Method | Samples/sec | GPU Util | Memory |
|--------|-------------|----------|--------|
| On-the-fly | 5-20 | 5-20% | Low |
| Pre-computed | 500-2000 | 80-95% | Medium |

## WAV Format (Compatibility)

For compatibility with external tools, you can generate data in WAV format:

```bash
uv run python scripts/generate_dataset.py \
  --num-samples 1000 \
  --output-dir data/synthetic_wav \
  --format wav
```

**Output structure:**
```
data/synthetic_wav/
├── inputs/
│   ├── input_00000000.wav
│   ├── input_00000001.wav
│   └── ...
├── targets/
│   ├── target_00000000.wav
│   ├── target_00000001.wav
│   └── ...
└── metadata.jsonl
```

**Note:** WAV format requires significantly more disk space and loading time compared to `.npy` format.

## Storage Requirements

Approximate disk space for 10,000 samples (1 second each, 705.6 kHz):

| Format | Size |
|--------|------|
| NPY (float64) | ~110 GB |
| WAV (PCM_32) | ~220 GB |

**Tip:** Use shorter duration (e.g., 0.1-0.5 seconds) for development to reduce storage requirements.

## Best Practices

1. **Always use NPY format for training**: Fastest I/O and most efficient storage
2. **Generate data once, train many times**: Pre-compute large datasets and reuse across experiments
3. **Use seed for reproducibility**: Set `--seed` to ensure consistent data generation
4. **Monitor throughput**: Check generation throughput to estimate time for large datasets
5. **Validate generated data**: Run integration tests after generation to verify data quality

## Example Workflow

```bash
# 1. Generate development dataset (small, for quick iteration)
uv run python scripts/generate_dataset.py \
  --num-samples 1000 \
  --duration 0.1 \
  --output-dir data/dev \
  --seed 42

# 2. Train model on dev set
uv run python scripts/train.py \
  --data-dir data/dev \
  --batch-size 32 \
  --epochs 10

# 3. Generate full training dataset
uv run python scripts/generate_dataset.py \
  --num-samples 100000 \
  --duration 1.0 \
  --output-dir data/train \
  --seed 42

# 4. Train final model
uv run python scripts/train.py \
  --data-dir data/train \
  --batch-size 64 \
  --epochs 100
```

## Troubleshooting

### Out of Memory during Generation

Reduce batch size:
```bash
uv run python scripts/generate_dataset.py \
  --batch-size 2 \
  --num-samples 10000
```

### Disk Space Issues

Use shorter duration:
```bash
uv run python scripts/generate_dataset.py \
  --duration 0.1 \
  --num-samples 10000
```

### SoX Not Found

Install SoX:
```bash
# Ubuntu/Debian
sudo apt-get install sox

# macOS
brew install sox
```

## Physical Basis

### Input Generation
1. Original signal → Bessel filter (target)
2. Target → Downsample with SoX (low rate)
3. Low rate → Upsample with **Bessel FIR** (input with aliasing near Nyquist)

### Why Bessel FIR Introduces Aliasing
Bessel FIR filters trade stopband attenuation for flat group delay. With 10k-20k
taps, attenuation near 22.05kHz is limited, so folded-back components appear in
the 20kHz-22.05kHz band. The network learns to suppress this aliasing.

### Why Bessel Preserves Transients
Bessel filters have maximally flat group delay, resulting in zero overshoot in
step response and minimal phase distortion across the passband.

### Training Objective
Learn mapping: `Input (with aliasing) → Target (clean)`
