# Memory Optimization Strategy

## Table of Contents

- [Memory Constraints Analysis](#memory-constraints-analysis)
- [Chunking Strategy](#chunking-strategy)
- [Model Architecture Optimization](#model-architecture-optimization)
- [Benchmarking](#benchmarking)

---

## Memory Constraints Analysis

### Hardware Assumptions

**Target System**: Consumer GPU with 8GB VRAM
- **Total GPU Memory**: 8GB (8,192MB)
- **System Reserved**: ~2GB (CUDA driver, OS, etc.)
- **Available for Training**: ~6GB (6,144MB)

### Memory Breakdown

When training a neural network, GPU memory is allocated for:

1. **Model Parameters**: Weights and biases
2. **Gradients**: Same size as parameters (for backpropagation)
3. **Optimizer State**: 2× parameters for Adam (momentum + variance)
4. **Activations**: Intermediate outputs during forward pass
5. **Input/Target Data**: Training batch

**Total Memory**: Parameters × 4 (params + grads + 2× optimizer) + Activations + Data

### Sample Memory Calculations

#### Scenario 1: Full-length (1 second @ 705.6kHz)

**Audio Data**:
- 1 sample: 705,600 samples × 4 bytes (float32) = **2.8MB**
- Batch size = 2: 2 × 2.8MB = **5.6MB**

**Model (U-Net depth=3, base_channels=32)**:
- Parameters: ~1.2M × 4 bytes = **4.8MB**
- Gradients: **4.8MB**
- Optimizer (Adam): 2 × 4.8MB = **9.6MB**
- Activations: ~600MB (depends on batch size, depth)

**Total**: 5.6MB (data) + 19.2MB (model) + 600MB (activations) ≈ **625MB**

**Analysis**: Fits easily in 6GB, but **batch size limited to 2** (small batch → slow convergence).

#### Scenario 2: Chunked (0.25 seconds @ 705.6kHz)

**Audio Data**:
- 1 sample: 176,400 samples × 4 bytes = **0.7MB**
- Batch size = 8: 8 × 0.7MB = **5.6MB**

**Model (U-Net depth=4, base_channels=32)**:
- Parameters: ~1.5M × 4 bytes = **6MB**
- Gradients: **6MB**
- Optimizer (Adam): 2 × 6MB = **12MB**
- Activations: ~1.5GB (larger model, but smaller input)

**Total**: 5.6MB (data) + 24MB (model) + 1,500MB (activations) ≈ **1,530MB**

**Analysis**: Fits comfortably in 6GB, **batch size = 8** (4× larger → faster training).

**Key Insight**: Chunking allows larger batch sizes with the same memory budget.

### Memory vs Chunk Size

| Chunk Size | Samples @ 705.6kHz | MB per Sample (float32) | Max Batch Size (6GB) |
|------------|-------------------|------------------------|---------------------|
| 1.0s | 705,600 | 2.8MB | ~2 |
| 0.5s | 352,800 | 1.4MB | ~4 |
| 0.25s | 176,400 | **0.7MB** | **~8** |
| 0.1s | 70,560 | 0.3MB | ~16 |

**Recommendation**: **0.25 seconds** (good balance between batch size and context length)

### U-Net Depth and Channel Impact

**Memory scaling**:
- **Depth**: Each additional depth level doubles resolution in encoder/decoder → activations grow exponentially
- **Channels**: Linear relationship with memory (2× channels → 2× activations)

| Depth | Base Channels | Parameters | Activations (est.) | Total Memory (batch=8, 0.25s) |
|-------|---------------|------------|-------------------|-------------------------------|
| 3 | 24 | 0.8M | ~800MB | ~900MB |
| 3 | 32 | 1.2M | ~1,000MB | ~1,100MB |
| 4 | 32 | 1.5M | ~1,500MB | ~1,600MB |
| 4 | 48 | 3.2M | ~2,000MB | ~2,100MB |
| 5 | 32 | 2.0M | ~3,000MB | ~3,200MB |

**Recommendation for 6GB GPU**:
- **Depth**: 3-4 (depth=5 risks OOM)
- **Base Channels**: 24-48 (32 is sweet spot)

---

## Chunking Strategy

### Overview

**Chunking** divides long audio files into fixed-size segments for training and inference. This enables:
1. **Arbitrary-length audio** processing with fixed memory footprint
2. **Larger batch sizes** → faster convergence
3. **Parallelizable** processing

### Training-Time Chunking

#### ChunkedAudioDataset

**Implementation**: `src/neural_deringer/training/chunked_dataloader.py`

**Key Design Decisions**:
- **Deterministic chunking**: Same chunks every epoch (reproducibility)
- **Overlap handling**: Overlap region blended with window function during training
- **Boundary padding**: Zero-pad or reflect-pad last chunk if needed

### Overlap-Add Window Functions

**Goal**: Ensure smooth transitions at chunk boundaries

**Window Functions**:
1. **Hann Window** (Recommended)
   - Perfect Reconstruction when 50% overlap
   - Smooth transitions
   - Computationally efficient

2. **Hamming Window**
   - Good frequency characteristics
   - Not Perfect Reconstruction at 50% overlap

3. **Triangular Window**
   - Simple, fast
   - Requires 50% overlap for Perfect Reconstruction

**Perfect Reconstruction Condition**:
```
sum(window[i] + window[i + hop_size]) = constant for all i
```

For Hann window with 50% overlap:
```python
window = np.hanning(chunk_size)
hop_size = chunk_size // 2  # 50% overlap

# Verify Perfect Reconstruction
assert np.allclose(
    window[:hop_size] + window[hop_size:],
    np.ones(hop_size)
)
```

**Recommendation**: **Hann window with 50% overlap**

### Boundary Artifact Mitigation

Chunk boundaries can introduce artifacts if not handled carefully:

**Problem**:
- Discontinuities at chunk boundaries → audible clicks
- Model trained on isolated chunks may not learn global context

**Solutions**:
1. **Overlap-Add with windowing** (prevents discontinuities)
2. **Cross-fade** between chunks (smooth transitions)
3. **Context padding** (provide extra samples beyond chunk for model context)

**Example: Context Padding**
```python
# Instead of processing chunk[0:chunk_size]
# Process chunk[-context:chunk_size+context] and only use center
context = 512  # samples of extra context

input_with_context = input_audio[start-context:end+context]
output_with_context = model(input_with_context)
output_chunk = output_with_context[context:-context]  # Discard edges
```

### Inference-Time Overlap-Add

**Implementation**: `src/neural_deringer/inference/chunking.py` (already implemented)

**Process**:
1. Split input audio into overlapping chunks (50% overlap recommended)
2. Process each chunk through model
3. Apply window function to each chunk
4. Overlap-add windowed chunks to reconstruct full output

**Pseudocode**:
```python
def process_with_overlap_add(
    audio: np.ndarray,
    model: nn.Module,
    chunk_size: int = 176_400,
    overlap: float = 0.5,
) -> np.ndarray:
    """Process long audio with Overlap-Add.

    Args:
        audio: Input audio (T,)
        model: Neural network model
        chunk_size: Chunk size in samples
        overlap: Overlap fraction (0.5 = 50%)

    Returns:
        Processed audio (T,)
    """
    hop_size = int(chunk_size * (1 - overlap))
    window = np.hanning(chunk_size)

    output = np.zeros_like(audio)
    window_sum = np.zeros_like(audio)

    for start in range(0, len(audio) - chunk_size + 1, hop_size):
        # Extract chunk
        chunk = audio[start:start + chunk_size]

        # Process chunk
        processed = model(chunk)

        # Apply window
        windowed = processed * window

        # Overlap-add
        output[start:start + chunk_size] += windowed
        window_sum[start:start + chunk_size] += window

    # Normalize by window sum
    output /= (window_sum + 1e-8)

    return output
```

**GPU Optimization** (EPIC 3):
- Batch multiple chunks together
- Run Overlap-Add on GPU (avoid CPU-GPU transfers)
- Use CUDA streams for overlapping compute and data transfer

---

## Model Architecture Optimization

### Receptive Field Design

**Key Principle**: Model's receptive field should be appropriate for chunk size

**Receptive Field Formula** (approximate for U-Net):
```
RF = kernel_size × (2^depth - 1) × dilation
```

For kernel_size=3, dilation=1:
- Depth=3: RF ≈ 21 samples (0.03ms @ 705.6kHz)
- Depth=4: RF ≈ 45 samples (0.06ms @ 705.6kHz)
- Depth=5: RF ≈ 93 samples (0.13ms @ 705.6kHz)

**Reality**: Skip connections increase effective receptive field significantly

**Guideline**:
- For 0.25s chunks (176k samples): Depth=4 is sufficient
- For 0.1s chunks (70k samples): Depth=3 is sufficient

### Depth vs Channels Trade-off

**Trade-off**:
- **More depth**: Larger receptive field, but exponentially more memory
- **More channels**: More expressiveness, but linearly more memory

**Recommendations**:
| Chunk Size | Depth | Base Channels | Rationale |
|------------|-------|---------------|-----------|
| 0.1s | 3 | 24 | Minimize memory, small context |
| 0.25s | 4 | 32 | **Balanced** (recommended) |
| 0.5s | 4 | 48 | High quality, larger context |
| 1.0s | 3-4 | 32 | Full-length (no chunking needed) |

### Mixed Precision Training

**Automatic Mixed Precision (AMP)** uses float16 for some operations to save memory:

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in mixed precision
    with autocast():
        output = model(input)
        loss = criterion(output, target)

    # Backward pass (automatic scaling)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits**:
- **2× less memory** for activations (float16 vs float32)
- **1.5-2× faster** on modern GPUs (Tensor Cores)

**Caution**:
- Requires careful loss scaling (AMP handles automatically)
- Not all operations support float16

**Recommendation**: Enable AMP for training depth=4-5 models on 6GB GPUs

### Gradient Checkpointing

**Gradient Checkpointing** trades compute for memory by recomputing activations during backward pass:

```python
from torch.utils.checkpoint import checkpoint

class UNetWithCheckpointing(nn.Module):
    def forward(self, x):
        # Use checkpointing for memory-intensive layers
        x = checkpoint(self.encoder_block1, x)
        x = checkpoint(self.encoder_block2, x)
        # ... etc
```

**Benefits**:
- **30-50% less memory** for activations
- Enables training larger models on limited memory

**Cost**:
- **30-50% slower** training (recompute overhead)

**Recommendation**: Use only if necessary (depth=5+, or very large batch sizes)

---

## Benchmarking

### Memory Profiling Script

**Tool**: `scripts/profile_memory.py` (EPIC 3)

**Features**:
- Measure GPU memory usage vs batch size
- Measure GPU memory usage vs chunk size
- Measure GPU memory usage vs model architecture (depth, channels)
- Plot memory usage graphs
- Generate recommendations

**Example Output**:
```
=== Memory Profiling Results ===

Chunk Size: 0.25s (176,400 samples @ 705.6kHz)
Model: U-Net depth=4, base_channels=32
Device: NVIDIA GTX 1660 SUPER (6GB)

Batch Size | GPU Memory | Status
-----------|------------|--------
1          | 750 MB     | ✓ OK
2          | 1,100 MB   | ✓ OK
4          | 1,800 MB   | ✓ OK
8          | 2,600 MB   | ✓ OK (Recommended)
16         | 4,500 MB   | ✓ OK
32         | OOM        | ✗ Out of Memory

Recommendation: Use batch_size=8-16 for optimal training speed.
```

### Chunk Size Optimization Experiment

**Tool**: `scripts/optimize_chunk_size.py` (EPIC 3)

**Experiment Protocol**:
1. Train model with different chunk sizes (0.1s, 0.25s, 0.5s)
2. Measure:
   - Validation loss convergence
   - Training speed (samples/sec)
   - GPU memory usage
   - Inference quality (boundary artifacts)
3. Plot trade-off curves
4. Recommend optimal chunk size

**Expected Results**:
- **0.1s**: Fast training, but boundary artifacts
- **0.25s**: **Balanced** (recommended)
- **0.5s**: High quality, but slower training

### Recommended Chunk Sizes

Based on theoretical analysis and expected experimental results:

**For Training**:
- **Development**: 0.1s (fast iteration, lower quality)
- **Production**: **0.25s** (balanced quality/speed)
- **High-Quality**: 0.5s (best quality, slower)

**For Inference**:
- **Real-time**: 0.1s (low latency)
- **Offline**: **0.25s** (good quality, reasonable speed)
- **Archival**: 0.5-1.0s (best quality)

### Overlap Size Optimization

**Tool**: `scripts/evaluate_overlap_add_windows.py` (EPIC 5)

**Experiment**: Compare 25%, 50%, 75% overlap with different window functions

**Expected Results**:
- **25% overlap**: Fast, but more boundary artifacts
- **50% overlap**: **Balanced** (recommended)
- **75% overlap**: Fewer artifacts, but 4× slower

**Recommendation**: **50% overlap with Hann window**

---

## Summary

Memory optimization for Neural-DeRinger on 8GB (6GB usable) GPUs:

1. **Use chunking**: 0.25-second chunks enable batch_size=8 (4× larger than full-length)
2. **Optimize U-Net**: Depth=4, base_channels=32 (sweet spot for 6GB)
3. **Enable AMP**: Float16 mixed precision for 2× memory savings
4. **Overlap-Add inference**: 50% overlap with Hann window for smooth boundaries
5. **Profile regularly**: Use `scripts/profile_memory.py` to monitor usage

With these optimizations, **training on 6GB GPUs is feasible** while maintaining high quality.

For implementation, see:
- `src/neural_deringer/training/chunked_dataloader.py` (implemented)
- `src/neural_deringer/training/chunked_trainer.py` (EPIC 3)
- `scripts/profile_memory.py` (EPIC 3)
- `scripts/optimize_chunk_size.py` (EPIC 3)
