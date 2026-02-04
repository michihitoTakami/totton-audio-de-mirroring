# Architecture Diagrams

This document provides visual representations of the Neural-DeRinger pipeline architecture.

## Table of Contents

- [Data Generation Flow](#data-generation-flow)
- [Training Pipeline](#training-pipeline)
- [Inference Pipeline](#inference-pipeline)
- [Memory Allocation](#memory-allocation)

---

## Data Generation Flow

### Overview Diagram

```mermaid
graph TD
    A[Start: Generate High-Res Audio] --> B[705.6kHz Synthetic Audio]
    B --> C[Apply Bessel Filter<br/>20kHz Cutoff, Order 10]
    C --> D[Target: Clean 705.6kHz]

    D --> E[SoX Downsample<br/>Sinc Interpolation]
    E --> F[44.1kHz Audio<br/>16x Downsampled]

    F --> G[Bessel FIR Upsample<br/>10k-20k taps]
    G --> H[Input: 705.6kHz with Aliasing<br/>20-22kHz artifacts]

    D --> I[Training Pair]
    H --> I

    I --> J[Save to Disk<br/>NPY format]

    style D fill:#90EE90
    style H fill:#FFB6C1
    style I fill:#87CEEB
```

### Detailed Step-by-Step

1. **Generate Synthetic Audio (705.6kHz)**
   - Random oscillators (Sine, Square, Saw, Noise)
   - Random envelopes (ADSR)
   - Duration: 0.25s - 1.0s

2. **Apply Bessel Filter (Target Generation)**
   - Cutoff: 20kHz
   - Order: 8-12
   - Result: Zero overshoot, flat group delay
   - **Output**: Target (clean 705.6kHz)

3. **SoX Downsample (44.1kHz)**
   - Method: Sinc interpolation (high quality)
   - Ratio: 16× downsample (705.6kHz → 44.1kHz)
   - Result: Standard CD-quality audio

4. **Bessel FIR Upsample (Input Generation)**
   - Taps: 10k-20k
   - Method: Zero-stuffing + FIR convolution
   - Result: Aliasing artifacts in 20-22kHz range
   - **Output**: Input (705.6kHz with aliasing)

5. **Training Pair**
   - Input: Bessel-upsampled (with aliasing)
   - Target: Bessel-filtered (clean)
   - Format: NPY (float32)

---

## Training Pipeline

### Training Loop Diagram

```mermaid
graph LR
    A[Load Training Data<br/>ChunkedAudioDataset] --> B[Batch: 8 chunks<br/>0.25s each]
    B --> C[GPU Memory<br/>~2.6GB total]

    C --> D[U-Net Model<br/>Depth=4, Channels=32]
    D --> E[Forward Pass<br/>Predict clean output]

    E --> F[Loss Calculation<br/>L1 + Spectral + AntiAliasing]
    F --> G[Backward Pass<br/>Compute gradients]

    G --> H[Optimizer Step<br/>Adam, lr=1e-4]
    H --> I[Update Weights]

    I --> J{More Batches?}
    J -->|Yes| A
    J -->|No| K[Validation]

    K --> L[Save Checkpoint]
    L --> M{More Epochs?}
    M -->|Yes| A
    M -->|No| N[Training Complete]

    style C fill:#FFE4B5
    style D fill:#87CEEB
    style F fill:#FFB6C1
```

### Memory Layout (Training)

```
GPU Memory (6GB available):

┌─────────────────────────────────────────────┐
│ Input Batch (8 × 0.25s @ 705.6kHz)   5.6MB │
├─────────────────────────────────────────────┤
│ U-Net Parameters (1.5M params)         6MB │
├─────────────────────────────────────────────┤
│ Gradients (same as params)             6MB │
├─────────────────────────────────────────────┤
│ Optimizer State (Adam: 2× params)    12MB │
├─────────────────────────────────────────────┤
│ Activations (intermediate outputs) 1,500MB │
├─────────────────────────────────────────────┤
│ Loss & Misc                           20MB │
└─────────────────────────────────────────────┘
Total:                                ~1,550MB

Remaining: ~4,500MB (buffer for spikes)
```

### Chunked Training Process

```mermaid
sequenceDiagram
    participant D as Dataset
    participant L as DataLoader
    participant M as Model (U-Net)
    participant O as Optimizer

    Note over D: Load full audio files<br/>(1 second each)

    D->>L: Split into chunks<br/>(0.25s with 50% overlap)
    L->>L: Shuffle chunks

    loop Each Batch
        L->>M: Batch of 8 chunks
        M->>M: Forward pass
        M->>M: Compute loss
        M->>O: Gradients
        O->>M: Update weights
    end

    Note over M: Fixed memory footprint<br/>Independent of audio length
```

---

## Inference Pipeline

### Overlap-Add Inference

```mermaid
graph TD
    A[Input Audio<br/>Arbitrary Length] --> B[Split into Chunks<br/>0.25s with 50% overlap]

    B --> C1[Chunk 1]
    B --> C2[Chunk 2]
    B --> C3[Chunk 3]
    B --> C4[...]

    C1 --> D1[U-Net Process]
    C2 --> D2[U-Net Process]
    C3 --> D3[U-Net Process]
    C4 --> D4[U-Net Process]

    D1 --> E1[Apply Hann Window]
    D2 --> E2[Apply Hann Window]
    D3 --> E3[Apply Hann Window]
    D4 --> E4[Apply Hann Window]

    E1 --> F[Overlap-Add<br/>Combine chunks]
    E2 --> F
    E3 --> F
    E4 --> F

    F --> G[Output Audio<br/>Same Length as Input]

    style A fill:#90EE90
    style F fill:#FFD700
    style G fill:#87CEEB
```

### Overlap-Add Visualization

```
Input Audio: [=====================================]
                     (10 seconds)

Split into Chunks (0.25s, 50% overlap):
[====]
    [====]
        [====]
            [====]
                ...

Process Each Chunk:
[OUT]
    [OUT]
        [OUT]
            [OUT]
                ...

Apply Hann Window:
 /\
   /\
     /\
       /\
         ...

Overlap-Add:
 /\
  +/\
    +/\
      +/\
        +...
= [=====================================]
         (10 seconds, smooth)
```

### Real-Time Processing

```mermaid
graph LR
    A[Audio Input<br/>Streaming] --> B[Buffer: 0.25s]
    B --> C{Buffer Full?}
    C -->|No| A
    C -->|Yes| D[Process Chunk<br/>U-Net]
    D --> E[Apply Window]
    E --> F[Overlap-Add]
    F --> G[Audio Output<br/>Streaming]
    G --> A

    style A fill:#90EE90
    style D fill:#87CEEB
    style G fill:#FFD700
```

---

## Memory Allocation

### Scenario Comparison

```mermaid
graph TD
    subgraph Full-Length Training
        A1[1.0s Audio<br/>2.8MB per sample] --> B1[Batch Size = 2<br/>5.6MB total]
        B1 --> C1[U-Net Depth=3<br/>Base=32]
        C1 --> D1[Activations<br/>~600MB]
        D1 --> E1[Total: ~625MB<br/>Small batch slow]
    end

    subgraph Chunked Training
        A2[0.25s Chunk<br/>0.7MB per sample] --> B2[Batch Size = 8<br/>5.6MB total]
        B2 --> C2[U-Net Depth=4<br/>Base=32]
        C2 --> D2[Activations<br/>~1,500MB]
        D2 --> E2[Total: ~1,550MB<br/>Large batch fast]
    end

    style E1 fill:#FFB6C1
    style E2 fill:#90EE90
```

### Memory Scaling

```
Chunk Size vs Memory Usage (Batch Size = 8):

1.0s  ┤ ██████████████████████ 5.6GB (batch=2 only)
      │
0.5s  ┤ ████████████ 2.8GB (batch=4)
      │
0.25s ┤ ██████ 1.5GB (batch=8) ← Recommended
      │
0.1s  ┤ ██ 0.9GB (batch=16)
      │
      └────────────────────────────────────────
       0GB    1GB    2GB    3GB    4GB    5GB    6GB
                    Available GPU Memory
```

### Model Architecture Impact

```
U-Net Configuration vs Memory (0.25s chunks, batch=8):

Depth=5  ┤ ██████████████ 3.2GB (risky)
Base=32  │
         │
Depth=4  ┤ ████████ 1.5GB (recommended)
Base=48  │
         │
Depth=4  ┤ █████ 1.1GB (balanced) ← Recommended
Base=32  │
         │
Depth=3  ┤ ███ 0.9GB (conservative)
Base=32  │
         │
Depth=3  ┤ ██ 0.7GB (minimal)
Base=24  │
         └────────────────────────────────────────
          0GB    1GB    2GB    3GB    4GB
                    GPU Memory Usage
```

---

## Pipeline Comparison

### Traditional Sinc Upsampling

```
Input (44.1kHz)
    ↓ Zero-stuffing
Intermediate (705.6kHz, zeros)
    ↓ Sinc FIR Filter (ideal frequency response)
Output (705.6kHz)
    ✓ Flat frequency response
    ✗ Severe ringing (Gibbs phenomenon)
```

### Minimum Phase FIR Upsampling

```
Input (44.1kHz)
    ↓ Zero-stuffing
Intermediate (705.6kHz, zeros)
    ↓ Minimum Phase FIR (640k taps)
Output (705.6kHz)
    ✓ Good frequency response
    ✓ Minimal ringing
    ✗ Nonlinear group delay (phase distortion)
    ✗ Very high computational cost
```

### Neural-DeRinger (Bessel FIR + NN)

```
Input (44.1kHz)
    ↓ Zero-stuffing
Intermediate (705.6kHz, zeros)
    ↓ Bessel FIR (10k-20k taps)
Aliased (705.6kHz with 20-22kHz artifacts)
    ↓ U-Net Neural Network
Output (705.6kHz)
    ✓ Flat group delay (from Bessel)
    ✓ Zero overshoot (from Bessel)
    ✓ Clean frequency response (from NN)
    ✓ Reasonable computational cost
    ✗ Requires training
```

---

## Training Data Flow

```mermaid
graph LR
    subgraph Data Generation
        A[Synthetic Audio<br/>Generator] --> B[705.6kHz<br/>High-Res]
        B --> C[Bessel Filter<br/>Target]
        B --> D[SoX Downsample<br/>44.1kHz]
        D --> E[Bessel FIR<br/>Upsample]
        E --> F[Input with<br/>Aliasing]
    end

    subgraph Storage
        C --> G[targets.npy<br/>Float32 Array]
        F --> H[inputs.npy<br/>Float32 Array]
        G --> I[metadata.jsonl<br/>Parameters]
        H --> I
    end

    subgraph Training
        G --> J[DataLoader<br/>Chunking]
        H --> J
        J --> K[Batch<br/>8 samples]
        K --> L[U-Net<br/>Training]
        L --> M[Checkpoint<br/>best.pth]
    end

    subgraph Inference
        M --> N[Load Model]
        N --> O[Process<br/>Real Audio]
        O --> P[Upsampled<br/>Output]
    end

    style C fill:#90EE90
    style F fill:#FFB6C1
    style L fill:#87CEEB
    style P fill:#FFD700
```

---

## Summary

Key architectural decisions:

1. **Data Generation**: Bessel FIR upsampling intentionally creates aliasing for NN to learn
2. **Training**: Chunked processing (0.25s) enables larger batch sizes on limited GPU memory
3. **Inference**: Overlap-Add (50% overlap, Hann window) for smooth boundaries
4. **Memory**: U-Net depth=4, base_channels=32 optimized for 6GB GPUs

This architecture achieves the "time-domain first" philosophy while maintaining practical computational constraints.
