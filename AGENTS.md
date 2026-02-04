# AI Agents Integration Guide

## Overview

This document provides a structured guide for AI agents (Claude Code, GitHub Copilot, Cursor, etc.) working on the totton-audio-de-mirroring project.

---

## Quick Reference

### Primary Documentation

1. **[CLAUDE.md](./CLAUDE.md)** - Main development guide (READ THIS FIRST)
2. **[.claude/rules/testing.md](./.claude/rules/testing.md)** - Testing guidelines
3. **[.claude/rules/coding-style.md](./.claude/rules/coding-style.md)** - Code style rules
4. **[.claude/rules/security.md](./.claude/rules/security.md)** - Security best practices

### Communication Language

**CRITICAL**: Think in English, respond in Japanese (日本語で回答)

---

## Project Context

### What is totton-audio-de-mirroring?

Hybrid Neural Bessel SR (HNB-SR) system designed to suppress mirror/aliasing artifacts while preserving time-domain characteristics (transients, phase, group delay). The project focuses on **mirror-removal** rather than aggressive high-frequency generation.

**Target Platform**: Jetson Orin Nano (8GB)
**Input**: 44.1kHz / 16bit or 24bit PCM
**Output**: 705.6kHz (16× Upsampling)
**Latency**: Several seconds order acceptable (non-realtime OK)

### Core Principles

1. **Anti-Hallucination**: No frequency content generation beyond Nyquist limit
2. **Time-Response First**: Focus on flat group delay and transient preservation
3. **Mirror Removal**: Suppress mirror/aliasing artifacts, not generate ultrasonic content
4. **Physics-Informed**: Training based on Bessel FIR upsampling with mirror artifacts
5. **0-20kHz Preservation**: Guaranteed by structure (band-split architecture)

### Technical Stack

- Python 3.13+
- PyTorch 2.5+ / torchaudio 2.5+
- uv (package manager)
- ruff (linting/formatting)
- mypy (type checking)
- pytest (testing)

---

## Critical Rules for AI Agents

### 1. Type Hints (REQUIRED)

All functions MUST have type hints:

```python
def bessel_filter(
    signal: np.ndarray,
    cutoff_freq: float,
    order: int = 8
) -> np.ndarray:
    """Apply Bessel filter for zero-ringing upsampling."""
    pass
```

### 2. Docstrings (REQUIRED)

Use Google-style docstrings with "Physical Basis" section:

```python
def suppress_mirror_artifacts(signal: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Suppress mirror/aliasing artifacts using neural network.

    Args:
        signal: Input signal with mirror artifacts
        cutoff_hz: Cutoff frequency for mirror detection

    Returns:
        Signal with mirror artifacts suppressed

    Physical Basis:
        Bessel FIR upsampling creates mirror/aliasing artifacts in the
        20-22kHz range due to insufficient stopband attenuation. The
        neural network learns to identify and suppress these artifacts
        while preserving 0-20kHz content.
    """
    pass
```

### 3. Testing (REQUIRED)

- Every module needs tests in `tests/test_<module>.py`
- Target: 80%+ coverage
- Critical paths: 90%+ coverage

### 4. Immutability (CRITICAL)

NEVER mutate input arguments:

```python
# ❌ BAD
def add_noise(signal: np.ndarray, noise_level: float) -> np.ndarray:
    signal += np.random.randn(*signal.shape) * noise_level
    return signal

# ✅ GOOD
def add_noise(signal: np.ndarray, noise_level: float) -> np.ndarray:
    noise = np.random.randn(*signal.shape) * noise_level
    return signal + noise
```

### 5. Input Validation (ALWAYS)

Validate all inputs at function entry:

```python
def upsample_audio(signal: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if signal.ndim not in (1, 2):
        raise ValueError(f"Signal must be 1D or 2D, got {signal.ndim}D")
    if source_sr <= 0:
        raise ValueError(f"source_sr must be positive, got {source_sr}")
    # ... rest of validation
    return _upsample_impl(signal, source_sr, target_sr)
```

### 6. Error Handling (ALWAYS)

All I/O operations need try/catch:

```python
def load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    return checkpoint
```

---

## Development Workflow

### Starting Work

```bash
# 1. Switch to main
git checkout main

# 2. Pull latest
git pull origin main

# 3. Create feature branch
git checkout -b feat/your-feature-name
```

### Branch Naming

- `feat/`: New features
- `fix/`: Bug fixes
- `refactor/`: Code refactoring
- `test/`: Test improvements
- `docs/`: Documentation
- `perf/`: Performance

### Quality Checks

**NEVER** skip pre-commit/pre-push hooks:

```bash
# ❌ PROHIBITED
git commit --no-verify
git push --no-verify

# ✅ REQUIRED
# Let hooks run automatically
git commit -m "feat: your message"
git push
```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add NMSE architecture with band-split bypass

Implement Neural Mirror Suppression Engine with band-split architecture
that guarantees 0-20kHz preservation by structure. Includes tests for
mirror suppression and energy cap enforcement.

Physical basis: Band-split architecture bypasses 0-20kHz, AI processes
only 20-44kHz for mirror removal, eliminating risk of audible band modification.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Code Quality Standards

### File Size

- **Target**: 200-400 lines per file
- **Maximum**: 800 lines per file
- If exceeded, split into multiple files

### Function Length

- **Target**: 10-20 lines per function
- **Maximum**: 50 lines per function
- If exceeded, extract sub-functions

### Nesting Depth

- **Maximum**: 3 levels of indentation
- Use early returns to reduce nesting

### Import Order

```python
# Standard library
import os
from pathlib import Path
from typing import Optional, Tuple

# Third-party
import numpy as np
import torch
import torch.nn as nn
from scipy import signal

# Local
from totton_audio_de_mirroring.data.filters import bessel_filter
from totton_audio_de_mirroring.models.nmse import NMSE
```

---

## Directory Structure

```
/path/to/totton-audio-de-mirroring/
├── src/totton_audio_de_mirroring/  # Main package
│   ├── data/                        # Data generation and loading
│   │   ├── generator.py             # Synthetic data generation
│   │   └── filters.py               # Bessel/Sinc implementations
│   ├── models/                      # Neural network architectures
│   │   ├── nmse.py                  # Neural Mirror Suppression Engine
│   │   └── unet.py                  # U-Net for audio processing
│   ├── training/                    # Training loop and utilities
│   │   ├── trainer.py               # Main training logic
│   │   └── metrics.py               # Loss functions and metrics
│   └── inference/                   # Production inference
│       └── upsampler.py             # Real-time upsampling
├── tests/                           # Unit and integration tests
├── scripts/                         # Training/evaluation scripts
├── notebooks/                       # Jupyter notebooks for analysis
├── data/                            # Data storage (gitignored)
│   ├── synthetic/                   # Generated training data
│   └── checkpoints/                 # Model checkpoints
└── docs/                            # Additional documentation
```

---

## Common Commands

### Code Quality

```bash
# Format code
uv run ruff format src/

# Lint and fix
uv run ruff check src/ --fix

# Type check
uv run mypy src/

# All checks
uv run pre-commit run --all-files
```

### Testing

```bash
# Run all tests
uv run pytest -v

# Fast tests only
uv run pytest -m "not slow and not gpu" -v

# With coverage
uv run pytest --cov=totton_audio_de_mirroring --cov-report=html
```

### Training

```bash
# Generate synthetic data
uv run python scripts/generate_data.py --num-samples 10000

# Train Stage 1 (NMSE)
uv run python scripts/train_stage1.py --config configs/nmse_base.yaml

# Train Stage 2 (HIE)
uv run python scripts/train_stage2.py --config configs/hie_base.yaml

# Evaluate model
uv run python scripts/evaluate.py --checkpoint data/checkpoints/best.pth
```

---

## Anti-Patterns to Avoid

### ❌ Don't

- Generate frequencies above Nyquist limit
- Modify 0-20kHz content (breaks preservation requirement)
- Mutate input arguments
- Skip quality checks (`--no-verify`)
- Use `eval()` or `exec()` on untrusted input
- Hardcode secrets (API keys, passwords)
- Create files unnecessarily (prefer editing existing files)
- Ignore test failures
- Use deep nesting (>3 levels)
- Write functions >50 lines
- Skip docstrings or type hints
- Use high-frequency energy without fixed cap (IMD risk)

### ✅ Do

- Validate 0-20kHz preservation (waveform, phase, group delay)
- Validate inputs at function entry
- Use error handling for I/O operations
- Write tests for all new code
- Follow immutability principles
- Document physical assumptions
- Use environment variables for config
- Keep functions short and focused
- Use descriptive variable names
- Add "Physical Basis" section to docstrings
- Enforce high-frequency energy cap
- Measure mirror suppression quantitatively

---

## Physics Context

### Two-Stage Hybrid Architecture

**Stage 1: Neural Mirror Suppression Engine (NMSE)**
- 44.1kHz → 88.2kHz (2×)
- **0-20kHz complete preservation** (guaranteed by structure)
- **Detect and suppress** mirror/aliasing artifacts in 20-44kHz band
- **Manage high-frequency energy** with fixed upper limit

**Stage 2: DSP High-Rate Interpolation Engine (HIE)**
- 88.2kHz → 705.6kHz (8×)
- High-rate conversion using efficient DSP
- Interpolation that doesn't break time response

### Band-Split Architecture (Stage 1)

Stage 1 uses **band splitting** with low-band complete bypass:

- `x_full`: 44.1kHz input upsampled to 88.2kHz via 2× reference SRC
- `LB_in = LPF(20kHz, x_full)` (0–20kHz)
- `HB_in = HPF(20kHz, x_full)` (20–44.1kHz)

Output:
- `LB_out = LB_in` (fixed, no modification)
- `HB_out = Suppress(HB_in)` (AI suppression)
- `y_full = LB_out + HB_out`

> LB identity is **guaranteed by structure**, not "hoped for via loss".

### Network Goal

Learn the mapping: `Input (Bessel-upsampled with mirror artifacts) → Target (Clean)`

The network learns to:
- **Remove mirror/aliasing artifacts** in 20kHz-22.05kHz range
- **Preserve time-domain characteristics** (flat group delay from Bessel)
- **Maintain 0-20kHz fidelity** without introducing phase distortion
- **Enforce energy cap** for 20-44kHz band (IMD safety)

### Memory Constraints

**Hardware**: 8GB GPU (6GB usable for training)
**Strategy**: Chunked processing (0.25 seconds recommended @ 88.2kHz for Stage 1)
- Chunk size 0.25s: 22,050 samples × 4 bytes = 88KB per sample
- Batch size = 32: ~2.8MB audio + ~3GB model/activations = ~3.8GB total
**Overlap-Add**: 50% overlap with Hann window for inference

---

## Hard Requirements

1. **0–20kHz must be identical to input** (waveform, phase, group delay preservation)
2. **Suppress mirror patterns** and reduce audible "digital harshness/graininess"
3. **20–44kHz can be near-zero** (no forced harmonic generation)
4. **High-frequency total energy cap** (20–44kHz) always enforced (IMD safety)

---

## Security Checklist

- [ ] No hardcoded secrets
- [ ] Input validation on all user inputs
- [ ] Parameterized queries (no string concatenation)
- [ ] Safe error messages (no sensitive info leak)
- [ ] File path validation (no path traversal)
- [ ] Pinned dependency versions
- [ ] Environment variables for configuration
- [ ] No `eval()` or `exec()` on untrusted input

---

## When to Ask Questions

Consider asking the user when:

1. **Physical Validity**: Does this approach respect signal processing theory?
2. **0-20kHz Preservation**: Will this modify the audible band?
3. **Mirror Suppression**: Does this actually reduce mirror/aliasing artifacts?
4. **Unclear Requirements**: Multiple valid interpretations exist
5. **Architectural Decisions**: Choice between patterns/technologies
6. **Safety**: Is high-frequency energy capped? IMD risk managed?

---

## Additional Resources

- [Bessel Filter Theory](https://en.wikipedia.org/wiki/Bessel_filter)
- [Aliasing in Digital Signal Processing](https://en.wikipedia.org/wiki/Aliasing)
- [Group Delay](https://en.wikipedia.org/wiki/Group_delay_and_phase_delay)
- [Intermodulation Distortion](https://en.wikipedia.org/wiki/Intermodulation)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## Quick Start for New AI Agents

1. **Read [CLAUDE.md](./CLAUDE.md)** - Full development guide
2. **Check [.claude/rules/](./.claude/rules/)** - Specific guidelines
3. **Review recent commits** - `git log --oneline -10`
4. **Check open issues** - `gh issue list`
5. **Run tests** - `uv run pytest -v`
6. **Start coding** - Follow workflow above

---

**Remember**: Think in English, respond in Japanese (日本語で回答してください)
