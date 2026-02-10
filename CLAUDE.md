# totton-audio-de-mirroring Development Guide

## Most Important Rule About Language

**Think in English, answer in Japanese**

- When Claude asks questions or provides explanations, use Japanese
- Code comments, docstrings, commit messages: English
- Documentation (this file): English for technical terms, Japanese for explanations if needed

---

## Project Context

### Overview

**totton-audio-de-mirroring** is a Hybrid Neural SR (HNSR) system designed to suppress mirror/aliasing artifacts while preserving time-domain characteristics (transients, phase, group delay).

### Core Concepts

1. **Anti-Hallucination**: No frequency content generation beyond Nyquist limit
2. **Time-Response First**: Focus on flat group delay and transient preservation
3. **Mirror Removal**: Suppress mirror/aliasing artifacts from upsampling, not generate ultrasonic content

### Problem Statement

The system targets **mirror-removal and time-response preservation** rather than aggressive high-frequency generation. The goal is to suppress aliasing artifacts from upsampling paths while maintaining 0–20kHz fidelity (waveform, phase, group delay).

**Target Platform**: Jetson Orin Nano (8GB)
**Target Output**: 705.6kHz (16× Upsampling)
**Input**: 44.1kHz / 16bit or 24bit PCM
**Latency**: Several seconds order acceptable (non-realtime OK)

### Design Intent / Success Criteria

The system aims for **no ringing regression with time-response preservation (transients, phase, group delay) while removing audible unnaturalness from mirror/aliasing**, not "aggressive ultrasonic generation". Since content above 22.05kHz cannot be uniquely reconstructed from 44.1kHz input, the 20kHz+ band is treated as **suppression of unnatural components and safe shaping (zero is acceptable if needed)**, not "reconstruction".

#### Hard Requirements (Failure if not met)

1. **0–20kHz must be identical to input** (waveform, phase, group delay preservation forbidden)
2. **Suppress mirror patterns** and reduce audible "digital harshness/graininess"
3. **20–44kHz can be near-zero** (no forced harmonic generation)
4. **High-frequency total energy cap** (20–44kHz) always enforced (IMD safety)
5. **No ringing regression** on square-wave probes against reference SRC

#### Stage 1 Quantitative Acceptance

- `symmetry_reduction_ratio >= 0.70`
- `hb_energy_cap_violation_rate == 0.0`
- `plateau_ripple_rms_after / before <= 1.10`
- `plateau_ripple_p2p_after / before <= 1.10`
- `overshoot_abs_after - overshoot_abs_before <= 5e-3`
- `ringing_ratio_after - ringing_ratio_before <= 0.0`

### Stage1 Teacher Policy (EPIC #103 / Issue #111)

- Default teacher policy: **raw 88.2kHz (`raw88`)**
- Legacy Bessel teacher is retained as a **comparison baseline**, not the default target policy
- Every Stage1 experiment must encode teacher type in run ID and artifact paths
- Use `docs/stage1_raw_teacher_policy.md` for naming, storage conventions, and migration checklist

---

## Technical Stack

### Core

- **Python**: 3.13+
- **PyTorch**: 2.5+
- **torchaudio**: 2.5+
- **librosa**: Audio processing
- **scipy**: Filter design (Bessel)

### Development Tools

- **uv**: Fast Python package manager
- **ruff**: Fast linter and formatter
- **mypy**: Static type checker
- **pytest**: Testing framework
- **pre-commit**: Git hooks for quality checks

---

## System Architecture

### Two-Stage Hybrid Design

The system uses a two-stage hybrid architecture:

#### Stage 1: Neural Mirror Suppression Engine (NMSE)
44.1kHz → 88.2kHz (2×)

**Purpose**:
- **0–20kHz complete preservation** (guaranteed by structure)
- **Detect and suppress** mirror/aliasing artifacts in 20–44kHz band
- **Manage high-frequency energy** with fixed upper limit to suppress IMD risk

#### Stage 2: DSP High-Rate Interpolation Engine (HIE)
88.2kHz → 705.6kHz (8×)

**Purpose**:
- High-rate conversion of "safe 88.2kHz signal" from Stage 1 to facilitate analog LPF design
- Interpolation that doesn't break time response (minimum-phase-like, gentle slope)

---

## Coding Guidelines

### Type Hints (REQUIRED)

All functions must have type hints:

```python
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
        order: Bessel filter order (default: 10)

    Returns:
        FIR filter coefficients

    Physical Basis:
        Bessel filters have maximally flat group delay, resulting in
        zero overshoot and excellent transient preservation. However,
        insufficient stopband attenuation causes aliasing near Nyquist.
    """
    pass
```

### Docstrings (REQUIRED)

Use Google-style docstrings with a "Physical Basis" section:

```python
def suppress_mirror_artifacts(
    upsampled_signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 20_000.0,
) -> np.ndarray:
    """Suppress mirror/aliasing artifacts using neural network.

    Args:
        upsampled_signal: Bessel-upsampled signal with mirror artifacts
        sample_rate: Sample rate in Hz
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

### Testing (REQUIRED)

Every module must have corresponding tests:

- **Unit tests**: `tests/test_<module>.py`
- **Integration tests**: `tests/integration/test_<feature>.py`
- **Fixtures**: `tests/fixtures/` for test data

---

## Code Style

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
from totton_audio_de_mirroring.models.unet import UNet
```

### Type Annotations

```python
# Good
def process_audio(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 44100
) -> Tuple[np.ndarray, int]:
    pass

# Bad
def process_audio(input_path, output_path, sample_rate=44100):
    pass
```

---

## Directory Structure

```
/path/to/totton-audio-de-mirroring/
├── src/totton_audio_de_mirroring/  # Main package
│   ├── data/                        # Data generation and loading
│   │   ├── generator.py             # Synthetic data generation
│   │   └── bessel_upsample.py       # Bessel FIR filter implementations
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

## Development Commands

### Code Quality

```bash
# Lint and fix
uv run ruff check src/ --fix

# Format code
uv run ruff format src/

# Type check
uv run mypy src/

# Run all checks
uv run pre-commit run --all-files
```

### Testing

```bash
# Run all tests
uv run pytest -v

# Run without slow tests
uv run pytest -m "not slow" -v

# Run with coverage
uv run pytest --cov=totton_audio_de_mirroring --cov-report=html

# Run specific test
uv run pytest tests/test_filters.py::test_bessel_filter -v
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

## Training Data Generation Strategy

### Overview

The training pipeline targets **raw 88.2kHz teacher policy** and learns mirror suppression with high-band safety constraints. Legacy Bessel-teacher runs are kept only for baseline comparison.

### Stage 1: Neural Mirror Suppression Engine

#### Core Strategy: Band Split + Low-Band Bypass

Stage 1 doesn't perform full-band generation but uses **band splitting** with low-band complete bypass.

- `x_full`: 44.1kHz input upsampled to 88.2kHz via 2× reference SRC
- `LB_in = LPF(20kHz, x_full)` (0–20kHz)
- `HB_in = HPF(20kHz, x_full)` (20–44.1kHz)

Output:
- `LB_out = LB_in` (fixed, no modification)
- `HB_out = Suppress(HB_in)` (AI suppression)
- `y_full = LB_out + HB_out`

> LB identity is **guaranteed by structure**, not "hoped for via loss".

#### Model Output: Suppression Mask (Recommended)

AI estimates **suppression mask (gain)**, not generates high-frequency:

- Output: `M ∈ [0, 1]` (time-frequency mask or time-domain gain sequence)
- Application: `HB_out = HB_in ⊙ M`

Intent:
- Strongly suppress components containing mirror patterns
- Preserve others
- HB can be near-zero if needed (avoid "creation")

#### Fixed Safety Constraints (Post-Processing)

After network output, always apply:

1. **Energy Cap (fixed upper limit)**
   - If 20–44kHz total energy exceeds limit, scale/clamp
2. **Envelope Target (fixed envelope)**
   - Project to shape that "gently decays" beyond 20kHz (suppress excessive peaks)
3. **DC/Leak Countermeasures**
   - Reconfirm with HPF to prevent HB leaking into LB side

### Network Goal

Learn the mapping: `Input (degradation-path 88.2kHz with mirror artifacts) → Target (HB anti-mirror target + no-ringing-regression constraints)`

The network learns to:
1. **Remove mirror/aliasing artifacts** in the 20kHz-22.05kHz range
2. **Preserve time-domain characteristics** (transients, phase, group delay)
3. **Maintain 0-20kHz fidelity** without introducing phase distortion

### Baseline and Degradation Policy

- Bessel degradation remains valid as one of degradation profile candidates.
- Bessel teacher should be treated as **baseline-only** in experiment comparison tables.
- Report artifacts must separate `raw88` and `bessel` directories to avoid accidental mixing.

---

## Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Feature
git commit -m "feat: add NMSE architecture with band-split bypass"

# Bug fix
git commit -m "fix: correct phase calculation in mirror suppression"

# Documentation
git commit -m "docs: add mirror removal training guide"

# Tests
git commit -m "test: add integration tests for NMSE"

# Refactor
git commit -m "refactor: extract filter design to separate module"

# Performance
git commit -m "perf: optimize batch processing in data loader"
```

Always include:
```
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

---

## Development Workflow

### Starting New Work

**CRITICAL**: **NEVER commit directly to main branch**. All work MUST be done in feature branches.

**ALWAYS** start from the latest `origin/main` and create a feature branch:

```bash
# 1. Switch to main branch
git checkout main

# 2. Pull latest changes from remote
git pull origin main

# 3. Create feature branch with descriptive name
git checkout -b feat/your-feature-name

# Examples:
git checkout -b feat/nmse-implementation
git checkout -b fix/mirror-detection-bug
git checkout -b refactor/data-loader-optimization
```

**Worktreeを使う場合も必ず`origin/main`から作成すること（ローカル`main`からは作らない）**:

```bash
# 最新のorigin/mainを取得
git fetch origin

# origin/mainからworktreeを作成
git worktree add -b feat/your-feature-name /path/to/worktrees/your-feature origin/main
```

**❌ NEVER DO THIS**:
```bash
# BAD: Working directly on main
git checkout main
# ... make changes ...
git commit -m "some changes"  # ← This commits to main! FORBIDDEN!
```

**✅ ALWAYS DO THIS**:
```bash
# GOOD: Create feature branch first
git checkout main
git pull origin main
git checkout -b feat/my-feature
# ... make changes ...
git commit -m "feat: add my feature"
git push -u origin feat/my-feature
gh pr create  # Create PR for review
```

**Branch naming convention:**
- `feat/`: New features
- `fix/`: Bug fixes
- `refactor/`: Code refactoring
- `test/`: Test additions/improvements
- `docs/`: Documentation updates
- `perf/`: Performance improvements

### Commit and Push Rules

#### NEVER Skip Quality Checks

**PROHIBITED**:
- ❌ `git commit --no-verify` (skips pre-commit hooks)
- ❌ `git push --no-verify` (skips pre-push hooks)
- ❌ Running tests with `-x` or `--exitfirst` and ignoring failures
- ❌ Committing with failing tests or linting errors

**REQUIRED**:
- ✅ All pre-commit hooks must pass (ruff, formatting, basic checks)
- ✅ All pre-push hooks must pass (mypy, pytest)
- ✅ Fix all test failures before pushing
- ✅ Fix all type errors before pushing
- ✅ Fix all linting errors before pushing

#### Fixing Others' Test Failures

**If you encounter test failures from previous commits:**

1. **DO NOT** ignore or skip tests
2. **DO** investigate the root cause
3. **DO** fix the failing tests or notify the original author
4. **DO** document the fix in your commit message

```bash
# Example: Fixing inherited test failure
git commit -m "fix: resolve test failure in data loader

The test_data_loader_batch_size test was failing due to incorrect
batch size calculation inherited from previous commit.

Fixed by adjusting the batch size calculation logic to handle
edge cases with non-divisible dataset sizes.

Related to: commit abc123 (original data loader implementation)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Pre-commit and Pre-push Hooks

**Pre-commit** (runs on `git commit`):
- Trailing whitespace removal
- End-of-file fixing
- YAML/TOML syntax checking
- Large file detection (>10MB)
- Private key detection
- Ruff linting with auto-fix
- Ruff formatting

**Pre-push** (runs on `git push`):
- Mypy static type checking
- Pytest test suite (excluding slow and GPU tests)

**If hooks fail:**
1. Read the error message carefully
2. Fix the issues (don't use `--no-verify`)
3. Re-run the command
4. Verify fixes with manual checks if needed:
   ```bash
   uv run ruff check src/ --fix
   uv run mypy src/
   uv run pytest -m "not slow and not gpu"
   ```

### Pull Request Workflow

1. **Push feature branch**:
   ```bash
   git push -u origin feat/your-feature-name
   ```

2. **Create PR**:
   ```bash
   gh pr create --title "feat: Your Feature Title" --body "$(cat <<'EOF'
   ## Summary
   Brief description of changes

   ## Changes
   - Change 1
   - Change 2

   ## Test Plan
   - [ ] Tested scenario 1
   - [ ] Tested scenario 2

   🤖 Generated with [Claude Code](https://claude.com/claude-code)
   EOF
   )"
   ```

3. **Address review comments**:
   ```bash
   # Make changes
   git add .
   git commit -m "fix: address review comments"
   git push
   ```

4. **Merge PR** (after approval):
   - Use "Squash and merge" for clean history
   - Delete feature branch after merge

### Clean Working Tree

**Before starting work**, ensure clean state:

```bash
# Check for uncommitted changes
git status

# If there are changes, commit or stash them
git stash  # Temporarily save changes
# or
git commit -am "wip: work in progress"

# Then pull latest changes
git pull origin main
```

### Example Complete Workflow

```bash
# 1. Start from latest main
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feat/implement-nmse

# 3. Make changes
# ... edit files ...

# 4. Run quality checks manually (optional, will run on commit/push)
uv run ruff check src/ --fix
uv run mypy src/
uv run pytest -v

# 5. Commit (pre-commit hooks will run)
git add src/totton_audio_de_mirroring/models/nmse.py tests/test_nmse.py
git commit -m "feat: implement NMSE for mirror suppression

Add Neural Mirror Suppression Engine with band-split bypass
and suppression mask output. Includes comprehensive tests for
0-20kHz preservation, mirror suppression, and energy cap enforcement.

Physical basis: Band-split architecture guarantees 0-20kHz preservation
by structure, not loss optimization.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# 6. Push (pre-push hooks will run: mypy + pytest)
git push -u origin feat/implement-nmse

# 7. Create PR
gh pr create --title "feat: Implement NMSE" --body "..."

# 8. After merge, clean up
git checkout main
git pull origin main
git branch -d feat/implement-nmse
```

---

## Model Architecture

### U-Net for Mirror Suppression

**First Choice**: U-Net with skip connections

**Rationale**:
- Proven for signal restoration tasks
- Skip connections preserve high-frequency detail
- Suitable for local artifact correction (mirror removal)

### Chunk Size and Receptive Field

The model architecture must be optimized for the chunk size used during training:

- **Chunk Size**: 0.1s - 0.5s (推奨: 0.25s @ 88.2kHz for Stage 1)
- **Receptive Field**: U-Net depth and kernel size determine the receptive field
- **Guideline**: Receptive field should be ~256-512 samples (sufficient for local mirror removal)

#### Recommended U-Net Configurations by Chunk Size

| Chunk Size | U-Net Depth | Base Channels | Receptive Field | Parameters | Use Case |
|------------|-------------|---------------|-----------------|------------|----------|
| 0.1s (8.8k samples @ 88.2kHz) | 3 | 24 | ~128 samples | ~0.8M | Memory-constrained training |
| 0.25s (22k samples @ 88.2kHz) | 4 | 32 | ~256 samples | ~1.5M | **Recommended balance** |
| 0.5s (44k samples @ 88.2kHz) | 4 | 48 | ~256 samples | ~3.2M | High-quality training |

### Residual Learning

Consider using residual connections to predict only the **suppression mask** rather than the full signal:

```python
class ResidualNMSE(UNet):
    """U-Net with residual learning for mirror suppression.

    Physical Basis:
        Residual learning focuses the model on predicting the suppression
        mask (what to remove) rather than the full signal, accelerating
        convergence and improving mirror removal accuracy.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = super().forward(x)  # Predict suppression mask [0, 1]
        return x * mask  # Apply mask to input
```

**Benefits**:
- Faster convergence (network learns suppression, not reconstruction)
- Better preservation of input characteristics (Bessel group delay)
- Reduced risk of phase distortion

---

## Memory Constraints

### Hardware Assumptions

- **GPU Memory**: 8GB total, ~6GB available for training (2GB reserved for system/CUDA)
- **Sample Rate**: 88.2kHz (Stage 1), 705.6kHz (Stage 2)
- **Precision**: float32 (4 bytes per sample)

### Memory Calculations

**Stage 1 (88.2kHz)**:
- Chunk size 0.25s: 22,050 samples × 4 bytes = 88KB per sample
- Batch size = 32: ~2.8MB audio + ~3GB model/activations = ~3.8GB total
- **Feasible** within 6GB budget

**Stage 2 (705.6kHz)**:
- Chunk size 0.25s: 176,400 samples × 4 bytes = 705KB per sample
- Batch size = 8: ~5.6MB audio + ~2GB model/activations = ~2.6GB total
- **Feasible** with DSP-based approach (lower memory)

### Chunking Strategy

1. **Training-Time Chunking**:
   - Split long audio files into fixed-size chunks (0.25s recommended)
   - Use `ChunkedAudioDataset` for efficient data loading
   - Apply Overlap-Add during training for smooth boundaries

2. **Inference-Time Chunking**:
   - Process long audio files in chunks with overlap (50% recommended)
   - Use Hann window for Perfect Reconstruction
   - GPU-accelerated Overlap-Add for efficiency

3. **Overlap-Add Parameters**:
   - **Window Function**: Hann window (Perfect Reconstruction guarantee)
   - **Overlap**: 50% (balance between artifact reduction and computational cost)

---

## Key Design Decisions

### 1. Mirror Removal vs Generation

**Decision**: Focus on mirror/aliasing suppression, not ultrasonic generation

**Rationale**:
- Content above Nyquist cannot be uniquely reconstructed
- Safety-first approach (energy cap, envelope shaping)
- Preserves time-domain characteristics

### 2. Band-Split Architecture

**Decision**: Low-band bypass (0-20kHz), AI processes only high-band (20-44kHz)

**Rationale**:
- **Guarantees** 0-20kHz preservation by structure
- Reduces AI burden (doesn't need to learn identity mapping)
- Safer for audio quality

### 3. Two-Stage Hybrid

**Decision**: Stage 1 (Neural 44.1→88.2kHz) + Stage 2 (DSP 88.2→705.6kHz)

**Rationale**:
- Stage 1 focuses on critical mirror removal at lower rate (faster training)
- Stage 2 uses efficient DSP for high-rate conversion
- Modular design allows independent optimization

---

## Anti-Patterns to Avoid

### ❌ Don't

- Generate frequencies above Nyquist limit
- Modify 0-20kHz content (breaks time-domain preservation requirement)
- Use high-frequency energy without fixed cap (IMD risk)
- Train without mirror pattern detection mechanism
- Skip physical validation of mirror suppression

### ✅ Do

- Validate 0-20kHz preservation (waveform, phase, group delay)
- Test on edge cases (square waves, impulses, mirror-heavy content)
- Monitor high-frequency energy cap enforcement
- Measure mirror pattern reduction (STFT analysis)
- Document physical assumptions and safety constraints

---

## Questions for Claude

When implementing features, consider:

1. **Physical Validity**: Does this approach respect signal processing theory?
2. **0-20kHz Preservation**: Will this modify the audible band?
3. **Mirror Suppression**: Does this actually reduce mirror/aliasing artifacts?
4. **Safety**: Is high-frequency energy capped? IMD risk managed?
5. **Testability**: Can we measure mirror suppression quantitatively?

---

## References

- [Bessel Filter Theory](https://en.wikipedia.org/wiki/Bessel_filter)
- [Aliasing in Digital Signal Processing](https://en.wikipedia.org/wiki/Aliasing)
- [Group Delay](https://en.wikipedia.org/wiki/Group_delay_and_phase_delay)
- [Intermodulation Distortion](https://en.wikipedia.org/wiki/Intermodulation)
