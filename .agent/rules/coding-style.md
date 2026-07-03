# Coding Style Guidelines

## CRITICAL: Immutability

**Prefer immutable data structures and avoid mutation:**

```python
# ❌ BAD: Mutation
def add_noise(signal: np.ndarray, noise_level: float) -> np.ndarray:
    signal += np.random.randn(*signal.shape) * noise_level
    return signal

# ✅ GOOD: Immutable
def add_noise(signal: np.ndarray, noise_level: float) -> np.ndarray:
    """Add Gaussian noise to signal without mutation.

    Args:
        signal: Input signal (not modified)
        noise_level: Noise standard deviation

    Returns:
        New signal with added noise
    """
    noise = np.random.randn(*signal.shape) * noise_level
    return signal + noise
```

---

## File Size and Organization

### Small Files, Many Files

**Target**: 200-400 lines per file
**Maximum**: 800 lines per file

If a file exceeds 800 lines, split it:

```
# Before (1500 lines)
models/unet.py

# After
models/unet/
├── __init__.py
├── architecture.py  # Core U-Net
├── blocks.py        # Encoder/Decoder blocks
└── layers.py        # Custom layers
```

### Single Responsibility

Each file should have ONE clear purpose:

```python
# ✅ GOOD: Single responsibility
# data/filters.py
def bessel_filter(...): pass
def sinc_interpolate(...): pass
def butterworth_filter(...): pass

# ❌ BAD: Multiple responsibilities
# utils.py
def bessel_filter(...): pass
def load_config(...): pass
def plot_waveform(...): pass
def train_model(...): pass
```

---

## Error Handling

### ALWAYS Use Try/Catch

```python
# ❌ BAD: No error handling
def load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    return torch.load(path)

# ✅ GOOD: Explicit error handling
def load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    """Load model checkpoint with error handling.

    Args:
        path: Checkpoint file path

    Returns:
        Model state dictionary

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint is corrupted
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Invalid checkpoint format: missing model_state_dict")

    return checkpoint
```

---

## Input Validation

### ALWAYS Validate Inputs

```python
def upsample_audio(
    signal: np.ndarray,
    source_sr: int,
    target_sr: int
) -> np.ndarray:
    """Upsample audio signal with validation.

    Args:
        signal: Input audio signal
        source_sr: Source sample rate
        target_sr: Target sample rate

    Returns:
        Upsampled signal

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate signal
    if signal.ndim not in (1, 2):
        raise ValueError(f"Signal must be 1D or 2D, got {signal.ndim}D")

    if signal.size == 0:
        raise ValueError("Signal cannot be empty")

    # Validate sample rates
    if source_sr <= 0:
        raise ValueError(f"source_sr must be positive, got {source_sr}")

    if target_sr <= 0:
        raise ValueError(f"target_sr must be positive, got {target_sr}")

    if target_sr <= source_sr:
        raise ValueError(
            f"target_sr ({target_sr}) must be greater than source_sr ({source_sr})"
        )

    # Perform upsampling
    return _upsample_impl(signal, source_sr, target_sr)
```

---

## Function Length

**Target**: 10-20 lines per function
**Maximum**: 50 lines per function

If a function exceeds 50 lines, extract sub-functions:

```python
# ❌ BAD: Long function (80 lines)
def train_epoch(model, loader, optimizer):
    # 80 lines of training logic...
    pass

# ✅ GOOD: Split into smaller functions
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer
) -> float:
    """Train one epoch.

    Args:
        model: Neural network model
        loader: Training data loader
        optimizer: Optimizer

    Returns:
        Average loss
    """
    total_loss = 0.0

    for batch in loader:
        loss = train_step(model, batch, optimizer)
        total_loss += loss

    return total_loss / len(loader)


def train_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer
) -> float:
    """Single training step.

    Args:
        model: Neural network model
        batch: (input, target) tuple
        optimizer: Optimizer

    Returns:
        Batch loss
    """
    inputs, targets = batch
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = compute_loss(outputs, targets)

    loss.backward()
    optimizer.step()

    return loss.item()
```

---

## Avoid Deep Nesting

**Maximum nesting**: 3 levels

```python
# ❌ BAD: Deep nesting (4 levels)
def process_files(directory: Path) -> None:
    if directory.exists():
        for file in directory.iterdir():
            if file.suffix == ".wav":
                try:
                    data = load_audio(file)
                    if data.size > 0:
                        process_audio(data)
                except Exception:
                    pass

# ✅ GOOD: Early returns (2 levels max)
def process_files(directory: Path) -> None:
    """Process all audio files in directory.

    Args:
        directory: Directory containing audio files
    """
    if not directory.exists():
        return

    audio_files = [f for f in directory.iterdir() if f.suffix == ".wav"]

    for file in audio_files:
        try:
            process_audio_file(file)
        except Exception as e:
            logger.warning(f"Failed to process {file}: {e}")


def process_audio_file(file: Path) -> None:
    """Process single audio file.

    Args:
        file: Audio file path
    """
    data = load_audio(file)

    if data.size == 0:
        logger.warning(f"Empty audio file: {file}")
        return

    process_audio(data)
```

---

## Type Hints

**ALWAYS** use type hints:

```python
from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import torch

# ✅ GOOD: Full type hints
def generate_training_data(
    num_samples: int,
    sample_rate: int,
    duration: float,
    output_dir: Path,
    noise_level: Optional[float] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic training data.

    Args:
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        output_dir: Output directory
        noise_level: Optional noise level (default: no noise)

    Returns:
        List of (input, target) pairs
    """
    pass
```

---

## Naming Conventions

### Variables and Functions

- **snake_case**: `sample_rate`, `load_checkpoint`
- **Descriptive**: Avoid abbreviations unless standard (`sr` for sample rate is OK)

```python
# ❌ BAD
sr = 48000
n = 1024
def proc(x): pass

# ✅ GOOD
sample_rate = 48000
num_samples = 1024
def process_audio(signal: np.ndarray) -> np.ndarray: pass
```

### Classes

- **PascalCase**: `UNet`, `AudioDataset`, `TrainingConfig`

```python
class AudioUpsampler:
    """Neural network-based audio upsampler."""
    pass

class TrainingConfig:
    """Training configuration parameters."""
    pass
```

### Constants

- **UPPER_SNAKE_CASE**: `MAX_BATCH_SIZE`, `DEFAULT_SAMPLE_RATE`

```python
MAX_BATCH_SIZE = 32
DEFAULT_SAMPLE_RATE = 48000
CHECKPOINT_DIR = Path("data/checkpoints")
```

---

## Docstrings

**ALWAYS** use Google-style docstrings:

```python
def bessel_filter(
    signal: np.ndarray,
    cutoff_freq: float,
    order: int = 8,
    sample_rate: int = 48000
) -> np.ndarray:
    """Apply Bessel filter for zero-ringing response.

    Args:
        signal: Input audio signal
        cutoff_freq: Cutoff frequency in Hz
        order: Filter order (higher = sharper cutoff)
        sample_rate: Sample rate in Hz

    Returns:
        Filtered signal with minimal phase distortion

    Raises:
        ValueError: If cutoff_freq >= sample_rate / 2

    Physical Basis:
        Bessel filters have maximally flat group delay in passband,
        resulting in zero overshoot in step response. This makes them
        ideal for audio applications where phase linearity is critical.

    Example:
        >>> signal = np.random.randn(48000)
        >>> filtered = bessel_filter(signal, cutoff_freq=10000)
        >>> assert filtered.shape == signal.shape
    """
    pass
```

---

## Code Quality Checklist

Before committing, verify:

- [ ] **Readability**: Can another developer understand this in 30 seconds?
- [ ] **Short functions**: Most functions < 20 lines, none > 50 lines
- [ ] **Shallow nesting**: Maximum 3 levels of indentation
- [ ] **Type hints**: All function signatures have type hints
- [ ] **Docstrings**: All public functions have docstrings
- [ ] **Error handling**: All I/O operations have try/catch
- [ ] **Input validation**: All inputs validated at function entry
- [ ] **Immutability**: No in-place modification of arguments
- [ ] **Single responsibility**: Each file/function does ONE thing
- [ ] **No magic numbers**: Constants defined at module level

---

## Code Review Questions

When reviewing code (yours or others'), ask:

1. **Can I understand this without comments?**
   - If no, refactor for clarity or add comments

2. **Does this function do ONE thing?**
   - If no, split into multiple functions

3. **What happens if inputs are invalid?**
   - If crashes, add input validation

4. **What happens if I/O fails?**
   - If crashes, add error handling

5. **Would I be comfortable debugging this at 3am?**
   - If no, simplify

---

## Anti-Patterns to Avoid

### ❌ God Objects

```python
# BAD: Class that does everything
class AudioProcessor:
    def load_file(self): pass
    def apply_filter(self): pass
    def train_model(self): pass
    def plot_results(self): pass
    def save_checkpoint(self): pass
```

### ❌ Magic Numbers

```python
# BAD
signal = signal * 0.5
filtered = filter(signal, 10000, 8)

# GOOD
NORMALIZATION_FACTOR = 0.5
DEFAULT_CUTOFF_HZ = 10000
DEFAULT_FILTER_ORDER = 8

signal = signal * NORMALIZATION_FACTOR
filtered = filter(signal, DEFAULT_CUTOFF_HZ, DEFAULT_FILTER_ORDER)
```

### ❌ Premature Optimization

```python
# BAD: Optimizing before profiling
def process(data):
    # 100 lines of complex optimized code
    pass

# GOOD: Simple first, optimize if needed
def process(data: np.ndarray) -> np.ndarray:
    """Process audio data."""
    return simple_transform(data)
```

---

## Tools

Run before every commit:

```bash
# Format code
uv run ruff format src/

# Lint and fix
uv run ruff check src/ --fix

# Type check
uv run mypy src/

# Test
uv run pytest -v
```
