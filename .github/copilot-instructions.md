# GitHub Copilot Instructions

## Project: Neural-DeRinger

AI-powered audio upsampler for eliminating ringing artifacts using physics-informed neural networks.

---

## Core Principles

1. **Anti-Hallucination**: Never generate frequency content beyond Nyquist limit
2. **Time-Domain First**: Focus on impulse response characteristics
3. **Physics-Informed**: Base all approaches on filter theory

---

## Code Generation Rules

### ALWAYS Include

1. **Type Hints**: All function signatures must have complete type hints
2. **Docstrings**: Google-style with "Physical Basis" section
3. **Input Validation**: Validate all inputs at function entry
4. **Error Handling**: Try/catch for all I/O operations
5. **Tests**: Generate corresponding test cases

### NEVER

- Mutate input arguments (use immutable patterns)
- Skip type hints or docstrings
- Use `eval()` or `exec()` on untrusted input
- Hardcode secrets (API keys, passwords)
- Generate functions >50 lines (break into smaller functions)

---

## Code Style Template

### Function Template

```python
def function_name(
    arg1: Type1,
    arg2: Type2,
    optional_arg: Type3 = default_value
) -> ReturnType:
    """Brief description in imperative mood.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        optional_arg: Description with default mention

    Returns:
        Description of return value

    Raises:
        ValueError: When inputs are invalid
        RuntimeError: When operation fails

    Physical Basis:
        Explanation of the physical/mathematical foundation
        of this implementation and why it works.

    Example:
        >>> result = function_name(value1, value2)
        >>> assert isinstance(result, ExpectedType)
    """
    # Input validation
    if arg1 <= 0:
        raise ValueError(f"arg1 must be positive, got {arg1}")

    # Implementation
    try:
        result = compute_result(arg1, arg2)
    except Exception as e:
        raise RuntimeError(f"Failed to compute: {e}") from e

    return result
```

### Class Template

```python
class ClassName:
    """Brief description of class purpose.

    Attributes:
        attr1: Description of attribute
        attr2: Description of attribute

    Physical Basis:
        Explanation of the underlying theory or approach.
    """

    def __init__(
        self,
        param1: Type1,
        param2: Type2
    ) -> None:
        """Initialize ClassName.

        Args:
            param1: Description
            param2: Description

        Raises:
            ValueError: If parameters are invalid
        """
        # Validation
        if param1 <= 0:
            raise ValueError(f"param1 must be positive, got {param1}")

        # Initialization
        self.attr1 = param1
        self.attr2 = param2
```

---

## Import Order

```python
# Standard library
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Third-party (alphabetical)
import numpy as np
import torch
import torch.nn as nn
from scipy import signal

# Local (alphabetical)
from neural_deringer.data.filters import bessel_filter
from neural_deringer.models.unet import UNet
```

---

## Testing Template

For every function `foo()`, generate `test_foo()`:

```python
# tests/test_module.py
import pytest
import numpy as np
from neural_deringer.module import foo


def test_foo_basic_case():
    """Test foo with standard inputs."""
    # Arrange
    input_data = create_test_input()

    # Act
    result = foo(input_data)

    # Assert
    assert result.shape == expected_shape
    assert np.allclose(result, expected_output, rtol=1e-5)


def test_foo_edge_case_empty():
    """Test foo handles empty input."""
    with pytest.raises(ValueError, match="cannot be empty"):
        foo(np.array([]))


def test_foo_edge_case_invalid():
    """Test foo rejects invalid input."""
    with pytest.raises(ValueError, match="must be positive"):
        foo(np.array([1, 2, 3]), invalid_param=-1)


@pytest.mark.parametrize("input_val,expected", [
    (1.0, 1.0),
    (2.0, 4.0),
    (3.0, 9.0),
])
def test_foo_parametrized(input_val: float, expected: float):
    """Test foo with various inputs."""
    result = foo(input_val)
    assert np.isclose(result, expected)
```

---

## Domain-Specific Guidelines

### Audio Processing

```python
# Always validate sample rates
if sample_rate <= 0:
    raise ValueError(f"sample_rate must be positive, got {sample_rate}")

if cutoff_freq >= sample_rate / 2:
    raise ValueError(f"cutoff_freq must be < Nyquist frequency ({sample_rate/2} Hz)")

# Always check audio shape
if signal.ndim not in (1, 2):
    raise ValueError(f"signal must be 1D or 2D, got {signal.ndim}D")
```

### Filter Implementation

```python
def apply_filter(
    signal: np.ndarray,
    cutoff_freq: float,
    order: int = 8,
    sample_rate: int = 48000
) -> np.ndarray:
    """Apply filter with proper validation.

    Physical Basis:
        [Explain the filter's characteristics and why it's chosen]
    """
    # Validate inputs
    if cutoff_freq >= sample_rate / 2:
        raise ValueError("cutoff_freq must be less than Nyquist frequency")

    # Design filter
    from scipy import signal as sp_signal
    sos = sp_signal.bessel(order, cutoff_freq, fs=sample_rate, output='sos')

    # Apply filter (immutable)
    filtered = sp_signal.sosfilt(sos, signal)

    return filtered
```

### Neural Network Layers

```python
class CustomLayer(nn.Module):
    """Custom neural network layer.

    Physical Basis:
        [Explain the architectural choice and expected behavior]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ) -> None:
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channels must be positive")

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with shape validation."""
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B, C, T), got {x.ndim}D")

        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x
```

---

## Variable Naming

### Good Examples

```python
sample_rate = 48000
cutoff_frequency = 10000
filter_order = 8
num_samples = 1024
batch_size = 32

# Acceptable abbreviations
sr = 48000  # sample rate
n_fft = 2048  # FFT size
eps = 1e-8  # epsilon
```

### Bad Examples (Avoid)

```python
sr = 48000  # Use sample_rate for clarity
n = 1024    # Use num_samples
x = 10000   # Use cutoff_frequency
tmp = []    # Use descriptive name
data = {}   # Use specific name (config, results, etc.)
```

---

## Error Messages

### Good Error Messages

```python
# Specific and actionable
raise ValueError(
    f"cutoff_freq ({cutoff_freq}) must be less than Nyquist frequency "
    f"({sample_rate / 2} Hz)"
)

# Clear constraint
raise ValueError(
    f"Signal must be 1D or 2D, got {signal.ndim}D tensor with shape {signal.shape}"
)
```

### Bad Error Messages (Avoid)

```python
# Too vague
raise ValueError("Invalid input")

# No context
raise ValueError("Wrong shape")

# Exposes sensitive info
raise RuntimeError(f"Database connection failed at {db_host} with password {db_pass}")
```

---

## File Organization

### Module Structure (Prefer Small Files)

```
data/
├── __init__.py
├── filters.py          # Filter implementations (200 lines)
├── generator.py        # Data generation (300 lines)
└── loader.py          # Data loading (250 lines)

# NOT this (avoid large files):
data/
└── processing.py      # 1500 lines - TOO BIG!
```

### When to Split Files

If file exceeds 800 lines, split by:
- **Functionality**: `filters.py` → `filters/bessel.py`, `filters/sinc.py`
- **Layers**: `model.py` → `model/encoder.py`, `model/decoder.py`, `model/blocks.py`
- **Use cases**: `utils.py` → `io_utils.py`, `signal_utils.py`, `plot_utils.py`

---

## Performance Considerations

### Vectorization (Prefer)

```python
# ✅ GOOD: Vectorized
result = signal + np.random.randn(*signal.shape) * noise_level

# ❌ BAD: Loop-based
result = np.zeros_like(signal)
for i in range(len(signal)):
    result[i] = signal[i] + np.random.randn() * noise_level
```

### GPU Operations

```python
def process_on_device(
    signal: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Process signal on specified device.

    Args:
        signal: Input tensor
        device: Compute device (CPU or CUDA)

    Returns:
        Processed signal on same device
    """
    # Move to device
    signal = signal.to(device)

    # Process
    result = expensive_operation(signal)

    # Result is already on device
    return result
```

---

## Configuration Pattern

Use dataclasses for configuration:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    """Training configuration.

    Attributes:
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        num_epochs: Number of training epochs
        checkpoint_dir: Directory for saving checkpoints
    """
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    checkpoint_dir: Path = Path("data/checkpoints")

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
```

---

## Logging Pattern

```python
import logging

logger = logging.getLogger(__name__)

def process_batch(batch: torch.Tensor) -> torch.Tensor:
    """Process batch with logging."""
    logger.debug(f"Processing batch with shape {batch.shape}")

    try:
        result = expensive_computation(batch)
        logger.info(f"Batch processed successfully, output shape: {result.shape}")
        return result
    except Exception as e:
        logger.error(f"Failed to process batch: {e}", exc_info=True)
        raise
```

---

## Context Managers (Prefer)

```python
# ✅ GOOD: Context manager
def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        torch.save(model.state_dict(), f)

# ❌ BAD: Manual close
def save_checkpoint_bad(model: nn.Module, path: Path) -> None:
    """Save model checkpoint."""
    f = open(path, 'wb')
    torch.save(model.state_dict(), f)
    f.close()  # Might not execute if exception occurs
```

---

## Additional Resources

- Full guidelines: [CLAUDE.md](../CLAUDE.md)
- Testing guide: [.claude/rules/testing.md](../.claude/rules/testing.md)
- Style guide: [.claude/rules/coding-style.md](../.claude/rules/coding-style.md)
- Security: [.claude/rules/security.md](../.claude/rules/security.md)
- AI agents guide: [AGENTS.md](../AGENTS.md)

---

**Remember**: All code must have type hints, docstrings, input validation, and tests.
