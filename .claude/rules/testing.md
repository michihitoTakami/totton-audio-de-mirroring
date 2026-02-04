# Testing Guidelines

## Minimum Coverage

**Target**: 80% code coverage
**Acceptable**: 70% for experimental modules
**Required**: 90%+ for critical paths (filter implementations, data loaders)

```bash
# Check coverage
uv run pytest --cov=neural_deringer --cov-report=term --cov-report=html

# View HTML report
open htmlcov/index.html
```

---

## Test Types

### 1. Unit Tests

Test individual functions in isolation:

```python
# tests/test_filters.py
import numpy as np
import pytest
from neural_deringer.data.filters import bessel_filter


def test_bessel_filter_shape():
    """Test that Bessel filter preserves signal shape."""
    signal = np.random.randn(48000)
    filtered = bessel_filter(signal, cutoff_freq=10000)
    assert filtered.shape == signal.shape


def test_bessel_filter_zero_overshoot():
    """Test that Bessel filter has zero overshoot on step input."""
    # Create step function
    signal = np.concatenate([np.zeros(1000), np.ones(1000)])

    # Apply filter
    filtered = bessel_filter(signal, cutoff_freq=5000, sample_rate=48000)

    # Check no overshoot (max value should be <= 1.0)
    assert np.max(filtered) <= 1.01  # Allow 1% tolerance


def test_bessel_filter_invalid_cutoff():
    """Test that invalid cutoff frequency raises ValueError."""
    signal = np.random.randn(1000)

    with pytest.raises(ValueError, match="cutoff_freq must be less than Nyquist"):
        bessel_filter(signal, cutoff_freq=30000, sample_rate=48000)
```

### 2. Integration Tests

Test multiple components working together:

```python
# tests/integration/test_data_pipeline.py
import pytest
from pathlib import Path
from neural_deringer.data.generator import SyntheticDataGenerator
from neural_deringer.data.filters import bessel_filter, sinc_downsample


def test_synthetic_data_generation(tmp_path: Path):
    """Test complete synthetic data generation pipeline."""
    # Setup
    generator = SyntheticDataGenerator(
        sample_rate=48000,
        duration=1.0,
        num_samples=10
    )

    # Generate data
    dataset = generator.generate(output_dir=tmp_path)

    # Validate
    assert len(dataset) == 10
    for input_signal, target_signal in dataset:
        assert input_signal.shape == target_signal.shape
        assert input_signal.shape[0] == 48000  # 1 second at 48kHz
```

### 3. End-to-End Tests

Test complete workflows:

```python
# tests/e2e/test_training.py
import pytest
import torch
from pathlib import Path
from neural_deringer.models.unet import UNet
from neural_deringer.training.trainer import Trainer


@pytest.mark.slow
def test_training_one_epoch(tmp_path: Path):
    """Test that training runs for one epoch without errors."""
    # Setup model
    model = UNet(in_channels=1, out_channels=1)

    # Setup trainer
    trainer = Trainer(
        model=model,
        output_dir=tmp_path,
        batch_size=4,
        num_epochs=1
    )

    # Generate minimal dataset
    train_data = generate_dummy_data(num_samples=10)

    # Train
    trainer.train(train_data)

    # Verify checkpoint saved
    checkpoints = list(tmp_path.glob("*.pth"))
    assert len(checkpoints) > 0
```

---

## Test-Driven Development (TDD)

### Workflow

1. **RED**: Write failing test
2. **GREEN**: Write minimal code to pass
3. **IMPROVE**: Refactor and optimize
4. **VERIFY**: Ensure coverage increases

### Example

#### Step 1: Write Test (RED)

```python
# tests/test_filters.py
def test_sinc_interpolate():
    """Test Sinc interpolation upsamples correctly."""
    # Input: 1kHz sine wave at 16kHz
    t = np.linspace(0, 1, 16000)
    signal = np.sin(2 * np.pi * 1000 * t)

    # Upsample to 48kHz
    upsampled = sinc_interpolate(signal, source_sr=16000, target_sr=48000)

    # Check output length
    assert len(upsampled) == 48000

    # Check frequency content preserved
    fft = np.fft.rfft(upsampled)
    freqs = np.fft.rfftfreq(len(upsampled), 1/48000)
    peak_freq = freqs[np.argmax(np.abs(fft))]
    assert abs(peak_freq - 1000) < 10  # Within 10Hz
```

#### Step 2: Implement (GREEN)

```python
# src/neural_deringer/data/filters.py
def sinc_interpolate(
    signal: np.ndarray,
    source_sr: int,
    target_sr: int
) -> np.ndarray:
    """Upsample using Sinc interpolation.

    Args:
        signal: Input signal
        source_sr: Source sample rate
        target_sr: Target sample rate

    Returns:
        Upsampled signal
    """
    from scipy import signal as sp_signal
    return sp_signal.resample(signal, int(len(signal) * target_sr / source_sr))
```

#### Step 3: Refactor (IMPROVE)

```python
def sinc_interpolate(
    signal: np.ndarray,
    source_sr: int,
    target_sr: int
) -> np.ndarray:
    """Upsample using Sinc interpolation.

    Args:
        signal: Input signal
        source_sr: Source sample rate
        target_sr: Target sample rate

    Returns:
        Upsampled signal

    Raises:
        ValueError: If sample rates are invalid

    Physical Basis:
        Sinc interpolation is ideal in frequency domain (perfect
        reconstruction) but introduces Gibbs phenomenon (ringing)
        in time domain due to truncation of infinite sinc function.
    """
    # Validation
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive")

    if target_sr <= source_sr:
        raise ValueError("target_sr must be greater than source_sr")

    # Calculate upsampling ratio
    ratio = target_sr / source_sr
    num_samples = int(len(signal) * ratio)

    # Resample using Sinc interpolation
    from scipy import signal as sp_signal
    return sp_signal.resample(signal, num_samples)
```

#### Step 4: Verify Coverage

```bash
uv run pytest tests/test_filters.py::test_sinc_interpolate --cov=neural_deringer.data.filters --cov-report=term
```

---

## Test Organization

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_filters.py          # Unit tests for filters
├── test_models.py           # Unit tests for models
├── integration/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   └── test_training_pipeline.py
├── e2e/
│   ├── __init__.py
│   └── test_training.py
└── fixtures/                # Test data
    ├── sample.wav
    └── reference_output.npy
```

### Shared Fixtures

```python
# tests/conftest.py
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_signal() -> np.ndarray:
    """Generate test sine wave."""
    t = np.linspace(0, 1, 48000)
    return np.sin(2 * np.pi * 1000 * t)


@pytest.fixture
def temp_audio_file(tmp_path: Path) -> Path:
    """Create temporary audio file."""
    import soundfile as sf

    audio_path = tmp_path / "test.wav"
    signal = np.random.randn(48000)
    sf.write(audio_path, signal, 48000)

    return audio_path


@pytest.fixture
def mock_model():
    """Create mock neural network model."""
    import torch.nn as nn

    class MockModel(nn.Module):
        def forward(self, x):
            return x

    return MockModel()
```

---

## Test Markers

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.slow
def test_large_dataset_training():
    """Test training on large dataset (takes 10+ seconds)."""
    pass

@pytest.mark.gpu
def test_gpu_inference():
    """Test inference on GPU (requires CUDA)."""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Test complete data processing pipeline."""
    pass
```

Run specific test categories:

```bash
# Run only fast tests
uv run pytest -m "not slow and not gpu"

# Run only GPU tests
uv run pytest -m gpu

# Run integration tests
uv run pytest -m integration
```

---

## Parameterized Tests

Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize("sample_rate,duration,expected_length", [
    (16000, 1.0, 16000),
    (48000, 1.0, 48000),
    (44100, 0.5, 22050),
])
def test_signal_generation(sample_rate: int, duration: float, expected_length: int):
    """Test signal generation with various parameters."""
    signal = generate_signal(sample_rate, duration)
    assert len(signal) == expected_length
```

---

## Testing Best Practices

### 1. Test One Thing

```python
# ❌ BAD: Testing multiple things
def test_bessel_filter():
    filtered = bessel_filter(signal, 10000)
    assert filtered.shape == signal.shape
    assert np.max(filtered) <= 1.0
    assert np.mean(filtered) < 0.1

# ✅ GOOD: Separate tests
def test_bessel_filter_preserves_shape():
    filtered = bessel_filter(signal, 10000)
    assert filtered.shape == signal.shape

def test_bessel_filter_no_overshoot():
    filtered = bessel_filter(step_signal, 10000)
    assert np.max(filtered) <= 1.0

def test_bessel_filter_dc_removal():
    filtered = bessel_filter(dc_signal, 10000)
    assert abs(np.mean(filtered)) < 0.01
```

### 2. Use Descriptive Names

```python
# ❌ BAD
def test_filter():
    pass

# ✅ GOOD
def test_bessel_filter_removes_high_frequencies():
    pass
```

### 3. Arrange-Act-Assert Pattern

```python
def test_upsampler_doubles_length():
    # Arrange
    signal = np.random.randn(1000)
    upsampler = Upsampler(ratio=2)

    # Act
    upsampled = upsampler.process(signal)

    # Assert
    assert len(upsampled) == 2000
```

### 4. Test Edge Cases

```python
def test_filter_empty_signal():
    """Test filter handles empty input."""
    signal = np.array([])
    with pytest.raises(ValueError, match="Signal cannot be empty"):
        bessel_filter(signal, 10000)

def test_filter_single_sample():
    """Test filter handles single sample."""
    signal = np.array([1.0])
    filtered = bessel_filter(signal, 10000)
    assert filtered.shape == (1,)

def test_filter_nyquist_frequency():
    """Test filter at Nyquist frequency."""
    signal = np.random.randn(1000)
    filtered = bessel_filter(signal, cutoff_freq=24000, sample_rate=48000)
    # Should work without error
```

---

## Continuous Testing

### Pre-commit Hook

Fast tests run on every commit:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: pytest
      name: pytest
      entry: uv run pytest
      language: system
      args: [-m, "not slow and not gpu", --tb=short]
      pass_filenames: false
      always_run: true
```

### Pre-push Hook

Comprehensive tests run on push:

```yaml
- repo: local
  hooks:
    - id: pytest-full
      name: pytest-full
      stages: [pre-push]
      entry: uv run pytest
      language: system
      args: [--cov=neural_deringer, --cov-fail-under=80]
      pass_filenames: false
      always_run: true
```

---

## Test Coverage Goals

| Module | Target Coverage | Rationale |
|--------|----------------|-----------|
| `data/filters.py` | 95% | Core signal processing |
| `models/*.py` | 85% | Critical neural network logic |
| `training/trainer.py` | 80% | Training loop |
| `inference/*.py` | 90% | Production code |
| `scripts/*.py` | 50% | Utility scripts |

---

## Mocking

Use mocks for expensive operations:

```python
from unittest.mock import patch, MagicMock

def test_training_saves_checkpoint(tmp_path: Path):
    """Test that training saves checkpoints."""
    # Mock expensive model training
    with patch('neural_deringer.training.trainer.train_epoch') as mock_train:
        mock_train.return_value = 0.5  # Mock loss

        trainer = Trainer(model, tmp_path)
        trainer.train(num_epochs=1)

        # Verify checkpoint saved
        assert (tmp_path / "checkpoint.pth").exists()
```

---

## Performance Testing

Test computational performance:

```python
import time

@pytest.mark.slow
def test_filter_performance():
    """Test that filtering completes in reasonable time."""
    signal = np.random.randn(480000)  # 10 seconds at 48kHz

    start = time.time()
    filtered = bessel_filter(signal, 10000)
    elapsed = time.time() - start

    # Should complete in < 1 second
    assert elapsed < 1.0
```

---

## Testing Checklist

Before merging code, verify:

- [ ] All tests pass: `uv run pytest -v`
- [ ] Coverage ≥ 80%: `uv run pytest --cov=neural_deringer --cov-report=term`
- [ ] No skipped tests without reason
- [ ] Tests are deterministic (no random failures)
- [ ] Edge cases tested (empty input, boundary conditions)
- [ ] Error cases tested (invalid input, exceptions)
- [ ] Integration tests cover main workflows
- [ ] Slow tests marked with `@pytest.mark.slow`
- [ ] GPU tests marked with `@pytest.mark.gpu`
- [ ] Test names are descriptive

---

## Running Tests

```bash
# All tests
uv run pytest -v

# Fast tests only
uv run pytest -m "not slow and not gpu" -v

# Specific test file
uv run pytest tests/test_filters.py -v

# Specific test
uv run pytest tests/test_filters.py::test_bessel_filter -v

# With coverage
uv run pytest --cov=neural_deringer --cov-report=html

# Parallel execution (faster)
uv run pytest -n auto

# Stop on first failure
uv run pytest -x

# Verbose output
uv run pytest -vv --tb=long
```
