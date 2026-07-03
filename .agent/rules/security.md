# Security Guidelines

## Critical Rules

### 1. No Hardcoded Secrets

**NEVER** hardcode sensitive information:

- API keys
- Passwords
- Tokens
- Private keys
- Database credentials
- Encryption keys

### Example: Bad Practice

```python
# ❌ BAD
API_KEY = "sk_live_abc123xyz789"
DB_PASSWORD = "MySecretPassword123"
```

### Example: Good Practice

```python
# ✅ GOOD
import os

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

DB_PASSWORD = os.environ.get("DB_PASSWORD")
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD environment variable not set")
```

---

## 2. Input Validation

**ALWAYS** validate and sanitize user input:

```python
from pathlib import Path
from typing import Union

def load_audio_file(file_path: Union[str, Path]) -> np.ndarray:
    """Load audio file with path validation.

    Args:
        file_path: Path to audio file

    Returns:
        Audio data as numpy array

    Raises:
        ValueError: If file path is invalid or file doesn't exist
        SecurityError: If path traversal attempt detected
    """
    path = Path(file_path).resolve()

    # Prevent path traversal attacks
    if ".." in str(file_path):
        raise SecurityError("Path traversal attempt detected")

    # Validate file exists
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    # Validate file extension
    allowed_extensions = {".wav", ".flac", ".mp3", ".ogg"}
    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Load file
    return load_audio(path)
```

---

## 3. SQL Injection Prevention

**ALWAYS** use parameterized queries:

```python
# ❌ BAD
query = f"SELECT * FROM users WHERE username = '{username}'"
cursor.execute(query)

# ✅ GOOD
query = "SELECT * FROM users WHERE username = ?"
cursor.execute(query, (username,))
```

---

## 4. XSS Prevention

**ALWAYS** sanitize HTML output:

```python
from html import escape

# ❌ BAD
html = f"<div>{user_input}</div>"

# ✅ GOOD
html = f"<div>{escape(user_input)}</div>"
```

---

## 5. Safe Error Messages

**NEVER** expose sensitive information in error messages:

```python
# ❌ BAD
except Exception as e:
    return f"Database connection failed: {db_password} at {db_host}"

# ✅ GOOD
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    return "An error occurred. Please contact support."
```

---

## 6. File Operations

**ALWAYS** validate file paths and permissions:

```python
def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Save model checkpoint with security checks.

    Args:
        model: PyTorch model
        path: Save path

    Raises:
        SecurityError: If path is outside allowed directory
    """
    # Ensure path is within allowed directory
    allowed_dir = Path("data/checkpoints").resolve()
    save_path = path.resolve()

    if not str(save_path).startswith(str(allowed_dir)):
        raise SecurityError(f"Path outside allowed directory: {save_path}")

    # Create directory with restricted permissions
    save_path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)

    # Save with restricted permissions
    torch.save(model.state_dict(), save_path)
    os.chmod(save_path, 0o644)
```

---

## 7. Dependency Security

**ALWAYS** pin dependency versions:

```toml
# pyproject.toml
dependencies = [
    "torch>=2.5.0",  # Specify minimum version
    "numpy>=1.24",   # Avoid unpinned versions
]
```

**REGULARLY** audit dependencies:

```bash
# Check for security vulnerabilities
uv pip audit

# Update dependencies
uv sync --upgrade
```

---

## 8. Environment Variables

**ALWAYS** use environment variables for configuration:

```python
# config.py
import os
from pathlib import Path

class Config:
    """Application configuration from environment variables."""

    # Data paths
    DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
    CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "data/checkpoints"))

    # Training config
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))

    # Secrets (must be set)
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    if not WANDB_API_KEY and os.environ.get("USE_WANDB") == "true":
        raise ValueError("WANDB_API_KEY must be set when USE_WANDB=true")
```

Create `.env.example` (commit to repo):

```bash
# .env.example
DATA_DIR=data
CHECKPOINT_DIR=data/checkpoints
BATCH_SIZE=32
LEARNING_RATE=0.001
WANDB_API_KEY=your_key_here
```

**NEVER** commit actual `.env` file (should be in `.gitignore`).

---

## 9. GPU Security

**ALWAYS** validate device access:

```python
def get_device() -> torch.device:
    """Get compute device with fallback.

    Returns:
        CUDA device if available, else CPU
    """
    if torch.cuda.is_available():
        # Validate GPU access
        try:
            torch.cuda.current_device()
            return torch.device("cuda")
        except Exception as e:
            logger.warning(f"CUDA device access failed: {e}")
            return torch.device("cpu")
    return torch.device("cpu")
```

---

## 10. Code Execution

**NEVER** use `eval()` or `exec()` on untrusted input:

```python
# ❌ BAD
config = eval(user_input)  # Arbitrary code execution!

# ✅ GOOD
import json
config = json.loads(user_input)  # Safe parsing
```

---

## Security Checklist

Before committing code, verify:

- [ ] No hardcoded secrets (API keys, passwords, tokens)
- [ ] All user input validated and sanitized
- [ ] Parameterized queries used (no string concatenation)
- [ ] Error messages don't leak sensitive information
- [ ] File paths validated (no path traversal)
- [ ] Dependencies pinned to specific versions
- [ ] Environment variables used for configuration
- [ ] No use of `eval()` or `exec()` on untrusted input
- [ ] Proper file permissions set
- [ ] Security-sensitive operations logged

---

## Reporting Security Issues

If you discover a security vulnerability:

1. **DO NOT** open a public issue
2. Email security contact (if available)
3. Provide detailed description and reproduction steps
4. Allow time for fix before public disclosure
