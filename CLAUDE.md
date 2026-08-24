# totton-audio-de-mirroring Development Guide

Think in English and answer in Japanese. Use English for code comments, docstrings, and commit messages.

The active Stage 1 architecture is CAPB (Constrained Adaptive Prototype-Blend). Read `README.md` and `AGENTS.md` before changing the audio path.

## Architecture contract

- 44.1→88.2 kHz and 48→96 kHz use rate-specific fixed FIR prototype banks.
- Every prototype is symmetric, gain-matched, and centered at one common delay.
- The controller emits convex, slowly varying blend weights only.
- `sharp` prioritizes image rejection, `gentle` prioritizes low ringing, and `mid` is the intermediate endpoint.
- CAPB does not use a hard 20 kHz split because that would reintroduce Gibbs ringing.
- Low-band transparency and no-added-HF are enforced by worst-case probe gates.
- Release requires all canonical and held-out gates to pass for both rate families.

## Primary commands

```bash
uv run python scripts/train_capb.py \
  --data-config configs/data_generation_capb.yaml \
  --config configs/training_stage1_capb.yaml

uv run python scripts/evaluate_probe_gates.py \
  --backend capb --checkpoint <checkpoint> --rate-family 44k1

uv run pytest -m "not slow and not gpu" -v
uv run ruff format --check src tests scripts
uv run ruff check src tests scripts
uv run mypy src
```

## Implementation rules

- Type hints are required for all functions.
- DSP/model docstrings use Google style and include `Physical Basis`.
- Validate inputs at entry and do not mutate caller-owned values.
- Preserve fixed FIR kernels and the convex-blend constraint.
- Do not claim improvement without the versioned probe report and its worst binding probe.
- Never weaken a gate to accommodate a checkpoint.
- Keep generated data, reports, checkpoints, and local validation files untracked unless explicitly requested.

See `AGENTS.md` for testing, security, and Git workflow rules.
