# AI Agents Integration Guide

## Communication

Think in English and respond in Japanese. Code comments, docstrings, and commit messages are written in English.

## Read first

1. `README.md` — CAPB architecture, commands, and acceptance policy
2. `.agent/rules/testing.md`
3. `.agent/rules/coding-style.md`
4. `.agent/rules/security.md`

## Project

This repository implements CAPB (Constrained Adaptive Prototype-Blend), a time-response-first audio upsampler.

- Stage 1: 44.1→88.2 kHz or 48→96 kHz CAPB
- Stage 2: 88.2→705.6 kHz DSP cascade
- Target: Jetson Orin Nano 8GB
- Non-realtime processing is acceptable

CAPB blends fixed, symmetric, gain-matched FIR prototypes with one shared group delay. The neural controller selects convex weights only; it does not synthesize a waveform or modify the prototype kernels.

## Hard requirements

1. Do not infer or generate information above the input Nyquist frequency.
2. Do not introduce ringing regression against the Bessel reference.
3. Enforce low-band gain, waveform, phase, and group-delay gates.
4. Enforce image-band and no-added-HF gates.
5. Accept a checkpoint only when every canonical and held-out probe passes for both supported rate families.
6. Gate on the worst probe. Mean metrics are diagnostic only.

CAPB intentionally has no hard 20 kHz band split: such a split restores Gibbs ringing. Do not reintroduce an audible-band projection without new physical evidence and complete probe-gate validation.

## Coding rules

- Add type hints to every function.
- Use Google-style docstrings with a `Physical Basis` section for DSP/model behavior.
- Validate inputs at function entry.
- Never mutate input arrays or tensors.
- Wrap I/O failures with actionable exceptions.
- Keep files below 800 lines and functions below 50 lines where practical.
- Keep nesting at three levels or less; prefer early returns.
- Never use `eval()` or `exec()` on untrusted input.
- Never hardcode credentials.

## Tests

- Add tests for every new module and behavior.
- Target 80%+ coverage and 90%+ for DSP critical paths.
- Do not update gates or frozen probe manifests merely to make a candidate pass.

```bash
uv run pytest -m "not slow and not gpu" -v
uv run ruff format --check src tests scripts
uv run ruff check src tests scripts
uv run mypy src
uv run pytest -v
```

## CAPB workflows

Train 44.1 kHz family:

```bash
uv run python scripts/train_capb.py \
  --data-config configs/data_generation_capb.yaml \
  --config configs/training_stage1_capb.yaml
```

Train 48 kHz family:

```bash
uv run python scripts/train_capb.py \
  --data-config configs/data_generation_capb_48k.yaml \
  --config configs/training_stage1_capb_48k.yaml
```

Evaluate:

```bash
uv run python scripts/evaluate_probe_gates.py \
  --backend capb \
  --checkpoint <checkpoint> \
  --rate-family <44k1|48k>
```

Run the production pipeline after configuring `stage1.mode: capb`:

```bash
uv run totton-upsample input.wav -o output.wav \
  -c configs/stage1_stage2_pipeline.yaml
```

## Git workflow

Create worktrees from `origin/main`, never from a potentially stale local `main`:

```bash
git fetch origin
git worktree add -b feat/<name> <path> origin/main
```

Use Conventional Commits. Never bypass pre-commit or pre-push hooks with `--no-verify`.

Do not add generated checkpoints, reports, local audio, or ad-hoc validation files unless the user explicitly requests them.
