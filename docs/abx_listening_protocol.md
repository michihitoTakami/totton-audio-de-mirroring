# ABX Listening Protocol (Issue #59 / #64)

## Purpose

Evaluate whether Stage 1 NMSE output reduces audible mirror-related harshness
without degrading transient perception, using fixed triplets:

- `input`: Stage 1 input reference
- `naive`: Naive reference path
- `nmse`: Selected NMSE checkpoint output

Issue #64 freezes these triplets in:

- `tests/fixtures/golden_samples/abx_pairs.json`
- `tests/fixtures/golden_samples/issue64_model_selection.json`

## Frozen Model Reference

- Selected checkpoint name: `stage1_best.pt`
- Checkpoint SHA-256:
  `ef9f2815e57663140eff3ff6860847148eb527c733e17e2890cba9a24ec4e025`

## Test Material

Use the exact pair map from `tests/fixtures/golden_samples/abx_pairs.json`.
Current pair set contains:

- `sample_a`
- `sample_b`

Each pair must include the three files (`input` / `naive` / `nmse`) from the
paths declared in `abx_pairs.json`.

## Procedure

1. Loudness-match `naive` and `nmse` within ±0.1 dB before listening.
2. Present `A` and `B` in random order (`naive` or `nmse`).
3. Present `X` as either `A` or `B` randomly.
4. Listener answers whether `X == A` or `X == B`.
5. Repeat at least 10 trials per sample.
6. Record qualitative notes:
   - digital harshness / metallic texture
   - transient attack clarity
   - fatigue over repeated playback

## Acceptance Guidance

- Quantitative:
  - ABX hit rate should exceed chance level with statistical significance.
- Qualitative:
  - NMSE should reduce harshness/graininess versus naive.
  - No obvious transient blurring should be reported.

## Update Rule

When checkpoint selection changes, update all of:

1. `tests/fixtures/golden_samples/abx_pairs.json`
2. `tests/fixtures/golden_samples/issue64_model_selection.json`
3. `tests/fixtures/golden_samples/regression_baseline.json`
4. This document's "Frozen Model Reference" section
