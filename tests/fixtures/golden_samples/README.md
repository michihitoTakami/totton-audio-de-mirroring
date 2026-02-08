# Golden Samples for Regression Tests

This directory contains deterministic regression fixtures for Issue #30.
Issue #64 froze this fixture set to the retrained NMSE checkpoint selection.

## Contents

- `stage1/input/*.npy`: Stage 1 baseline inputs (88.2kHz domain)
- `stage1/output/*.npy`: Expected Stage 1 outputs for regression checks
- `imd/naive/*.npy`: Naive signals for IMD proxy baseline
- `imd/nmse/*.npy`: NMSE-like signals for IMD proxy baseline
- `regression_baseline.json`: Frozen expected metrics payload
- `issue64_model_selection.json`: Frozen candidate ranking and selected model hash
- `abx_pairs.json`: Frozen ABX triplets (`input` / `naive` / `nmse`)

## Update Procedure

When model behavior is intentionally changed and new baselines are approved,
regenerate this fixture set and update `regression_baseline.json` in the same commit.

Issue #64 reference command sequence:

```bash
uv run python scripts/run_issue63_stage1_workflow.py \
  --data-config configs/data_generation.yaml \
  --train-config configs/training_stage1.yaml \
  --eval-input-dir tests/fixtures/golden_samples/stage1/input \
  --imd-naive-dir tests/fixtures/golden_samples/imd/naive \
  --checkpoint-dir data/checkpoints \
  --report-dir reports/issue64 \
  --seed 1234 \
  --device cuda \
  --skip-training \
  --candidate-checkpoints stage1_best.pt stage1_last.pt stage1_emergency.pt
```

After selection:

1. Copy `reports/issue64/candidate_outputs/<selected>/` to
   `tests/fixtures/golden_samples/stage1/output/`
2. Copy the same outputs to `tests/fixtures/golden_samples/imd/nmse/`
3. Regenerate `tests/fixtures/golden_samples/regression_baseline.json`
4. Update `issue64_model_selection.json` and `abx_pairs.json`

Required validation after update:

1. `uv run --extra dev pytest tests/regression/test_stage1_regression.py -v`
2. `uv run --extra dev pytest -m "not slow and not gpu" -v`
