# Golden Samples for Regression Tests

This directory contains deterministic regression fixtures for Issue #30.

## Contents

- `stage1/input/*.npy`: Stage 1 baseline inputs (88.2kHz domain)
- `stage1/output/*.npy`: Expected Stage 1 outputs for regression checks
- `imd/naive/*.npy`: Naive signals for IMD proxy baseline
- `imd/nmse/*.npy`: NMSE-like signals for IMD proxy baseline
- `regression_baseline.json`: Frozen expected metrics payload

## Update Procedure

When model behavior is intentionally changed and new baselines are approved,
regenerate this fixture set with a deterministic script and update
`regression_baseline.json` in the same commit.

Required validation after update:

1. `uv run --extra dev pytest tests/regression/test_stage1_regression.py -v`
2. `uv run --extra dev pytest -m "not slow and not gpu" -v`
