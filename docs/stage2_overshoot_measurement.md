# Stage2 Overshoot Measurement (Issue #49)

## Purpose

Define reproducible overshoot metrics for Stage2 (88.2kHz -> 705.6kHz) FIR cascade and provide current baseline values.

## Probe Signals

- Step: `x[n]=1` with input length `4096` samples at 88.2kHz.
- Square: 1kHz square wave, duration `0.2s`, sampled at 88.2kHz.

## Processing Chain

- 2x zero-stuffing + FIR filtering per stage.
- 3-stage cascade (`2x x 2x x 2x`) with taps from `cpp/configs/stage{1,2,3}_taps.txt`.

## Metric Definition

For response `y`, settled region starts at `floor(0.75 * len(y))`.

- Reference level: `quantile(settled, 0.95)`
  - Square-wave metric uses positive settled samples only.
- Peak level: `max(y)`
- Overshoot ratio: `(peak - reference) / abs(reference)`

## Current Baseline

Measured by:

```bash
uv run python scripts/evaluate_stage2_overshoot.py --json
```

Result:

- Output sample rate: `705600` Hz
- Step overshoot ratio: `0.3299109960180487`
- Square overshoot ratio: `0.38790760060673335`

## Notes

- This metric definition is intended as the Issue #49 baseline.
- Candidate redesigns should be compared with the same script/parameters.
