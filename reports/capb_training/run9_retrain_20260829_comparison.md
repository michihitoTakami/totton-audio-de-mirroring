# CAPB July-baseline retraining comparison

## Scope

- Code: `2508996` (`refactor/capb-july4-baseline`)
- Training: 10,000 samples, 50 epochs, seed 1234, CUDA
- Evaluation: strict spec v3, canonical and held-out probes
- 44.1 kHz checkpoint: `data/checkpoints/capb/run9_retrain_20260829/capb_best.pt`
- 48 kHz checkpoint: `data/checkpoints/capb_48k/run9_retrain_20260829/capb_best.pt`

## Training reproducibility

| Candidate | Best validation total | Best epoch | Final mean weights (sharp/mid/gentle) |
|---|---:|---:|---|
| Original run9 record | 0.0102721912 | 34 | 0.8039 / 0.0899 / 0.1062 |
| 44.1 kHz retrain | 0.0102694446 | 48 | 0.8116 / 0.0832 / 0.1051 |
| 48 kHz retrain | 0.0145777017 | 49 | 0.8784 / 0.0128 / 0.1087 |

The 44.1 kHz retrain differs from the original run9 best validation total by
approximately -0.027%, which is sufficiently close to regard the July
training behavior as reproduced.

## Strict probe-gate results

| Rate family | Overall | Failed gates | Worst binding probe |
|---|---|---|---|
| 44.1 -> 88.2 kHz | FAIL | G2 high-frequency ringing; G2b pre-echo | `square_5000hz`; `impulse` |
| 48 -> 96 kHz | FAIL | G1 low-frequency ringing | `square_2000hz` |

### 44.1 kHz binding failures

- 5 kHz plateau RMS: 0.500682 (limit 0.499381)
- 5 kHz plateau P2P: 1.385499 (limit 1.294469)
- 5 kHz overshoot: 0.659238 (limit 0.548511)
- Impulse pre-echo energy: 9.32808e-7 (limit 2.5e-7)

### 48 kHz binding failure

- 2 kHz plateau RMS: 0.633887 (limit 0.620397)

## Decision

Neither rate family passes every strict canonical and held-out probe. The
44.1 kHz retrain confirms that the run9-era optimization is reproducible, but
the reproduced checkpoint is not release-acceptable. The 48 kHz retrain also
fails its worst-probe gate. These results support retaining the July CAPB code
as a historical baseline while declining to adopt either newly trained
checkpoint as a release pair.
