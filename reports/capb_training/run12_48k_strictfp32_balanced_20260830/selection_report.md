# 48 kHz strict-FP32 candidate selection

## Decision

Adopt `data/checkpoints/capb_48k/run12_48k_strictfp32_balanced_20260830/capb_best.pt`.

The apparent 48 kHz regression in the run11 plots was reproduced by enabling
CUDA TF32 and disappeared when the same checkpoints and FIR banks were run in
strict FP32. The release path therefore disables TF32 and records the effective
precision in every generated report.

## Training data

The selected candidate applies one low-rate margin epoch to the accepted run11
checkpoint. Its balanced 10,000-example dataset contains stationary distortion
probes, sweeps and noise as well as contained click/tone-burst events. The audit
found exact 2:1 decimation (`0.0` maximum error) and a worst sampled
image-to-main ratio of `-123.71 dB`.

## Candidate screen

All 20 epoch checkpoints from the two run10 relearn trials passed the unchanged
canonical and held-out G1–G9 gates. They were nevertheless rejected by the
additional cross-rate distortion and normalized impulse-response checks.

| Candidate | THD (dB) | SMPTE (dB) | CCIF (dB) | AM sideband (dB) | Pre-echo MSE | Decision |
|---|---:|---:|---:|---:|---:|---|
| run11 baseline | -139.88 | -144.40 | -142.92 | -155.21 | 1.52e-8 | baseline |
| selected balanced margin | -140.00 | -144.67 | -142.73 | -157.78 | 1.33e-8 | PASS |
| run10 relearn, head scale 0.75 | -139.76 | -110.11 | -126.53 | -99.57 | 2.43e-7 | reject |
| run10 relearn, head scale 0.85 | -139.55 | -125.45 | -142.83 | -163.67 | 2.47e-7 | reject |

The selected checkpoint improves pre-echo, THD, SMPTE IMD and AM sidebands
against run11. Its 0.19 dB CCIF change remains within the measured rate-local
strict-FP32 fixed-FIR floor. The relearn trials moved almost completely toward
the sharp prototype; their G2b values remained under the frozen absolute gate,
but their normalized impulse shape was materially worse than the 44.1 kHz
release and therefore failed the release-quality gate.

## Evidence

- Data audit: `reports/capb_data_audit/run12_48k_strictfp32_balanced_20260830/`
- CPU/CUDA frozen gates: `reports/probe_gates/run12_strictfp32_release_20260830/`
- Impulse, THD, IMD and sideband plots: `reports/capb_visualization/run12_strictfp32_release_20260830/`
- TF32 root-cause comparison: `reports/capb_precision/run12_strictfp32_release_20260830/`
- Cross-rate release decision: `reports/capb_release_quality/run12_strictfp32_release_20260830/`
- Offset robustness: `reports/capb_robustness/run12_48k_strictfp32_balanced_20260830/`
