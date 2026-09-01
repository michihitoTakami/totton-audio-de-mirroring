# CAPB transient robustness

- Checkpoint: `data/checkpoints/capb_48k/run12_48k_strictfp32_balanced_20260830/capb_best.pt`
- Overall: **PASS**
- Direct 64-phase worst: -8.87 dB at offset 14
- OLA-boundary worst: -9.44 dB at offset 30

Negative margin is below the unchanged G2b threshold. Each offset must pass
both the canonical 0.5--4 ms window and the supplemental 4--12 ms tail window.
