# CAPB transient robustness

- Checkpoint: `data/checkpoints/capb/run16_v5b_midflat_g03_20260903_44k1/capb_best.pt`
- Overall: **PASS**
- Direct 64-phase worst: -37.25 dB at offset 12
- OLA-boundary worst: -37.23 dB at offset -27

Negative margin is below the unchanged G2b threshold. Each offset must pass
both the canonical 0.5--4 ms window and the supplemental 4--12 ms tail window.
