# CAPB transient robustness

- Checkpoint: `data/checkpoints/capb_48k/run16_v5b_midflat_g03_20260903_48k/capb_best.pt`
- Overall: **PASS**
- Direct 64-phase worst: -38.39 dB at offset 11
- OLA-boundary worst: -38.42 dB at offset -29

Negative margin is below the unchanged G2b threshold. Each offset must pass
both the canonical 0.5--4 ms window and the supplemental 4--12 ms tail window.
