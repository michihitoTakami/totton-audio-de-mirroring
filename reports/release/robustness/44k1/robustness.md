# CAPB transient robustness

- Checkpoint: `data/checkpoints/capb/run11_44k1_optimized_20260829/capb_best.pt`
- Overall: **PASS**
- Direct 64-phase worst: -2.84 dB at offset 4
- OLA-boundary worst: -2.91 dB at offset 29

Negative margin is below the unchanged G2b threshold. Each offset must pass
both the canonical 0.5--4 ms window and the supplemental 4--12 ms tail window.
