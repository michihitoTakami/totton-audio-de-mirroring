# CAPB transient robustness

- Checkpoint: `data/checkpoints/capb_48k/run17_transient_dwell_20260922_48k/capb_best.pt`
- Overall: **PASS**
- Direct 64-phase worst: -37.41 dB at offset 33
- OLA-boundary worst: -38.18 dB at offset 27

Negative margin is below the unchanged G2b threshold. Each offset must pass
both the canonical 0.5--4 ms window and the supplemental 4--12 ms tail window.
