# CAPB transient robustness

- Checkpoint: `/tmp/capb_v5b_ft_48_g0.3_s1234/capb_best.pt`
- Overall: **PASS**
- Direct 64-phase worst: -38.39 dB at offset 11
- OLA-boundary worst: -38.42 dB at offset -29

Negative margin is below the unchanged G2b threshold. Each offset must pass
both the canonical 0.5--4 ms window and the supplemental 4--12 ms tail window.
