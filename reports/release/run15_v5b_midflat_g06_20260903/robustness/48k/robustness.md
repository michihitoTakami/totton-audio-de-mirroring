# CAPB transient robustness

- Checkpoint: `/tmp/capb_v5b_ft_48_g0.6_s1234/capb_best.pt`
- Overall: **PASS**
- Direct 64-phase worst: -42.58 dB at offset 23
- OLA-boundary worst: -42.65 dB at offset -25

Negative margin is below the unchanged G2b threshold. Each offset must pass
both the canonical 0.5--4 ms window and the supplemental 4--12 ms tail window.
