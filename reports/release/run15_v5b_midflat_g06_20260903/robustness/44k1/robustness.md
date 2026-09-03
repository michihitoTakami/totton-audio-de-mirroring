# CAPB transient robustness

- Checkpoint: `/tmp/capb_v5b_ft_44_g0.6_s1234/capb_best.pt`
- Overall: **PASS**
- Direct 64-phase worst: -32.80 dB at offset 32
- OLA-boundary worst: -33.26 dB at offset -7

Negative margin is below the unchanged G2b threshold. Each offset must pass
both the canonical 0.5--4 ms window and the supplemental 4--12 ms tail window.
