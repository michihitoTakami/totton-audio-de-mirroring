# CAPB transient robustness

- Checkpoint: `/tmp/capb_longfir_ft_48_long_sharp_1023_a140_s1234/capb_best.pt`
- Overall: **PASS**
- Direct 64-phase worst: -42.36 dB at offset 19
- OLA-boundary worst: -42.30 dB at offset -29

Negative margin is below the unchanged G2b threshold. Each offset must pass
both the canonical 0.5--4 ms window and the supplemental 4--12 ms tail window.
