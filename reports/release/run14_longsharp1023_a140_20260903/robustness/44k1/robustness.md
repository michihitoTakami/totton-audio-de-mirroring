# CAPB transient robustness

- Checkpoint: `/tmp/capb_longfir_ft_44_long_sharp_1023_a140_s1234/capb_best.pt`
- Overall: **PASS**
- Direct 64-phase worst: -36.12 dB at offset 27
- OLA-boundary worst: -36.06 dB at offset -12

Negative margin is below the unchanged G2b threshold. Each offset must pass
both the canonical 0.5--4 ms window and the supplemental 4--12 ms tail window.
