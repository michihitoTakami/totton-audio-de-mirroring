# CAPB transient robustness

- Checkpoint: `/tmp/capb_routing_v2_floor_48_s1234/capb_best.pt`
- Overall: **PASS**
- Direct 64-phase worst: -41.98 dB at offset 2
- OLA-boundary worst: -41.96 dB at offset 17

Negative margin is below the unchanged G2b threshold. Each offset must pass
both the canonical 0.5--4 ms window and the supplemental 4--12 ms tail window.
