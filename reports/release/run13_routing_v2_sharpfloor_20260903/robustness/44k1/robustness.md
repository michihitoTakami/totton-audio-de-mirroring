# CAPB transient robustness

- Checkpoint: `/tmp/capb_routing_v2_floor_44_s1234/capb_best.pt`
- Overall: **PASS**
- Direct 64-phase worst: -35.93 dB at offset 25
- OLA-boundary worst: -35.87 dB at offset -14

Negative margin is below the unchanged G2b threshold. Each offset must pass
both the canonical 0.5--4 ms window and the supplemental 4--12 ms tail window.
