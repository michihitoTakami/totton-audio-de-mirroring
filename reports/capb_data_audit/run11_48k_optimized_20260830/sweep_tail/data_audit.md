# CAPB training-data audit

- Config: `configs/data_generation_capb_48k_sweep_tail_polish.yaml`
- Samples: 10000
- Exact-decimation maximum error: 0.000e+00
- Worst sampled image/main: -147.55 dB
- Focused clean/augmented: 340 / 124

## Family counts

| Family | Count |
|---|---:|
| imd_two_tone | 1029 |
| isolated_click | 279 |
| near_nyquist_noise | 1047 |
| pink_noise | 1015 |
| square_wave | 498 |
| sweep_linear | 2516 |
| sweep_log | 3431 |
| tone_burst | 185 |

## Sparse transient containment

| Family | Requested chunks | Event-bearing chunks |
|---|---:|---:|
| isolated_click | 279 | 279 |
| tone_burst | 185 | 185 |
