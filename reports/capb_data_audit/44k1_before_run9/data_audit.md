# CAPB training-data audit

- Config: `configs/data_generation_capb_run9_legacy.yaml`
- Samples: 10000
- Exact-decimation maximum error: 0.000e+00
- Worst sampled image/main: -129.55 dB
- Focused clean/augmented: 0 / 0

## Family counts

| Family | Count |
|---|---:|
| am_tone | 505 |
| band_limited_noise | 403 |
| fm_tone | 516 |
| isolated_click | 525 |
| multitone | 1013 |
| music_like_mixture | 1561 |
| near_nyquist_noise | 388 |
| percussive | 507 |
| pink_noise | 409 |
| sawtooth_wave | 452 |
| square_wave | 1016 |
| step_plateau | 978 |
| sweep_linear | 304 |
| sweep_log | 474 |
| tone_burst | 949 |

## Sparse transient containment

| Family | Requested chunks | Event-bearing chunks |
|---|---:|---:|
| isolated_click | 525 | 154 |
| tone_burst | 949 | 276 |
