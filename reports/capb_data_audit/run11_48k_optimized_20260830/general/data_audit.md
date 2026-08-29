# CAPB training-data audit

- Config: `configs/data_generation_capb_48k.yaml`
- Samples: 10000
- Exact-decimation maximum error: 0.000e+00
- Worst sampled image/main: -131.93 dB
- Focused clean/augmented: 1044 / 405

## Family counts

| Family | Count |
|---|---:|
| am_tone | 505 |
| band_limited_noise | 403 |
| fm_tone | 516 |
| imd_two_tone | 525 |
| isolated_click | 500 |
| multitone | 1021 |
| music_like_mixture | 1053 |
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
| isolated_click | 500 | 500 |
| tone_burst | 949 | 949 |
