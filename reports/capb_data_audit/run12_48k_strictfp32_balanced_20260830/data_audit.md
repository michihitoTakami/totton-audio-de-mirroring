# CAPB training-data audit

- Config: `configs/data_generation_capb_48k_balanced.yaml`
- Samples: 10000
- Exact-decimation maximum error: 0.000e+00
- Worst sampled image/main: -123.71 dB
- Focused clean/augmented: 995 / 386

## Family counts

| Family | Count |
|---|---:|
| am_tone | 604 |
| band_limited_noise | 304 |
| fm_tone | 400 |
| imd_two_tone | 832 |
| isolated_click | 628 |
| multitone | 802 |
| music_like_mixture | 620 |
| near_nyquist_noise | 333 |
| percussive | 478 |
| pink_noise | 307 |
| sawtooth_wave | 405 |
| square_wave | 951 |
| step_plateau | 1007 |
| sweep_linear | 602 |
| sweep_log | 974 |
| tone_burst | 753 |

## Sparse transient containment

| Family | Requested chunks | Event-bearing chunks |
|---|---:|---:|
| isolated_click | 628 | 628 |
| tone_burst | 753 | 753 |
