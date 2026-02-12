# Issue #109 8指標 勝敗表

## Inputs
- metrics_root: `/home/michihito/Working/totton-audio-de-mirroring/reports/microstructure_original88k_eval_20260210/metrics`
- audio_root: `/home/michihito/Working/totton-audio-de-mirroring/reports/microstructure_method_compare_20260210_full`
- target_root: `/home/michihito/Working/microstructure-metrics/test_signals_88k`
- methods: bessel_iir, bessel_fir, fir_10k, baseline_nn, distillation_nn

## Aggregate Winners

| Metric | Better | bessel_iir | bessel_fir | fir_10k | baseline_nn | distillation_nn | Winner | Coverage |
|---|---|---|---|---|---|---|---|---|
| MPS Corr | high | 0.999857 | 0.998746 | 1.000000 | n/a | 0.999857 | fir_10k | 4 |
| MPS Dist | low | 0.000051 | 0.000235 | 0.000001 | n/a | 0.000051 | fir_10k | 4 |
| TFS Corr | high | 0.977117 | 0.309833 | 0.955742 | n/a | 0.943105 | bessel_iir | 4 |
| Attack P95 (ms) | abs_low | 0.313647 | 4.805298 | 0.118532 | n/a | 0.318800 | fir_10k | 4 |
| Bass Cycle Corr | high | 0.817878 | 0.121652 | 0.818182 | n/a | 0.817878 | fir_10k | 4 |
| Lowband Wave Error (dB) | low | n/a | 3.551116 | -64.376737 | n/a | n/a | fir_10k | 2 |
| Lowband Phase Error (deg) | low | n/a | 94.659140 | 0.003064 | n/a | n/a | fir_10k | 2 |
| Lowband Group Delay Error (ms) | low | n/a | 56.662813 | 0.148175 | n/a | n/a | fir_10k | 2 |

## Points

| method | points |
|---|---:|
| fir_10k | 7.00 |
| bessel_iir | 1.00 |
| baseline_nn | 0.00 |
| bessel_fir | 0.00 |
| distillation_nn | 0.00 |

## Per-file Winner Counts

| Metric | bessel_iir | bessel_fir | fir_10k | baseline_nn | distillation_nn | tie | n/a |
|---|---|---|---|---|---|---|---|
| MPS Corr | 0 | 0 | 11 | 0 | 0 | 0 | 0 |
| MPS Dist | 0 | 0 | 11 | 0 | 0 | 0 | 0 |
| TFS Corr | 3 | 0 | 8 | 0 | 0 | 0 | 0 |
| Attack P95 (ms) | 0 | 0 | 9 | 0 | 0 | 2 | 0 |
| Bass Cycle Corr | 0 | 0 | 9 | 0 | 0 | 2 | 0 |
| Lowband Wave Error (dB) | 0 | 0 | 11 | 0 | 0 | 0 | 0 |
| Lowband Phase Error (deg) | 0 | 0 | 11 | 0 | 0 | 0 | 0 |
| Lowband Group Delay Error (ms) | 0 | 0 | 11 | 0 | 0 | 0 | 0 |
