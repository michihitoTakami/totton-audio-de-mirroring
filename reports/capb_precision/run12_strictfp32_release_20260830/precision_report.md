# CAPB CUDA precision investigation

| Family | Metric | strict FP32 | TF32 | TF32 degradation |
|---|---|---:|---:|---:|
| 44k1 | thd_1khz_20khz_db | -145.67 dB | -85.10 dB | 60.57 dB |
| 44k1 | smpte_imd_db | -125.60 dB | -87.09 dB | 38.52 dB |
| 44k1 | ccif_imd_db | -154.44 dB | -97.18 dB | 57.26 dB |
| 44k1 | added_am_sideband_db | -131.16 dB | -113.37 dB | 17.80 dB |
| 48k | thd_1khz_20khz_db | -140.00 dB | -76.70 dB | 63.30 dB |
| 48k | smpte_imd_db | -144.67 dB | -85.91 dB | 58.77 dB |
| 48k | ccif_imd_db | -142.73 dB | -85.86 dB | 56.87 dB |
| 48k | added_am_sideband_db | -157.78 dB | -111.57 dB | 46.21 dB |

The checkpoints and prototype banks are unchanged between rows. The measured difference is caused by CUDA convolution precision, not by a different controller checkpoint.
