# CAPB long-FIR FineTuning selection

Selected profile: **release_v4**

Both FineTuned long-FIR profiles improved image rejection, but neither produced two passing 44.1 kHz seeds. The incumbent release is retained because pre-echo is a hard, worst-probe acceptance condition.

| Profile | Family | Passing seeds | G3 peak (dB) | G2b | G9 (dB) | Robustness |
|---|---|---:|---:|---:|---:|---|
| long_sharp_1535_a120 | 44k1 | 0/3 | -119.25 | 4.959e-07 | -125.41 | FAIL |
| long_sharp_1535_a120 | 48k | 3/3 | -128.04 | 7.913e-08 | -139.54 | PASS |
| long_sharp_2047_a120 | 44k1 | 0/3 | -119.12 | 5.112e-07 | -126.29 | FAIL |
| long_sharp_2047_a120 | 48k | 3/3 | -128.57 | 8.492e-08 | -136.45 | FAIL |

## Distortion and sideband comparison

| Profile | Family | THD (dB) | SMPTE (dB) | CCIF (dB) | AM sideband (dB) |
|---|---|---:|---:|---:|---:|
| release | 44k1 | -145.67 | -125.60 | -154.44 | -131.16 |
| release | 48k | -140.00 | -144.67 | -142.73 | -157.78 |
| long_sharp_1535_a120 | 44k1 | -140.56 | -125.58 | -151.56 | -130.94 |
| long_sharp_1535_a120 | 48k | -131.63 | -139.26 | -145.15 | -154.24 |
| long_sharp_2047_a120 | 44k1 | -138.41 | -126.34 | -145.47 | -130.91 |
| long_sharp_2047_a120 | 48k | -129.86 | -136.77 | -135.08 | -153.79 |
