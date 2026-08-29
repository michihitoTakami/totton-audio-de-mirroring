# CAPB visualization and distortion investigation

## Checkpoints

- `44k1`: `data/checkpoints/capb/run9_retrain_20260829/capb_best.pt`
- `48k`: `data/checkpoints/capb_48k/run10_stationary_20260829/capb_best.pt`

## Coherent-line results

| Diagnostic | 44.1→88.2 kHz | 48→96 kHz |
|---|---:|---:|
| 1 kHz THD, harmonics through 20 kHz | -146.29 dB | -139.95 dB |
| SMPTE IMD, 60 Hz + 7 kHz | -120.38 dB | -141.43 dB |
| CCIF IMD, 19 + 20 kHz | -155.19 dB | -142.83 dB |
| Strongest added 10 kHz AM sideband | -169.56 dB | -161.76 dB |

The 48 kHz checkpoint is no longer the SMPTE
two-tone outlier: its sideband RSS is 21.05 dB lower than the
44.1 kHz diagnostic checkpoint and corresponds to an amplitude ratio of
approximately 0.000008%. The per-family plots confirm that the
symmetric `7 kHz ± n·60 Hz` family and controller-weight excursion were
suppressed together.

The 1 kHz audio-band THD, CCIF products, and higher-order AM sidebands are
reported separately so a two-tone modulation defect is not mislabeled as
broad harmonic distortion on every steady signal.

## Method

- All distortion measurements use the steady center one second of a
  three-second signal and integer-Hz coherent projections.
- THD includes harmonics only through 20 kHz so interpolation images are not
  mislabeled as nonlinear harmonics.
- SMPTE IMD is the RSS of the first five `7 kHz ± n·60 Hz` sideband pairs,
  relative to 7 kHz.
- CCIF IMD is the RSS of 1, 18, and 21 kHz products relative to the two
  primaries.
- Sinusoidal AM contains only orders 0 and ±1; orders ±2 through ±6 are
  treated as added sidebands.
- Ideal and Bessel paths are linear references. The versioned probe-gate
  reports remain authoritative for checkpoint acceptance; these plots are
  supplementary diagnostics and define no new gate.
