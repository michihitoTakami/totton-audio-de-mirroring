# CAPB visualization and distortion investigation

## Checkpoints

- `44k1`: `data/checkpoints/capb/run16_v5b_midflat_g03_20260903_44k1/capb_best.pt`
- `48k`: `data/checkpoints/capb_48k/run16_v5b_midflat_g03_20260903_48k/capb_best.pt`

## Coherent-line results

| Diagnostic | 44.1→88.2 kHz | 48→96 kHz |
|---|---:|---:|
| 1 kHz THD, harmonics through 20 kHz | -141.95 dB | -133.05 dB |
| SMPTE IMD, 60 Hz + 7 kHz | -137.47 dB | -137.57 dB |
| CCIF IMD, 19 + 20 kHz | -152.33 dB | -141.00 dB |
| Strongest added 10 kHz AM sideband | -143.94 dB | -157.25 dB |

Both checkpoints are below the unchanged -110 dB
SMPTE modulation-sideband gate. The 48 kHz result is 0.10 dB
lower than the 44.1 kHz result and corresponds to an amplitude ratio of
approximately 0.000013%. The per-family plots confirm that the
symmetric `7 kHz ± n·60 Hz` family and controller-weight excursion are
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
