# CAPB retraining visualization and distortion investigation

## Checkpoints

- `44k1`: `data/checkpoints/capb/run9_retrain_20260829/capb_best.pt`
- `48k`: `data/checkpoints/capb_48k/run9_retrain_20260829/capb_best.pt`

## Coherent-line results

| Diagnostic | 44.1→88.2 kHz | 48→96 kHz |
|---|---:|---:|
| 1 kHz THD, harmonics through 20 kHz | -146.29 dB | -139.36 dB |
| SMPTE IMD, 60 Hz + 7 kHz | -120.38 dB | -69.70 dB |
| CCIF IMD, 19 + 20 kHz | -155.19 dB | -143.23 dB |
| Strongest added 10 kHz AM sideband | -169.56 dB | -164.72 dB |

The 48 kHz checkpoint is the clear outlier on the SMPTE two-tone probe. Its
sideband RSS is 50.68 dB higher than the 44.1 kHz checkpoint
and corresponds to an amplitude ratio of approximately 0.0327%.
The individual `7 kHz ± n·60 Hz` lines and the controller-weight trajectory
are shown together in each `sideband_degradation.png`. The symmetric product
family and larger weight excursion are consistent with modulation introduced
by the time-varying prototype blend. This is a diagnosis, not proof of a
single internal causal mechanism.

The 1 kHz audio-band THD, CCIF products, and higher-order AM sidebands remain
far below the SMPTE result. The negative finding is therefore specific: the
48 kHz controller reacts poorly to the mixed low/high two-tone condition; it
is not evidence of broad harmonic distortion on every steady signal.

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
