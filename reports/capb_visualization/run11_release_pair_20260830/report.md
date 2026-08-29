# CAPB visualization and distortion investigation

## Checkpoints

- `44k1`: `data/checkpoints/capb/run11_44k1_optimized_20260829/capb_best.pt`
- `48k`: `data/checkpoints/capb_48k/run11_48k_optimized_20260830/capb_best.pt`

## Coherent-line results

| Diagnostic | 44.1→88.2 kHz | 48→96 kHz |
|---|---:|---:|
| 1 kHz THD, harmonics through 20 kHz | -85.10 dB | -76.70 dB |
| SMPTE IMD, 60 Hz + 7 kHz | -87.09 dB | -85.91 dB |
| CCIF IMD, 19 + 20 kHz | -97.18 dB | -85.87 dB |
| Strongest added 10 kHz AM sideband | -113.37 dB | -111.60 dB |

The 48 kHz checkpoint has the larger SMPTE result. Its
sideband RSS is 1.18 dB higher than the 44.1 kHz checkpoint and
corresponds to an amplitude ratio of approximately 0.0051%.
The symmetric `7 kHz ± n·60 Hz` family and controller-weight excursion are
consistent with modulation introduced by the time-varying prototype blend.
This is a diagnosis, not proof of a single internal causal mechanism.

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
