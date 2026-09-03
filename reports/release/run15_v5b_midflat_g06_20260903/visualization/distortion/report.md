# CAPB visualization and distortion investigation

## Checkpoints

- `44k1`: `/tmp/capb_v5b_ft_44_g0.6_s1234/capb_best.pt`
- `48k`: `/tmp/capb_v5b_ft_48_g0.6_s1234/capb_best.pt`

## Coherent-line results

| Diagnostic | 44.1→88.2 kHz | 48→96 kHz |
|---|---:|---:|
| 1 kHz THD, harmonics through 20 kHz | -142.05 dB | -133.08 dB |
| SMPTE IMD, 60 Hz + 7 kHz | -137.90 dB | -137.77 dB |
| CCIF IMD, 19 + 20 kHz | -152.18 dB | -140.85 dB |
| Strongest added 10 kHz AM sideband | -132.63 dB | -157.00 dB |

The 48 kHz checkpoint has the larger SMPTE result. Its
sideband RSS is 0.13 dB higher than the 44.1 kHz checkpoint and
corresponds to an amplitude ratio of approximately 0.0000%.
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
