# 48 kHz transient optimization result

## Decision

`run11_48k_optimized_20260830` is selected. The frozen v4 canonical and
held-out suite passes every G1–G9 gate for both release rate families. The
48 kHz controller-phase and production Hann-OLA boundary sweeps also pass.

## 48 kHz before / after

| Metric | run10 stationary | run11 optimized | Gate |
|---|---:|---:|---:|
| G2b impulse pre-echo energy | 2.473e-7 | 1.523e-8 | <= 2.5e-7 |
| Impulse-train gain error | 0.0215 dB | 0.3902 dB | <= 0.5 dB |
| Canonical sweep peak image | -112.42 dB | -112.42 dB | <= -65 dB |
| SMPTE modulation sideband | -141.43 dB | -144.40 dB | <= -110 dB |

The final checkpoint trades part of the former gain margin for a 16.2x
reduction in the binding impulse pre-echo energy. Its impulse local energy is
0.3044 versus 0.2653 for Bessel and 0.4754 for sharp, placing the selected
response substantially closer to the low-ringing reference than to sharp.

## Cause and correction

The original 48 kHz training labels treated discontinuous periodic signals,
focused impulses, and continuous stationary probes too similarly. Hard
gentle routing plus long quiet masks saturated impulse logits, while sweep
chunks under-sampled the 20 kHz terminal boundary. The correction:

1. keeps square/sawtooth in the edge-rich path rather than stationary routing;
2. gives continuous sweeps fixed-sharp supervision and samples sweep starts
   logarithmically;
3. excludes focused transients from one-hot routing and broad quiet masks;
4. learns high-frequency sweep boundaries with random-position chunks; and
5. applies a short clean impulse full-fidelity margin phase, selected by G2b.

No FIR prototype, runtime guard, frozen probe manifest, or acceptance
threshold was changed. Teacher targets remain exact 2:1 decimations and
input-Nyquist limited; all three final data audits report zero decimation
error.

## Supplementary distortion diagnostics

The coherent-line report remains diagnostic rather than an acceptance gate.
For 48 kHz it reports THD -76.70 dB, SMPTE IMD -85.91 dB, CCIF IMD -85.87 dB,
and the strongest added 10 kHz AM sideband at -111.60 dB. The authoritative
frozen G9 probe is lower at -144.40 dB because its metric and stimulus are
defined separately. See `report.md`, `summary.json`, and the per-family plots
in this directory for the complete ideal/Bessel/sharp/gentle/CAPB comparison.
