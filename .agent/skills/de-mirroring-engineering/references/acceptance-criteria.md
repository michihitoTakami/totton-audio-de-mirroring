# CAPB Acceptance Criteria

Run `scripts/evaluate_probe_gates.py --backend capb` for `44k1` and `48k`.

- Plateau RMS/P2P ratio: at most 1.10 against Bessel reference, with absolute floors.
- Overshoot increase: at most 5e-3.
- Pre-echo energy growth: at most 1.44, with an amplitude-relative floor.
- Integrated image/main and sweep-ridge image peak: at most -65 dB.
- Added HF against Bessel: at most +3 dB unless already negligible.
- Smoothed 100 Hz–18 kHz dip/boost: at most ±1 dB.
- 18–20 kHz dip: at most 3 dB.
- Low-band gain error: at most 0.5 dB.
- Low-band phase error: at most 15 degrees.
- Low-band group-delay error: at most 600 samples.
- Low-band waveform error: at most -20 dB.
- SMPTE-style modulation sidebands on canonical and held-out two-tone probes:
  at most -110 dBc (G9, gate specification v4 / probe manifest v2).

Do not weaken a threshold to admit a checkpoint. A threshold or probe change requires independent physical justification and before/after reports using the same candidate.
