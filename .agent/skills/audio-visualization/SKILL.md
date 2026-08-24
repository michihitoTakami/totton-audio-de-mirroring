---
name: audio-visualization
description: "Use to inspect CAPB spectrograms, image ridges, and impulse/step responses. Trigger: visualize, spectrogram, impulse response, plot, 可視化, 波形確認."
---

# CAPB Audio Visualization

Use `scripts/run_capb_phase0.py` for prototype response summaries and `scripts/evaluate_probe_gates.py` for reportable measurements. When adding plots, compare CAPB against both Bessel (ringing reference) and ideal polyphase (fidelity reference).

Inspect plateau ripple, overshoot, pre-echo, sweep image ridge, low-band gain, phase, and group delay. Visuals are diagnostic; the versioned worst-case gate report is authoritative.
