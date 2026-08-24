---
name: de-mirroring-engineering
description: "Use for CAPB audio-path architecture, prototype policy, controller constraints, and acceptance-gate changes. Trigger: architecture, design decision, mirror suppression policy, 設計方針, アーキテクチャ."
---

# CAPB Engineering

CAPB Stage 1 blends fixed, symmetric, gain-matched interpolation FIRs with a shared group delay. The controller may select convex weights only; it must not alter kernels or synthesize a free waveform.

## Invariants

- No reconstruction or generation above input Nyquist.
- No hard 20 kHz split: it reintroduces Gibbs ringing.
- Every prototype stays symmetric, gain-matched, and centered to one delay.
- Stationary content requires strong image rejection; discontinuities require the validated low-ringing endpoint.
- Acceptance binds on the worst canonical or held-out probe, never the mean.
- Both 44.1→88.2 kHz and 48→96 kHz families must pass before release.

Read `references/acceptance-criteria.md` before changing prototypes, guards, losses, probe definitions, or thresholds.
