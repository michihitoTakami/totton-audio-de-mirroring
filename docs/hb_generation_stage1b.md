# Stage 1b — Neural Bandwidth Extension (controlled HB generation)

## Why this exists (and why it is opt-in)

The default Stage 1 NMSE applies a `[0, 1]` suppression mask. An experiment with
genuine hi-res teacher data showed that **better data alone does not raise the
quality ceiling** for real high-band reconstruction: on real hi-res clips the
high-band (20-44kHz) log-magnitude L1 error vs the true native high band was
3.255 for the synthetic-trained teacher and 3.260 (no gain) after hi-res
fine-tuning. The reason is architectural — content above 22.05kHz is destroyed
by downsampling to 44.1kHz, and a suppress-only mask cannot recreate what is not
in the degraded input.

Stage 1b (NBEE) therefore **generates** the high band from the full-band
context, trained on real native high-band targets. This deliberately crosses the
project's default anti-hallucination line, so it is a **separate, opt-in engine**:

- The suppression NMSE is unchanged and remains the default.
- NBEE lives in its own model class, scripts, configs, and `stage1b_*` checkpoints.

## Safety constraints (retained from NMSE)

- **0–20kHz is a structural bypass** (`LB_out = LB_in`) — never modified.
- **High-band energy cap** (IMD safety) — generation is bounded, not free.
- **Envelope shaping** — gentle high-frequency decay.
- **Band-limit** — output high band is high-pass filtered (no low-band leakage).
- Ringing edge/step losses retained to limit transient regression.

The energy cap is raised for generation (default `1e-2` vs `1e-3` suppression)
so real high-band energy can be reproduced; tune against the IMD proxy.

## Architecture

`NeuralBandwidthExtension` (`src/.../models/nbee.py`) subclasses `NMSE` and reuses
its band-split, STFT/iSTFT, and safety buffers. Differences:

- UNet input = **full-band** magnitude (so it has 0–20kHz context to infer HB).
- UNet output activation = **linear** (`none`), predicting an **absolute** HB
  magnitude (`clamp(min=0)`), not a `[0, 1]` gain.
- `apply_safety_constraints` shapes/caps the generated magnitude; the complex
  spectrum uses the input phase (v1). Phase prediction is a future v2.

## Generative target

`build_generative_hb_target` (`src/.../data/mirror_detection.py`) keeps the FULL
real teacher magnitude (no `min(mag_teacher, mag_in)` cap, no suppression floor),
shaped by the shared envelope + energy cap. Datasets emit it via
`assemble_stage1_sample(target_mode="generate")` (wired into both
`MirrorSuppressionDataset` and `HiResTeacherDataset`).

## Usage

```bash
# Train (synthetic raw88 generative target + optional hi-res mix)
uv run python scripts/train_stage1b_generation.py \
  --data-config configs/data_generation_gen88k2.yaml \
  --hires-root data/hires_corpus --energy-cap 1e-2 \
  --epochs 30 --output data/checkpoints/stage1b/stage1b_nbee.pt

# Evaluate the ceiling (HB vs true native) against the suppression baseline
uv run python scripts/evaluate_hb_generation.py \
  --nbee-checkpoint data/checkpoints/stage1b/stage1b_nbee.pt \
  --baseline-checkpoint data/checkpoints/stage1_best_raw88.pt \
  --hires-root data/hires_corpus --energy-cap 1e-2
```

## Results (honest, measured) — generation did NOT raise the faithful ceiling

Two variants were trained (synthetic raw88 generative target + the 62-file
hi-res corpus, 20 epochs) and evaluated on 15 real hi-res clips:

| Method | HB(20-44kHz) log-mag L1 vs true native (lower better) |
|--------|--------------------------------------------------------|
| degraded input | 3.66 |
| **suppression NMSE baseline** | **3.28** |
| NBEE v1 (absolute magnitude) | 5.76 |
| NBEE v2 (suppression-anchored residual + gated add) | 5.98 |

Fair, energy-weighted checks (only bins where the true HB has energy) agree:
- weighted HB log-mag error: baseline 3.24 vs v2 3.42 (baseline better, 53% win);
- linear-mag error in the loudest 20% of true HB bins: baseline 0.115 vs v2 0.350.

Safety held throughout: **HB energy-cap violations 0/15**, **LB(0-20kHz)
preservation error ~4e-4** (structural bypass intact).

### Conclusion

Controlled high-band generation does **not** beat suppression on faithful
reconstruction, by any metric tried. Reasons:

1. The 20–22.05kHz band already exists in the 44.1kHz input — suppression
   preserves it optimally; generation only perturbs it.
2. The 22.05–44kHz band is genuinely lost; recreating the *correct* content is
   ill-posed, so the generator produces plausible-but-wrong energy that is
   *further* from truth than silence.
3. The mandated safety constraints (energy cap, envelope) cap how much real
   high-band energy can be reproduced anyway.

The faithful-reconstruction ceiling is therefore fundamental, not an
architecture or data shortfall. Raising perceived quality would require
abandoning faithful-L1 objectives for **perceptual ones** (adversarial / A-B
listening) and relaxing the safety envelope — a research-grade direction that
moves further from the project's anti-hallucination and IMD-safety principles.

This engine is retained as a **tested, safe, opt-in experimental scaffold** and a
documented negative result; it is not recommended for production over the
suppression NMSE.

## Limitations

- v1/v2 reuse the input phase for the generated band (phase above 22.05kHz is
  mirror-derived and approximate); magnitude/energy dominate HF perception.
- The hi-res corpus is small; synthetic raw88 is the primary data to avoid
  overfitting. Generation is bounded by the energy cap by design.
- Evaluation is faithful-reconstruction (log-mag L1 vs true native). A
  perceptual evaluation (listening / adversarial) would be needed to judge
  plausibility, which is a different objective than faithfulness.
