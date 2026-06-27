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

## Results (2): transient de-ringing on the transparent FIR base — also negative

After replacing Bessel IIR with the 32-bit-transparent Kaiser FIR (which removes
mirror images entirely), the only remaining transient artifact is the inherent
band-limited (Gibbs) ringing. An NBEE was trained with `input_mode=transparent`,
ringing-weighted losses (edge/step = 1.0), and the hi-res transient corpus to add
>22.05kHz content that sharpens edges. Measured on square probes
(`scripts/evaluate_antiringing.py`):

| probe | overshoot FIR | overshoot +NBEE | ripple FIR | ripple +NBEE |
|-------|---------------|-----------------|------------|--------------|
| 500Hz | 0.274 | 0.294 | 0.0063 | 0.0051 |
| 1kHz  | 0.296 | 0.303 | 0.000  | 0.000  |
| 5kHz  | 0.416 | 0.409 | 0.000  | 0.000  |

The de-ringer does **not** meaningfully reduce ringing (overshoot is slightly
worse at 500Hz/1kHz). Under the energy cap, the generated high band cannot add
enough correct edge-sharpening content to cancel the Gibbs ringing; it mostly
perturbs the overshoot. This confirms, a second time, that generation under the
safety constraints does not raise the ceiling.

### Removing the safety valve makes ringing strictly WORSE

To test the hypothesis "hallucinate freely (no energy cap) to remove ringing",
the cap was swept up. 500Hz square overshoot / plateau-ripple:

| energy cap | overshoot | ripple |
|------------|-----------|--------|
| FIR only (no generation) | 0.274 | 0.0063 |
| 1e-2 (weak generation)   | 0.294 | 0.0051 |
| 0.1  (10x generation)    | 0.559 | 0.0365 |
| inf  (uncapped)          | clips at +13 dBFS (target peak 4.59) — training aborts |

**More generation monotonically worsens ringing.** This is exactly the Fourier
result: added high-frequency energy (with the input phase, and unconstrained
magnitude) *steepens* the edge and *adds* in-phase energy at the transition,
increasing overshoot rather than cancelling it. Mis-phased HF piles onto the
ringing instead of removing it; fully unbounded generation clips the signal.

**Definitive conclusion: ringing cannot be removed by hallucination — it is
intrinsic to band-limiting a discontinuity (Gibbs), and generating more high
band makes it worse.** The only real levers are: the transparent FIR (minimal
correct reconstruction, Part A), apodization (less ring at the cost of high-
frequency dulling), minimum-phase (relocation only), or accepting the ~9% Gibbs
overshoot (a ~22kHz oscillation that is largely inaudible and gently removed by
the final analog reconstruction filter).

## Overall conclusion

- **The real win is Part A — the 32-bit-transparent Kaiser FIR upsampler**, which
  eliminates mirror images at the source (image at 21kHz: -4.8 dB Bessel IIR ->
  -168 dB FIR) and is transparent below the float32 floor. It makes the original
  neural mirror-suppression largely unnecessary.
- **Neural high-band generation (NBEE) does not help** — neither faithful HB
  reconstruction nor transient de-ringing beats the safe baseline, because the
  >22.05kHz content is unrecoverable and the IMD-safety constraints cap
  generation. Retained as a tested, opt-in scaffold and a documented negative
  result; not recommended for production.

## Limitations

- v1/v2 reuse the input phase for the generated band (phase above 22.05kHz is
  mirror-derived and approximate); magnitude/energy dominate HF perception.
- The hi-res corpus is small; synthetic raw88 is the primary data to avoid
  overfitting. Generation is bounded by the energy cap by design.
- Evaluation is faithful-reconstruction (log-mag L1 vs true native). A
  perceptual evaluation (listening / adversarial) would be needed to judge
  plausibility, which is a different objective than faithfulness.
