# ABX Listening Protocol (Issue #59 / #64)

## Purpose

Validate README 7.3 listening requirements with a reproducible ABX workflow:

1. NMSE output reduces mirror-related harshness/graininess versus naive output.
2. Transient attack remains perceptually intact (no clear blunting).

The protocol uses frozen triplets:

- `input`: Stage 1 input reference
- `naive`: Naive reference path
- `nmse`: Selected NMSE checkpoint output

## References and Fixed Assets

- Pair definition: `tests/fixtures/golden_samples/abx_pairs.json`
- Frozen checkpoint metadata:
  `tests/fixtures/golden_samples/issue64_model_selection.json`
- Frozen model: `stage1_best.pt`
- Frozen model SHA-256:
  `ef9f2815e57663140eff3ff6860847148eb527c733e17e2890cba9a24ec4e025`

## Minimum Sample Set

Use the exact pair map from `abx_pairs.json`. The minimum fixed set is:

- `sample_a`
- `sample_b`

Each sample must include `input`, `naive`, and `nmse` files at the declared
paths in `abx_pairs.json`.

## Evaluation Environment

- Quiet environment (recommended ambient noise < 35 dBA)
- Same playback chain for all trials (DAC / amp / headphone or monitor)
- Playback sample rate fixed per session (no auto SRC switching)
- Disable all DSP enhancements (EQ, loudness normalizer, crossfeed, limiter)
- Listener takes a short break every 15-20 trials to reduce fatigue bias

## Loudness Matching Rule

Before ABX trials, loudness-match `naive` and `nmse` per sample:

1. Measure integrated loudness (LUFS) if available.
2. If LUFS is unavailable, use RMS in the same analysis window.
3. Gain-adjust one side so level difference is within ±0.1 dB.
4. Store applied gain in trial metadata.

`input` is used only as contextual reference and is not part of ABX hit scoring.

## ABX Trial Procedure

1. Choose one sample (`sample_a` or `sample_b`).
2. Randomize assignment for `A` and `B` (`naive`/`nmse`).
3. Randomize `X` as either `A` or `B`.
4. Listener can replay A/B/X freely during that trial.
5. Listener answers `X == A` or `X == B`.
6. Record confidence (`low`, `mid`, `high`) and free notes.
7. Repeat at least 10 trials per sample (20+ total recommended).

## Judgment Criteria

### Quantitative (ABX correctness)

- Use one-sided binomial test with null hypothesis `p = 0.5`.
- Significance threshold: `p < 0.05`.
- Practical thresholds:
  - 10 trials: 9+ correct
  - 12 trials: 10+ correct
  - 16 trials: 12+ correct
  - 20 trials: 15+ correct

### Qualitative (README 7.3 alignment)

Review free notes and confirm both:

1. `nmse` is described as less harsh/grainy/metallic than `naive`.
2. No obvious attack softening is reported for `nmse`.

If ABX significance is met but qualitative notes contradict these points, mark
the session as "needs follow-up".

## Recording Format

Use both templates:

1. Trial-level log:
   `docs/templates/abx_trial_log_template.csv`
2. Session summary:
   `docs/templates/abx_session_summary_template.md`

Minimum recorded fields:

- Trial data: sample id, A/B assignment, X truth, answer, correctness
- Quantitative result: hit rate, trial count, binomial p-value
- Qualitative result: harshness and transient comments (free text)

## Save Location and Reproducibility

Store each listening session under:

- `reports/abx/<session_id>/`

Recommended structure:

- `reports/abx/<session_id>/trial_log.csv`
- `reports/abx/<session_id>/summary.md`
- `reports/abx/<session_id>/session_meta.json`

`session_meta.json` should include listener ID, date, playback chain, room notes,
sample rate, loudness-matching method, and commit hash.

Reproducibility steps:

1. Checkout commit under evaluation.
2. Confirm frozen pairs and checkpoint hash:
   - `tests/fixtures/golden_samples/abx_pairs.json`
   - `tests/fixtures/golden_samples/issue64_model_selection.json`
3. Copy templates from `docs/templates/`.
4. Execute ABX following this protocol and save all artifacts under
   `reports/abx/<session_id>/`.

## Update Rule

When checkpoint selection changes, update all of:

1. `tests/fixtures/golden_samples/abx_pairs.json`
2. `tests/fixtures/golden_samples/issue64_model_selection.json`
3. `tests/fixtures/golden_samples/regression_baseline.json`
4. This document's "References and Fixed Assets" section
