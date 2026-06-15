# Hi-Res Teacher Data (Genuine ≥88.2kHz Ground Truth)

## Why

The default `raw` teacher path synthesizes the 88.2kHz teacher from analytic
signals (tones, sweeps, noise) and derives the high-band suppression target with
a mirror-detection DSP rule. The network therefore learns to imitate a
hand-designed DSP suppressor, and its quality ceiling is that DSP rule.

Real instruments and voices carry **structured energy above 22.05kHz** that
synthetic generators cannot reproduce. Using genuine hi-res recordings
(native sample rate ≥ the Stage 1 target rate) as the teacher lets the network
learn a physically faithful high-band target. This stays within the
anti-hallucination policy: the high band is still real recorded content, the
0–20kHz band remains a structural bypass, and the energy cap is still enforced.

## How it fits the pipeline

`HiResTeacherDataset` reuses the existing `raw` teacher flow:

1. Load a teacher chunk from a hi-res file at the target rate (88.2kHz).
2. Downsample the same chunk to 44.1kHz to form the input proxy.
3. Degrade → band-split → `project_teacher_hb_target` (amplitude-capped).

The produced batch (`high_band`, `hb_target`, `mirror_mask`, …) is identical in
shape and keys to `MirrorSuppressionDataset`, so the training loop is unchanged.

## Data requirements

- **Native sample rate ≥ 88,200 Hz.** 48kHz "hi-res" files are rejected — their
  Nyquist (24kHz) holds almost no content above 22.05kHz. (Note: VCTK's public
  release is 48kHz and is therefore *not* usable here.)
- **License: CC0, CC-BY (any version), or public domain.** SA/ND are accepted;
  NC requires `--allow-noncommercial`.
- **Genuine ultrasonic energy** above 22.05kHz, verified automatically. Files
  upsampled from a band-limited master are rejected.

## Acquiring data

Two options:

1. **Drop your own licensed hi-res files** into `data/hires_corpus/` (any
   directory of `.wav/.flac/.aiff`). The corpus loader reads them directly.
2. **Manifest-driven download** of verified CC-BY/CC0 sources:

   ```bash
   # Edit configs/hires_teacher_manifest.yaml to add verified sources first.
   uv run python scripts/download_hires_teacher_data.py \
     --manifest configs/hires_teacher_manifest.yaml \
     --output-dir data/hires_corpus
   ```

   The downloader validates license, sample rate, optional SHA-256, and HF
   energy, then writes `data/hires_corpus/ATTRIBUTION.md` and
   `downloaded_manifest.json` for provenance.

> No default sources are bundled: genuine CC-BY/CC0 audio at ≥88.2kHz is scarce
> and source URLs are unstable. Populate the manifest with sources you have
> verified (e.g. CC0/CC-BY 96kHz field recordings, your own hi-res masters).

## Training

```bash
uv run python scripts/train_stage1.py \
  --data-config configs/data_generation_hires88k2.yaml \
  --hires-root data/hires_corpus \
  --teacher-type raw_88k2
```

When `--hires-root` is omitted, training falls back to the synthetic dataset.

## Validation checklist

- 0–20kHz preservation is still guaranteed by the band-split structure.
- `hb_energy_cap_violation_rate == 0` must hold (run `scripts/evaluate_stage1.py`).
- Compare against the synthetic-teacher baseline using the win/loss report and
  microstructure comparisons before adopting a hi-res-trained checkpoint.
