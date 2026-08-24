---
name: pipeline-inference
description: "Use to run CAPB Stage 1 to DSP Stage 2 offline upsampling with totton-upsample. Trigger: run pipeline, upsample audio, inference, 推論実行, アップサンプル."
---

# CAPB Pipeline Inference

Set `stage1.mode: capb`, `checkpoint_path`, and `device` in `configs/stage1_stage2_pipeline.yaml`, then run:

```bash
uv run totton-upsample input.wav -o output.wav \
  -c configs/stage1_stage2_pipeline.yaml --output-format wav --output-format metadata
```

`reference` is a wiring baseline, not the release backend. Long inputs use Hann 50% overlap-add. The standard pipeline is 44.1→88.2→705.6 kHz; validate any rate-family change before processing.
