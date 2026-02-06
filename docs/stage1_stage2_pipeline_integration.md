# Stage1 (NMSE) -> Stage2 (HIE) Pipeline Integration (Issue #25)

## Purpose

Implement and validate an end-to-end integration path:

- `44.1kHz -> 88.2kHz` (Stage1)
- `88.2kHz -> 705.6kHz` (Stage2, `2x x 2x x 2x`)

with configuration-driven execution, long-audio boundary handling, and reproducible performance reporting.

## Implemented Components

- `src/totton_audio_de_mirroring/inference/pipeline.py`
  - `Stage1Processor` interface
  - `ReferenceStage1Processor` (baseline 2x SRC)
  - `NMSEStage1Processor` + `load_nmse_stage1_processor()` (checkpoint inference path)
  - `run_stage1_stage2_pipeline()` (chunked execution, crossfade stitching, performance/memory report)
- `scripts/run_stage1_stage2_pipeline.py`
  - YAML configuration-driven CLI
  - `.npy` / `.wav` input support
  - benchmark mode (`--benchmark-duration-sec`)
  - JSON output including Stage1 hard metrics and throughput
- `configs/stage1_stage2_pipeline.yaml`
  - default integration runtime config

## Boundary Handling

Long audio is processed with input-domain chunks (`chunk_duration_sec`) and overlap (`crossfade_duration_sec`).
Each chunk is individually processed through Stage1 and Stage2, then merged with linear crossfade at both:

- Stage1 domain (for metric assembly)
- Final 705.6kHz output domain

This avoids hard boundary discontinuities while keeping the implementation deterministic and configuration-driven.

## Hard Requirement Checks

When `evaluate_stage1_metrics=true`, the pipeline computes Stage1 hard metrics (`evaluate_stage1_hard_metrics`) against a reference 2x SRC baseline:

- low-band preservation metrics (amplitude / phase / group delay)
- mirror reduction ratio
- high-band energy cap violation flag
- touch minimization metric

## Benchmark / Memory Measurement

Command:

```bash
uv run python scripts/run_stage1_stage2_pipeline.py \
  --config configs/stage1_stage2_pipeline.yaml \
  --benchmark-duration-sec 60 \
  --json
```

Measured on this implementation (2026-02-06):

- Input duration: `60.0 sec`
- Output samples: `42,336,000` (`705,600 Hz`)
- Latency: `14.048973835015204 sec`
- Throughput: `4.270774556534368 x realtime`
- Peak memory: `1369.77734375 MB` (process peak RSS)
- Stage1 energy cap violated: `false` (`cap=0.001`)

## End-to-End Test Coverage

- `tests/test_inference_pipeline.py`
  - chunk splitting
  - crossfade stitching
  - end-to-end output generation
  - energy cap violation detectability
- `tests/test_run_stage1_stage2_pipeline_script.py`
  - configuration-driven CLI execution and JSON payload validation
