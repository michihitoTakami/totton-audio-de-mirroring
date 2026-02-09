# Stage1 (NMSE) -> Stage2 (HIE) Pipeline Integration (Issue #25 / #55)

## Purpose

Provide an end-to-end integration path:

- `44.1kHz -> 88.2kHz` (Stage1)
- `88.2kHz -> 705.6kHz` (Stage2, `2x x 2x x 2x`)

while routing Stage2 through the **C++ core API** defined in EPIC #22 / Issue #43.

## Implemented Components

- `src/totton_audio_de_mirroring/inference/pipeline.py`
  - `Stage1Processor` interface
  - `ReferenceStage1Processor` / `NMSEStage1Processor`
  - `Stage2Processor` interface
  - `CppStage2Processor` (default runtime path)
  - `PythonStage2Processor` (regression/testing fallback)
  - `run_stage1_stage2_pipeline()` with chunked E2E processing
- `src/totton_audio_de_mirroring/stage2/cpp_backend.py`
  - `ctypes` bridge for C++ Stage2 core API
  - automatic CMake configure/build for `tadm_dsp_capi`
  - stateful handle lifecycle (`create/process_block/destroy`)
- `cpp/src/multistage_upsampler_c_api.cpp`
  - C API wrapper over `MultiStageUpsampler`
  - stage taps loaded from `stage{i}_taps.txt`
  - streaming-safe `process_block` API
- `scripts/run_stage1_stage2_pipeline.py`
  - YAML-driven CLI
  - `.npy` / `.wav` input
  - benchmark mode (`--benchmark-duration-sec`)
  - JSON output including Stage2 backend and performance
- `configs/stage1_stage2_pipeline.yaml`
  - default runtime config (`stage2_backend: cpp`)

## Stage1 -> Stage2 Interface Contract

- Signal type: `numpy.ndarray` (`float64`, mono, 1D)
- Stage1 output sample rate: `88,200 Hz`
- Stage2 input sample rate: `88,200 Hz` (exact handoff, no resampling between stages)
- Stage2 output length: `input_length * (2 ** stage2_num_stages)`
- Stage2 processing: stateful block processing via C++ `MultiStageUpsampler`

## Boundary Handling

Long audio is processed in input-domain chunks (`chunk_duration_sec`) with
**Hann window + 50% overlap-add** (`overlap_ratio: 0.5`, `chunk_window: hann`).
Each chunk passes through Stage1 and Stage2, then stitched in:

- Stage1 domain (for metric assembly)
- Final 705.6kHz output domain

## Hard Requirement Checks

When `evaluate_stage1_metrics=true`, Stage1 hard metrics are computed against reference 2x SRC baseline:

- low-band preservation (amplitude / phase / group delay)
- mirror reduction ratio
- high-band energy cap violation flag
- touch minimization metric

## Benchmark / Memory Measurement (C++ Stage2 Path)

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
- Latency: `13.963446070963982 sec`
- Throughput: `4.296933557452257 x realtime`
- Peak memory: `1370.30859375 MB` (process peak RSS)
- Stage1 energy cap violated: `false` (`cap=0.001`)
- Stage2 backend: `cpp`

`performance` payload now also reports:

- `num_chunks`
- `chunk_latency_ms` (mean per-input-chunk latency)

## End-to-End Test Coverage

- `tests/test_inference_pipeline.py`
  - chunk splitting
  - Hann-window overlap-add stitching
  - end-to-end output generation
  - energy cap violation detectability
- `tests/test_chunk_processor.py`
  - overlap-add reconstruction
  - long-duration streaming (10min, `@pytest.mark.slow`)
- `tests/test_run_stage1_stage2_pipeline_script.py`
  - YAML-driven CLI and JSON payload validation
- `tests/test_stage2_cpp_backend.py`
  - C++ backend bridge parity with Python cascade
  - missing Stage2 config directory error path
