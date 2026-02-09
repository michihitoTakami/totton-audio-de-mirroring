# ONNX Runtime Benchmark (Issue #96)

## Scope

Issue #96 introduces:

1. ONNX export script for Stage 1 (`scripts/export_to_onnx.py`)
2. ONNX Runtime Stage 1 processor (`src/totton_audio_de_mirroring/inference/onnx_processor.py`)
3. Stage1 backend integration in pipeline CLI (`stage1.mode=onnx`)
4. Full-song comparison CLI (`scripts/compare_full_song.py`)

This benchmark note is GPU-oriented (CUDA provider) as requested.

## Export Validation

```bash
uv run --with onnx --with onnxruntime python scripts/export_to_onnx.py \
  --checkpoint-path data/checkpoints/stage1_best.pt \
  --data-config-path configs/data_generation.yaml \
  --output-path data/checkpoints/stage1_best.onnx \
  --opset-version 17 \
  --dummy-samples 22050 \
  --check-model \
  --verify-ort \
  --tolerance 1e-5
```

Expected result:

- ONNX checker passes
- ONNX Runtime parity check passes (`max_abs_error < 1e-5` on export dummy input)

## GPU Benchmark Procedure (CUDAExecutionProvider)

```bash
uv run --with onnxruntime-gpu python scripts/compare_full_song.py \
  --config configs/stage1_stage2_pipeline.yaml \
  --input-wav /path/to/input_44k1.wav \
  --output-dir reports/compare_full_song_gpu \
  --checkpoint-path data/checkpoints/stage1_best.pt \
  --onnx-model-path data/checkpoints/stage1_best.onnx \
  --onnx-device cuda
```

Default behavior:

- `--onnx-device cuda` で CUDA provider が無い場合はエラー終了
- CPU fallback は `--allow-onnx-cpu-fallback` を明示した場合のみ許可

Generated artifact:

- `reports/compare_full_song_gpu/summary.json`
  - `reference.performance.*`
  - `nmse.performance.*`
  - `onnx.performance.*`

Compare:

1. `nmse.performance.latency_sec`
2. `onnx.performance.latency_sec`
3. Stage1 hard metrics (`stage1_metrics`) regression

## Acceptance Mapping

Issue #96 acceptance criteria mapping:

1. ONNX export + checker: covered by `scripts/export_to_onnx.py --check-model`
2. ONNX Runtime numerical parity: covered by `--verify-ort --tolerance 1e-5`
3. Performance comparison: covered by `scripts/compare_full_song.py` summary
4. Quality regression guard: use existing Stage1 hard metrics in pipeline output
