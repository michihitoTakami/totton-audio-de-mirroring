---
name: pipeline-inference
description: "Use to run Stage1→Stage2 inference (44.1kHz → 705.6kHz upsampling) on audio files, including stereo and the totton-upsample CLI. Trigger: run pipeline, upsample audio, inference, 推論実行, アップサンプル, パイプライン実行."
---

# Pipeline Inference (Stage1 → Stage2)

44.1kHz 入力を Stage1（NMSE, →88.2kHz）+ Stage2（DSP HIE, →705.6kHz）で 16× アップサンプルする。

## Execution Steps

```bash
# 1. エンドユーザー向けバッチ CLI（wav/flac 対応、glob 可）
uv run totton-upsample input.wav -o output.wav \
  -c configs/stage1_stage2_pipeline.yaml \
  --device cuda --output-format wav --output-format metadata
# 複数入力時は -o は出力ディレクトリ。--fail-fast / --log-level DEBUG も可

# 2. 開発用パイプラインスクリプト（npy/wav 入出力、ベンチマーク）
uv run python scripts/run_stage1_stage2_pipeline.py \
  --config configs/stage1_stage2_pipeline.yaml \
  --input-wav input.wav --output-wav output.wav
# 合成サイン波ベンチ: --benchmark-duration-sec 2.0 --json

# 3. ステレオ（L/R 独立処理）
uv run python scripts/run_stereo_pipeline.py \
  --config configs/stage1_stage2_pipeline.yaml \
  --input-wav input_stereo.wav --output-wav output_stereo.wav
```

## Stage1 Backend Selection

`configs/stage1_stage2_pipeline.yaml` の `stage1.mode` で切替:

- `reference` — 参照 SRC（NMSE なし、比較ベースライン）
- `nmse` — PyTorch チェックポイント
- `onnx` — ONNX Runtime（`docs/onnx_runtime_benchmark.md`）
- `tensorrt` — TensorRT エンジン（Jetson 向け）

## Constraints

- 出力の 0–20kHz は入力と同一であること（構造保証だが、疑わしければ `audio-visualization` スキルで確認）
- 数秒オーダーのレイテンシは許容（非リアルタイム設計）
- 長尺ファイルはチャンク + 50% overlap (Hann) で処理される

## References

- `docs/stage1_stage2_pipeline_integration.md` — 統合設計
- `src/totton_audio_de_mirroring/cli.py` / `src/totton_audio_de_mirroring/inference/pipeline.py`
