---
name: model-export-verify
description: "Use to export the Stage 1 model to ONNX / TensorRT and verify numerical equivalence and quantization noise. Trigger: export ONNX, TensorRT, model export, エクスポート, ONNX変換, TensorRT変換."
---

# Model Export & Verify (ONNX / TensorRT)

Stage 1 NMSE チェックポイントを ONNX / TensorRT へエクスポートし、数値等価性と量子化ノイズを検証する。デプロイ先は Jetson Orin Nano (8GB)。

## Execution Steps

```bash
# 1. PyTorch → ONNX（モデル検証 + ONNX Runtime での出力一致確認付き）
uv run python scripts/export_to_onnx.py \
  --checkpoint-path data/checkpoints/<best>.pth \
  --data-config-path configs/data_generation.yaml \
  --output-path data/checkpoints/<best>.onnx \
  --check-model --verify-ort
# 他: --opset-version N --dummy-samples N --disable-dynamic-axes --device cpu --tolerance TOL

# 2. ONNX → TensorRT エンジン（Jetson 上で実行）
uv run python scripts/export_to_tensorrt.py \
  --onnx-path data/checkpoints/<best>.onnx \
  --output-dir data/checkpoints/trt \
  --modes fp16
# 他: --workspace-mb N --freq-bins N --min/opt/max-time-frames N --strict-mixed-io-fp32

# 3. FP16 量子化ノイズの確認（SNR / error RMS ゲート付き）
uv run python scripts/check_fp16_quantization_noise.py \
  --min-snr-db <threshold> --json
```

## Verification Checklist

- `--verify-ort` で PyTorch と ONNX Runtime の出力差が tolerance 内であること
- FP16 化後も `hb_energy_cap_violation_rate == 0.0` を維持すること（safety constraints は後処理なので通常保たれる）
- エクスポート後はパイプラインの `stage1.mode=onnx` / `tensorrt` でフルソング比較
  （`scripts/compare_full_song.py`、`pipeline-inference` スキル参照）

## Constraints

- TensorRT エクスポートは対象デバイス（Jetson）上で行う（エンジンはデバイス固有）
- dynamic axes を無効化する場合は推論側チャンクサイズと一致させる

## References

- `docs/onnx_runtime_benchmark.md` — ONNX Runtime 統合とベンチマーク結果
- `docs/model_compression_distillation_pruning.md` — 軽量化との組み合わせ
