---
name: model-export-verify
description: "Use to export a CAPB Stage 1 checkpoint to ONNX and verify waveform-level numerical parity. Trigger: export ONNX, model export, エクスポート, ONNX変換."
---

# CAPB Model Export & Verify

CAPB Stage 1 チェックポイントを、入力・出力とも波形の ONNX モデルへエクスポートする。出力時間軸は入力の 2 倍で、入力レート系列は ONNX metadata に保存する。

## Execution Steps

```bash
uv run --with onnx --with onnxruntime \
  python scripts/export_capb_to_onnx.py \
  --checkpoint data/checkpoints/<run>/capb_best.pt \
  --output data/checkpoints/<run>/capb_stage1.onnx
```

必要に応じて `--verify-lengths LEN1 LEN2 LEN3`、`--tolerance TOL`、`--opset-version N` を指定する。

## Verification Checklist

- ONNX checker が成功すること
- 複数の不均一な入力長で PyTorch と ONNX Runtime の shape が一致すること
- 全入力長の最大絶対誤差が tolerance 以下であること
- `expected_input_rate` metadata がチェックポイントのレート系列と一致すること
- エクスポート先の SHA-256 を記録すること

## Constraints

- CAPB の固定 prototype と convex blend をそのままエクスポートし、別の波形生成器へ置き換えない
- 量子化や TensorRT 化は別途、対象 Jetson 上で全 probe gate を再検証してから採用する
- 生成された ONNX、チェックポイント、検証レポートは明示依頼なしに Git へ追加しない
