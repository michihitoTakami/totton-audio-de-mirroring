---
name: distillation
description: "Use to distill the Stage 1 NMSE into a lightweight student model and compare it against teacher/FIR baselines. Trigger: distillation, model compression, pruning, 蒸留, モデル圧縮, 軽量化."
---

# Distillation (Stage 1 軽量化)

Stage 1 NMSE（~14.2M params）を teacher として軽量 student（目標 5–7M params）へ蒸留し、品質劣化がないか比較検証する。

## Execution Steps

```bash
# 1. 蒸留学習
uv run python scripts/train_distillation.py \
  --data-config configs/data_generation_distill_demo.yaml \
  --train-config configs/training_distillation_stage1.yaml \
  --teacher-checkpoint data/checkpoints/<teacher_best>.pth \
  --device cuda --seed 1234
# student 構造の調整: --base-channels N --num-downsamples N --channel-multiplier N
# プルーニング併用: --pruning-ratio 0.3

# 2. teacher vs student 比較（メトリクス + 成果物出力）
uv run python scripts/compare_distillation_model_v2.py \
  --distill-checkpoint data/checkpoints/<student>.pth \
  --baseline-checkpoint data/checkpoints/<teacher_best>.pth \
  --data-config configs/data_generation_distill_demo.yaml \
  --output-dir reports/distillation_comparison/<YYYYMMDD>_<variant> \
  --device cuda

# 3. FIR ベースラインとの比較
uv run python scripts/compare_distillation_vs_fir.py
```

## Acceptance

- student も Stage 1 受入基準（`stage1-evaluation` スキル参照）を全て満たすこと
- 特に `hb_energy_cap_violation_rate == 0.0` とリンギング非退行は必須
- 比較レポートは `reports/distillation_comparison/` に日付 + variant 付きで保存

## Constraints

- teacher policy は raw88 を継承（teacher checkpoint の学習条件に合わせる）
- 実装は `src/totton_audio_de_mirroring/training/distillation.py` /
  `src/totton_audio_de_mirroring/models/nmse_light.py`

## References

- `docs/model_compression_distillation_pruning.md` — 圧縮戦略の全体像 (#97)
- 蒸留後のデプロイは `model-export-verify` スキル
