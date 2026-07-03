---
name: stage1-training
description: "Use to train the Stage 1 NMSE (Neural Mirror Suppression Engine) with the raw88 teacher policy. Trigger: train stage1, NMSE training, 学習実行, Stage1学習, モデル学習."
---

# Stage 1 Training (NMSE)

Stage 1 NMSE（44.1kHz → 88.2kHz、ミラー抑制）の学習を実行する。

## Execution Steps

```bash
# 基本形（data config + training config を指定）
uv run python scripts/train_stage1.py \
  --data-config configs/data_generation.yaml \
  --config configs/training_stage1.yaml \
  --device cuda --seed 1234

# 主なオーバーライド
#   --batch-size 32 --epochs N --learning-rate LR --energy-cap CAP
#   --teacher-type {raw88|bessel} （デフォルトは config 依存、標準は raw88）
#   --edge-loss-weight / --step-loss-weight （リンギング抑制の補助損失）
#   --checkpoint-dir data/checkpoints --resume-from <ckpt>
#   --auto-batch-size --min-batch-size N --max-oom-retries N （OOM対策）
#   --no-amp （mixed precision 無効化）
```

## Config Families

| config | 用途 |
|--------|------|
| `configs/data_generation.yaml` + `configs/training_stage1.yaml` | 標準（合成データ） |
| `configs/data_generation_176k4*.yaml` + `configs/training_stage1_176k4.yaml` | 176.4kHz raw176k4 パス（zero-stuff 入力） |
| `configs/data_generation_hires88*.yaml` + `configs/training_stage1_hires88.yaml` | hires88 コーパス。`_smoke` はスモーク用 |

- データ生成は独立スクリプトではなく `data_generation*.yaml` 経由で学習に統合されている
  （`scripts/generate_data.py` は存在しない）

## Constraints

- **Teacher policy のデフォルトは raw88**。Bessel teacher は比較ベースライン用のみ
- run ID / 成果物パスに teacher 種別を必ず含める（`docs/stage1_raw_teacher_policy.md`）
- 劣化パスは Bessel IIR 固定（本システム自身がアップサンプラであるため意図的）
- seed は再現性のため固定（標準 1234）。レポートには date + seed + variant を含める
- GPU メモリ ~6GB 想定。chunk 0.25s @ 88.2kHz / batch 32 が推奨バランス

## References

- `docs/stage1_raw_teacher_policy.md` — 命名・保存規約・移行チェックリスト
- `docs/stage1_ringing_loss_ablation.md` — edge/step 損失のアブレーション
- 学習後の評価は `stage1-evaluation` スキル、一気通貫は `stage1-full-workflow` スキル
