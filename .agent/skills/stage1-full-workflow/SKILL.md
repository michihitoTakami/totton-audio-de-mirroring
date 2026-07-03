---
name: stage1-full-workflow
description: "Use to run the end-to-end Stage 1 workflow: retrain, evaluate with metric gates, and select the best checkpoint. Trigger: full workflow, retrain and select, issue63 workflow, 一気通貫, 再学習と選定, フルワークフロー."
---

# Stage 1 Full Workflow (retrain → evaluate → select)

Stage 1 の再学習 → メトリクスゲート付き評価 → ベストチェックポイント選定を一気通貫で実行する正準ドライバ。

## Execution Steps

```bash
uv run python scripts/run_issue63_stage1_workflow.py \
  --data-config configs/data_generation.yaml \
  --train-config configs/training_stage1.yaml \
  --eval-input-dir <評価入力npyディレクトリ> \
  --imd-naive-dir <IMD比較用naive出力ディレクトリ> \
  --report-root-dir reports/stage1 \
  --checkpoint-root-dir data/checkpoints/stage1 \
  --device cuda --seed 1234

# 学習をスキップして既存チェックポイントの評価・選定のみ:
#   --skip-training --candidate-checkpoints <ckpt...>
# run ID / teacher タグの明示: --run-id <id> --teacher-tag <raw88|bessel>
```

## Metric Gates（デフォルト値、必要時のみ緩和）

- `--mirror-target-reduction 0.70`
- `--sweep-min-mirror-band-reduction-db 20.0` / `--sweep-max-mirror-band-after-db -65.0`
- `--sweep-min-hump-reduction-db 18.0` / `--sweep-max-hump-after-db -65.0`
- `--sweep-max-ridge-excess-db 3.0`
- LB guards: `--max-lb-phase-error-deg 15.0` / `--max-lb-group-delay-error-samples 600.0` / `--max-lb-amplitude-error-db -20.0`
- リンギング: `--max-plateau-ripple-rms-ratio 1.10` / `--max-plateau-ripple-p2p-ratio 1.10` / `--max-overshoot-abs-increase 5e-3`
- 緩和フラグ（原則使わない。使う場合は PR で正当化）: `--allow-ringing-ratio-increase`,
  `--allow-energy-cap-violations`, `--allow-nonpositive-thdn-improvement`

## Constraints

- レポートは `--report-root-dir` 配下に teacher 別・日付・seed 入りで生成される
- 長時間ジョブになるため、バックグラウンド実行 + ログ tee を推奨
- ゲート不合格のチェックポイントを手動で採用しない（緩和フラグの濫用禁止）

## References

- `scripts/run_issue63_stage1_workflow.py --help` — 全オプション
- `stage1-training` / `stage1-evaluation` スキル — 個別ステップの詳細
