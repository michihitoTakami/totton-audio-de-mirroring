# Stage1 Ringing-Loss Ablation (Issue #81)

## Purpose

Stage1 の mirror 抑制性能を維持したまま、エッジ近傍リンギング悪化を学習段階で抑制できるかを検証する。

## Added Training Objective

- Existing: `mask`, `stft`, `preserve`, `energy`
- New auxiliary terms:
  - `edge`: ターゲット微分に対するエッジ重み付き微分一致
  - `step`: エッジ近傍の step-response 一致

`configs/training_stage1.yaml` では小係数で導入:

- `loss_weights.edge = 0.05`
- `loss_weights.step = 0.05`

## Non-Interference with 0-20kHz Requirement

- 0-20kHz は NMSE の band-split 構造 (`LB_out = LB_in`) で不変保証。
- 補助lossは `forward_highband()` の HB 学習にのみ適用され、LB へ直接作用しない。
- 低域非干渉は従来どおり評価指標 (`lb_phase_error_deg`, `lb_group_delay_error_samples`) で確認する。

## Loss Contribution Analysis

学習ログ (`EpochMetrics`) に以下の寄与率が追加される:

- `contrib_mask`, `contrib_stft`, `contrib_preserve`, `contrib_energy`
- `contrib_edge`, `contrib_step`

寄与率は weighted loss の正規化比で、1.0 に和が揃う。

## Ablation Report Workflow

1. Baseline 設定と ringing-loss 設定で Stage1 を学習・評価する。
2. `scripts/evaluate_stage1.py --json ...` で両者の JSON を保存する。
3. 必要に応じて両チェックポイントも指定して、以下を実行する:

```bash
uv run python scripts/report_stage1_ringing_ablation.py \
  --baseline-eval-json reports/issue81/baseline_eval.json \
  --ringing-eval-json reports/issue81/ringing_eval.json \
  --baseline-checkpoint data/checkpoints/issue81/baseline/stage1_best.pt \
  --ringing-checkpoint data/checkpoints/issue81/ringing/stage1_best.pt \
  --output-md reports/issue81/ablation_report.md
```

出力 Markdown には以下が含まれる:

- mirror 維持判定
- ringing 改善判定
- 低域非干渉ゲート判定
- energy cap violation 判定
- 評価メトリクス比較表
- （チェックポイント指定時）loss 寄与率比較表
