# Stage1 Regression Golden Update Procedure (Issue #85)

## Purpose

Stage1 の目的（ringing 非劣化 + mirror 抑制）を回帰テストで固定し続けるため、
golden サンプル更新時の手順を明文化する。

## When Update Is Allowed

- 損失設計・ゲート閾値・データ経路など、仕様変更が意図的に行われた場合のみ
- 主観評価（ABX）と客観評価（hard/mirror/ringing/IMD）の両方で改善または同等を確認した場合のみ
- PR に「なぜ golden を更新するのか」を明記する

## Source of Truth

- 入力: `tests/fixtures/golden_samples/stage1/input/*.npy`
- Stage1 出力: `tests/fixtures/golden_samples/stage1/output/*.npy`
- IMD 比較: `tests/fixtures/golden_samples/imd/naive/*.npy` と `imd/nmse/*.npy`
- 回帰基準 JSON: `tests/fixtures/golden_samples/regression_baseline.json`

## Update Steps

1. 候補チェックポイントを同一条件で評価する。

```bash
uv run python scripts/run_issue63_stage1_workflow.py \
  --data-config configs/data_generation.yaml \
  --train-config configs/training_stage1.yaml \
  --eval-input-dir tests/fixtures/golden_samples/stage1/input \
  --imd-naive-dir tests/fixtures/golden_samples/imd/naive \
  --checkpoint-dir data/checkpoints \
  --report-dir reports/issue85 \
  --seed 1234 \
  --device cuda \
  --energy-cap 1e-3 \
  --skip-training \
  --candidate-checkpoints stage1_best.pt stage1_last.pt stage1_emergency.pt
```

2. 選定結果を確認する。
   - `reports/issue85/selected/selection_report.json`
   - ringing/mirror/hard gate がすべて pass していること

3. golden 出力を更新する。
   - `reports/issue85/candidate_outputs/<selected>/` を `tests/fixtures/golden_samples/stage1/output/` に反映
   - 同じ出力を `tests/fixtures/golden_samples/imd/nmse/` に反映

4. baseline JSON と関連メタデータを更新する。
   - `tests/fixtures/golden_samples/regression_baseline.json`
   - `tests/fixtures/golden_samples/issue64_model_selection.json`（または後継ファイル）
   - `tests/fixtures/golden_samples/abx_pairs.json`

## Required Validation

golden 更新コミットでは、最低限以下を実行して pass させる。

```bash
uv run pytest tests/regression/test_stage1_regression.py -v
uv run pytest tests/test_evaluate_stage1_script.py tests/test_issue63_workflow_script.py -v
uv run pytest tests/evaluation/test_time_domain_visualization.py -v
uv run pytest -m "not slow and not gpu" -v
```

## Review Checklist

- `symmetry_reduction_ratio >= 0.70` を満たしている
- `hb_energy_cap_violation_rate == 0.0` を満たしている
- ringing gate（RMS/P2P/overshoot/ringing-ratio）が pass
- 変更理由と期待効果が PR 本文に記載されている
