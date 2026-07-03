---
name: regression-golden-update
description: "Use when updating Stage 1 regression golden samples/baselines — strict procedure requiring ABX evidence and objective improvement. Trigger: golden update, regression baseline, ゴールデン更新, リグレッション基準更新."
---

# Regression Golden Update

Stage 1 のリグレッション基準（golden サンプル / ベースラインメトリクス）を更新する際の厳格な手順。安易な更新は品質退行の見逃しに直結するため、証跡が必須。

## Preconditions（全て満たすこと）

1. **客観指標の改善**: `stage1-evaluation` の受入基準を全て満たし、かつ現行 golden 比で主要メトリクスが改善していること
2. **主観評価の裏付け**: `abx-protocol` スキルに従った ABX セッションで劣化がないこと
3. **PR での正当化**: なぜ golden を更新するのか、上記 1–2 の証跡（レポートパス・ABX サマリ）を PR 本文に記載

## Workflow

```bash
# 1. 新候補のフル評価（strict ゲート全部入り）
uv run python scripts/evaluate_stage1.py \
  --input-dir <...> --output-dir <...> \
  --report-dir reports/stage1/raw88/<run_id> \
  --strict-energy-cap --strict-mirror-reduction \
  --strict-ringing-regression --strict-lowband-preservation --strict-imd-proxy

# 2. 現行 golden とのメトリクス比較表を作成し、レポートに保存

# 3. ABX セッション実施（docs/templates/ のテンプレートで記録）

# 4. golden 差し替え + テスト更新を feature ブランチでコミットし PR
```

## Prohibited

- ABX なしでの golden 更新
- ゲート緩和フラグ（`--allow-*`）を使って通した結果での golden 更新
- 「テストを通すため」だけの閾値変更（閾値変更は設計判断として別 Issue で議論）

## References

- `docs/stage1_regression_golden_update.md` — 手順の原典 (#85)
- `docs/abx_listening_protocol.md` / `abx-protocol` スキル
