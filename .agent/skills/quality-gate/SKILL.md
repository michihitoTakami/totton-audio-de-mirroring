---
name: quality-gate
description: "Use to run the full lint / format / type-check / test quality gate before commit or push. Trigger: lint, format, type check, run tests, quality gate, テスト実行, 型チェック, 品質チェック."
---

# Quality Gate

コミット・プッシュ前に必須の品質チェック一式を実行する。pre-commit / pre-push hook と同じ内容を手動で先回りして確認する。

## Execution Steps

```bash
# 1. フォーマット
uv run ruff format src/ tests/ scripts/

# 2. Lint（自動修正付き）
uv run ruff check src/ tests/ scripts/ --fix

# 3. 型チェック
uv run mypy src/

# 4. テスト（slow / gpu を除外 — pre-push hook と同一）
uv run --extra dev pytest -m "not slow and not gpu" --tb=short

# 5. 全 hook をまとめて確認したい場合
uv run pre-commit run --all-files
```

## Decision Rules

- いずれかが失敗した場合、**修正してから** commit / push する。`--no-verify` は禁止
- カバレッジ確認: `uv run pytest --cov=totton_audio_de_mirroring --cov-report=term`
  （目標: 全体 80%、filters/データローダ等クリティカルパス 90%+、scripts 50%）
- GPU 依存テストは `-m gpu`、重いテストは `-m slow` で個別実行
- C++ (`cpp/`) を触った場合は pre-push hook で clang-tidy と cmake build + ctest も走る

## Constraints

- pytest markers は `slow` / `gpu` のみ（`--strict-markers` 有効）
- mypy は `disallow_untyped_defs` — 全関数に型ヒント必須
- `uv` が壊れている環境では実体バイナリを直接使う（過去事例あり）

## References

- `.agent/rules/coding-style.md`, `.agent/rules/testing.md`, `.agent/rules/security.md`
- `.pre-commit-config.yaml`
