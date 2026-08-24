---
name: stage1-evaluation
description: "Use to evaluate CAPB checkpoints against canonical and held-out worst-case probe gates. Trigger: evaluate stage1, CAPB metrics, acceptance check, 評価実行, 受入基準チェック."
---

# CAPB Stage 1 Evaluation

Evaluate both families:

```bash
uv run python scripts/evaluate_probe_gates.py \
  --backend capb --checkpoint <44k1-checkpoint> --rate-family 44k1

uv run python scripts/evaluate_probe_gates.py \
  --backend capb --checkpoint <48k-checkpoint> --rate-family 48k
```

Do not use `--no-strict` for release evidence. Preserve `gate_report.json`, `gate_report.md`, gate spec version, probe manifest hash, and worst binding probe.
