---
name: stage1-full-workflow
description: "Use for the complete CAPB Stage 1 train-and-gate workflow across both supported rate families. Trigger: full workflow, retrain and select, 一気通貫, 再学習と選定."
---

# CAPB Full Workflow

1. Train 44.1 kHz and 48 kHz controllers with `stage1-training`.
2. Evaluate each best checkpoint with `stage1-evaluation` using canonical and held-out probes.
3. Select only checkpoints whose reports have `all_passed=true`.
4. Record checkpoint hashes, config paths, seeds, gate spec version, manifest hashes, and binding probes.

There is no aggregate driver that may bypass a failing gate. Never select from training loss alone.
