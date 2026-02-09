# Stage1 Model Compression Report (Issue #97)

## Summary

This change introduces the Stage1 model-compression baseline:

- Lightweight student architecture (`NMSELight`) with ~5.53M parameters
- Teacher-student distillation training path
- Optional global magnitude pruning path
- Inference-side checkpoint restoration for compressed models

## Parameter Budget

- Teacher (`NMSE` default): 14,225,953 params
- Student (`NMSELight` default): 5,526,761 params
- Reduction: ~61.1%

## Scope Covered in This PR

- Architecture and training infrastructure
- Pruning utility integration
- Checkpoint metadata and inference compatibility
- Unit/integration tests for new paths

## Out of Scope

- Final retraining run for production checkpoint (`stage1_light.pt`)
- Runtime benchmark campaign on RTX 2070S / Jetson Orin Nano
- ABX listening results for teacher vs student

## Repro Commands

```bash
uv run python scripts/train_distillation.py \
  --data-config configs/data_generation.yaml \
  --train-config configs/training_distillation_stage1.yaml \
  --teacher-checkpoint data/checkpoints/stage1_best.pt \
  --student-model nmse_light \
  --checkpoint-dir data/checkpoints/distillation
```

Produced checkpoints:

- `data/checkpoints/distillation/stage1_distill_best.pt`
- `data/checkpoints/distillation/stage1_distill_last.pt`
- `data/checkpoints/distillation/stage1_light.pt` (copy of best)

```bash
uv run python scripts/train_distillation.py \
  --data-config configs/data_generation.yaml \
  --train-config configs/training_distillation_stage1.yaml \
  --teacher-checkpoint data/checkpoints/stage1_best.pt \
  --student-model nmse_light \
  --pruning-ratio 0.15 \
  --checkpoint-dir data/checkpoints/distillation
```
