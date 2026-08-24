---
name: stage1-training
description: "Use to train the CAPB Stage 1 controller for 44.1 kHz or 48 kHz input. Trigger: train stage1, CAPB training, 学習実行, Stage1学習, モデル学習."
---

# CAPB Stage 1 Training

Choose the rate family and keep its data/training configs paired:

```bash
uv run python scripts/train_capb.py \
  --data-config configs/data_generation_capb.yaml \
  --config configs/training_stage1_capb.yaml

uv run python scripts/train_capb.py \
  --data-config configs/data_generation_capb_48k.yaml \
  --config configs/training_stage1_capb_48k.yaml
```

Only the controller trains; prototype kernels remain frozen. Keep seed and config in the run record. Training loss is diagnostic and never substitutes for `stage1-evaluation`.
