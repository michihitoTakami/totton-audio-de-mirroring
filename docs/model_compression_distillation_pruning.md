# Stage1 Model Compression: Distillation + Pruning (Issue #97)

## Purpose

Stage1 NMSE を 14.2M params から 5-7M params へ削減し、推論速度/メモリ効率を改善する。

## Implemented Components

- `src/totton_audio_de_mirroring/models/nmse_light.py`
  - `NMSELightConfig` (default: base_channels=40, num_downsamples=3)
  - `NMSELight` (default params: 約5.53M)
- `src/totton_audio_de_mirroring/training/distillation.py`
  - `DistillationConfig`
  - `train_stage1_distillation(...)`
  - `apply_global_magnitude_pruning(...)`
- `scripts/train_distillation.py`
  - Teacher checkpoint を読み込み
  - teacher checkpoint の `training_config.teacher_type` と
    distillation 設定を照合（不一致はエラー）
  - Student を `nmse_light` もしくは `nmse` で構築
  - 蒸留学習 + 任意の magnitude pruning
  - teacher tag を含む checkpoint を出力
    - `stage1_distill_raw88_best.pt` / `stage1_distill_raw88_last.pt`
    - `stage1_distill_bessel_best.pt` / `stage1_distill_bessel_last.pt`
  - best checkpoint を `stage1_light_<teacher>.pt` として出力
    - 互換のため `stage1_light.pt` も同時出力
- `configs/training_distillation_stage1.yaml`
  - distillation 用学習設定

## Checkpoint Compatibility

`model_config.model_type=nmse_light` を checkpoint に保存し、
`load_nmse_stage1_processor(...)` が軽量モデルを自動復元する。

## Default Training Command

```bash
uv run python scripts/train_distillation.py \
  --data-config configs/data_generation.yaml \
  --train-config configs/training_distillation_stage1.yaml \
  --teacher-checkpoint data/checkpoints/stage1_best.pt \
  --student-model nmse_light \
  --checkpoint-dir data/checkpoints/distillation
```

Pruning を併用する場合:

```bash
uv run python scripts/train_distillation.py \
  --teacher-checkpoint data/checkpoints/stage1_best.pt \
  --student-model nmse_light \
  --pruning-ratio 0.15
```

## Raw vs Bessel Comparison Report

Issue #107 の比較実験では、Issue #63 workflow の出力ランを使って
raw88 と bessel の勝敗表を自動生成できる。

```bash
uv run python scripts/report_raw_teacher_comparison.py \
  --raw-run-dir reports/issue63/raw88/stage1_raw88_nmse_YYYYMMDD_s1234 \
  --bessel-run-dir reports/issue63/bessel/stage1_bessel_nmse_YYYYMMDD_s1234 \
  --output-md reports/teacher_comparison/raw88_vs_bessel.md \
  --output-csv reports/teacher_comparison/raw88_vs_bessel.csv
```

## Acceptance Mapping

- Param budget (5-7M): `NMSELight` defaultで達成（約5.53M）
- Distillation path: `train_stage1_distillation(...)`
- Pruning path: `apply_global_magnitude_pruning(...)`
- Inference compatibility: pipeline loaderで`nmse_light`復元対応
