---
name: audio-visualization
description: "Use to visualize audio quality: spectrograms, mirror patterns, impulse/step responses, checkpoint comparisons. Trigger: visualize, spectrogram, impulse response, plot, 可視化, スペクトログラム, 波形確認."
---

# Audio Visualization

Stage 1/2 の出力品質を可視化する。ミラーパターン・インパルス/ステップ応答・チェックポイント比較の定番プロットを生成する。

## Execution Steps

```bash
# 1. 入出力ペアの品質可視化（スペクトログラム等、npy ディレクトリ単位）
uv run python scripts/visualize_audio_quality.py \
  --input-dir <入力npyディレクトリ> --output-dir <出力npyディレクトリ> \
  --visual-dir reports/<...>/visuals \
  --sample-rate 88200 --cutoff-hz 20000 --limit 16 \
  --summary-json reports/<...>/visual_summary.json

# 2. インパルス / ステップ応答（リンギング・オーバーシュート確認の要）
uv run python scripts/visualize_impulse_step_response.py \
  --checkpoint data/checkpoints/<best>.pth \
  --data-config configs/data_generation.yaml \
  --output-dir reports/impulse_step_check_<YYYYMMDD> \
  --device cpu

# 3. 3-way 比較（Stage1 vs 蒸留 vs ベースライン、sweep/square プローブ）
uv run python scripts/visualize_threeway_stage1_probes.py \
  --stage1-checkpoint <ckpt> --distill-checkpoint <ckpt> \
  --output-dir reports/<...> --device cpu

# 4. その他
uv run python scripts/visualize_checkpoint_performance.py   # チェックポイント性能推移
uv run python scripts/visualize_square_wave_kaiser.py       # 矩形波 + Kaiser 比較
uv run python scripts/plot_comparison.py                    # 汎用比較プロット
uv run python scripts/plot_sweep_zoomed.py                  # sweep 拡大表示
```

## What to Look For

- **スペクトログラム**: 22.05kHz 折返し軸に対する鏡像パターンが消えているか。20kHz 以下に変化がないか
- **ステップ応答**: プラトー部リップル増加なし、オーバーシュート増加 ≤ 5e-3
- **インパルス応答**: プリエコー/ポストリンギングが参照 SRC より悪化していないか
- **sweep**: ミラーバンドのリッジが -65dB 以下へ落ちているか

## References

- 定量判定は `stage1-evaluation` スキル（可視化は定性確認の補助）
- 出力先は `reports/` 配下に日付付きディレクトリで保存する
