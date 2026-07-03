---
name: stage1-evaluation
description: "Use to evaluate Stage 1 outputs against hard metrics (mirror suppression, ringing, energy cap, LB preservation, IMD proxy). Trigger: evaluate stage1, metrics, acceptance check, 評価実行, メトリクス, 受入基準チェック."
---

# Stage 1 Evaluation

Stage 1 出力のハードメトリクス評価（ミラー抑制・リンギング・エネルギーキャップ・LB保存・IMDプロキシ）を実行し、受入基準に対する合否を判定する。

## Execution Steps

```bash
uv run python scripts/evaluate_stage1.py \
  --input-dir <NMSE入力npyディレクトリ> \
  --output-dir <NMSE出力npyディレクトリ> \
  --report-dir reports/stage1/raw88/stage1_raw88_nmse_<variant>_<YYYYMMDD>_s<seed> \
  --json <report>/metrics.json --csv <report>/metrics.csv \
  --strict-energy-cap --strict-mirror-reduction \
  --strict-ringing-regression --strict-lowband-preservation

# 可視化付き: --mirror-visual-dir <dir> --mirror-visual-limit 16
# リンギング詳細: --ringing-json <path> --ringing-csv <path>
```

デフォルトパラメータ: `--sample-rate 88200 --cutoff-hz 20000 --energy-cap 1.0e-3 --glob "*.npy"`

## Acceptance Criteria (Stage 1 定量受入基準)

ミラー抑制 & リンギング:

- `symmetry_reduction_ratio >= 0.70`（`--mirror-target-reduction`）
- `hb_energy_cap_violation_rate == 0.0`
- `plateau_ripple_rms_after / before <= 1.10`
- `plateau_ripple_p2p_after / before <= 1.10`
- `overshoot_abs_after - overshoot_abs_before <= 5e-3`
- `ringing_ratio_after - ringing_ratio_before <= 0.0`
- sweep gates: mirror band reduction ≥ 20dB / after ≤ -65dB / hump reduction ≥ 18dB / ridge excess ≤ 3dB

LB 保存（構造バイパスの regression guard、AI 品質基準ではない）:

- `lb_phase_error_deg <= 15.0`
- `lb_group_delay_error_samples <= 600.0`
- `lb_amplitude_error_db <= -20.0`

## Report Naming Convention

```
reports/stage1/<teacher>/stage1_<teacher>_nmse_<variant>_<YYYYMMDD>_s<seed>[_suffix]
# 例: reports/stage1/raw88/stage1_raw88_nmse_20260211_s1234_bs16
```

- `raw88` と `bessel` のレポートはディレクトリを分離し、混在させない

## References

- `docs/stage1_raw_teacher_policy.md`
- `src/totton_audio_de_mirroring/evaluation/mirror_metrics.py` — メトリクス実装
- 一気通貫（学習込み）は `stage1-full-workflow` スキル
