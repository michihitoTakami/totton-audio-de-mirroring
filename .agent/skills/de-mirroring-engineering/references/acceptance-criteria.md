# Stage 1 Quantitative Acceptance Criteria

Stage 1 の定量受入基準の詳細。`scripts/evaluate_stage1.py` / `scripts/run_issue63_stage1_workflow.py`
の strict ゲートとして実装されている。CLI デフォルト値と一致させること。

## Mirror Suppression & Ringing

| Metric | Threshold | CLI flag |
|--------|-----------|----------|
| `symmetry_reduction_ratio` | >= 0.70 | `--mirror-target-reduction` |
| `hb_energy_cap_violation_rate` | == 0.0 | `--strict-energy-cap` (cap: `--energy-cap 1.0e-3`) |
| `plateau_ripple_rms_after / before` | <= 1.10 | `--max-plateau-ripple-rms-ratio` |
| `plateau_ripple_p2p_after / before` | <= 1.10 | `--max-plateau-ripple-p2p-ratio` |
| `overshoot_abs_after - overshoot_abs_before` | <= 5e-3 | `--max-overshoot-abs-increase` |
| `ringing_ratio_after - ringing_ratio_before` | <= 0.0 | `--strict-ringing-regression` |

## Sweep Probe Gates

| Metric | Threshold | CLI flag |
|--------|-----------|----------|
| mirror band reduction | >= 20.0 dB | `--sweep-min-mirror-band-reduction-db` |
| mirror band after | <= -65.0 dB | `--sweep-max-mirror-band-after-db` |
| hump reduction | >= 18.0 dB | `--sweep-min-hump-reduction-db` |
| hump after | <= -65.0 dB | `--sweep-max-hump-after-db` |
| ridge excess | <= 3.0 dB | `--sweep-max-ridge-excess-db` |

Sweep probe: 20 Hz → 20 kHz, 2.0 s, amplitude 0.5（workflow デフォルト）。

## LB Preservation (structural regression guard)

LB bypass は NMSE forward の band-split 構造（`LB_out = LB_in`）で保証される。
以下は構造破壊を検出する regression guard であり、AI 出力品質の受入基準ではない。

| Metric | Threshold | CLI flag |
|--------|-----------|----------|
| `lb_phase_error_deg` | <= 15.0 | `--max-lb-phase-error-deg` |
| `lb_group_delay_error_samples` | <= 600.0 | `--max-lb-group-delay-error-samples` |
| `lb_amplitude_error_db` | <= -20.0 | `--max-lb-amplitude-error-db` |

## Relaxation Flags（原則禁止）

`--allow-ringing-ratio-increase` / `--allow-energy-cap-violations` /
`--allow-nonpositive-thdn-improvement` は診断・調査目的のみ。
これらを使って通した結果を golden や採用判定に使ってはならない。

## Implementation Pointers

- メトリクス実装: `src/totton_audio_de_mirroring/evaluation/mirror_metrics.py`
- 安全制約: `src/totton_audio_de_mirroring/models/safety_constraints.py`
- テスト: `tests/test_mirror_metrics.py`, `tests/test_safety_constraints.py`
