---
name: de-mirroring-engineering
description: "Use to plan/review any change touching the de-mirroring architecture: 2-stage hybrid design, band-split LB bypass, suppression mask, safety constraints, hard requirements. Trigger: architecture, design decision, mirror suppression policy, 設計方針, アーキテクチャ, ミラー抑制の考え方."
---

# De-Mirroring Engineering

本リポジトリの設計原則。アーキテクチャ・損失・安全制約に触れる変更は、実装前にこのスキルで物理的妥当性を確認する。

## System Overview

Hybrid Neural SR (HNSR): 44.1kHz PCM → **Stage 1** NMSE（Neural, 2×, →88.2kHz）→
**Stage 2** HIE（DSP, 8×, →705.6kHz）。目的は**ミラー/エイリアシング除去と時間応答保存**であり、超音波帯域の生成ではない。ターゲット: Jetson Orin Nano (8GB)、非リアルタイム可。

## Hard Requirements（違反 = 失敗）

1. **0–20kHz は入力と同一**（波形・位相・群遅延。band-split 構造 `LB_out = LB_in` で保証）
2. **ミラーパターン抑制**（可聴な digital harshness の除去）
3. **20–44kHz はゼロ近傍で可**（倍音の強制生成禁止）
4. **高域総エネルギーキャップ常時適用**（IMD 安全性）
5. **矩形波プローブでリンギング非退行**（参照 SRC 比）

## Core Design Decisions

- **Band-split + LB bypass**: `LB_out = LB_in` は損失ではなく構造で保証。AI は HB のみ処理
- **Suppression mask 出力**: `HB_out = HB_in ⊙ M`, `M ∈ [0,1]`。生成ではなく抑制
- **固定 safety constraints（後処理）**: energy cap → envelope target → HB→LB リーク対策 HPF
  （`src/totton_audio_de_mirroring/models/safety_constraints.py`）
- **Teacher policy**: raw88 がデフォルト。Bessel teacher は比較ベースラインのみ
- **劣化パス**: Bessel IIR 固定（本システム自身がアップサンプラのため意図的）

## Anti-Patterns（禁止）

- Nyquist 超の周波数成分生成
- 0–20kHz の内容改変（bypass 構造を迂回する変更を含む）
- energy cap なしの高域エネルギー使用
- ミラー検出機構なしでの学習
- 物理検証（STFT でのミラー低減測定）を省いた「改善」主張

## Review Questions（変更前に自問する）

1. 信号処理理論として妥当か（Physical Basis を docstring に書けるか）
2. 可聴帯域 0–20kHz を触っていないか
3. 実際にミラー/エイリアシングを減らすか（定量測定可能か）
4. 高域エネルギーはキャップされているか（IMD リスク）
5. 受入基準（`references/acceptance-criteria.md`）で測れるか

## References

- `references/acceptance-criteria.md` — Stage 1 定量受入基準の詳細
- `CLAUDE.md` — Project Context / Key Design Decisions
- `docs/stage1_raw_teacher_policy.md`, `docs/stage1_stage2_pipeline_integration.md`
