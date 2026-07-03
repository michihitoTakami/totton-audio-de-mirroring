---
name: abx-protocol
description: "Use to run or document ABX subjective listening tests with frozen triplets and repo templates. Trigger: ABX, listening test, subjective evaluation, 試聴テスト, 主観評価, リスニング."
---

# ABX Listening Protocol

再現可能な ABX 主観評価を実施・記録する。golden 更新や方式比較の主観的裏付けに使う。

## Workflow

1. **素材準備**: 比較対象（A: 参照 SRC / B: NMSE 出力など）の frozen triplet を用意する。
   `pipeline-inference` スキルでレンダリングし、レベルマッチ（ラウドネス整合）を確認する
2. **セッション実施**: `docs/abx_listening_protocol.md` の手順に従う
   （試行数・ランダム化・休憩・合否判定の統計基準が規定されている）
3. **記録**: テンプレートに記入する
   - 試行ログ: `docs/templates/abx_trial_log_template.csv` をコピーして使用
   - セッションサマリ: `docs/templates/abx_session_summary_template.md` をコピーして使用
4. **保存**: 結果は対応するレポートディレクトリ（`reports/<...>/abx/`）に配置し、
   関連 Issue / PR からパスを参照する

## Constraints

- 素材の triplet は凍結する（セッション途中での差し替え禁止）
- レベル差は ABX の交絡因子になるため、事前にラウドネスを整合させる
- 有意性の判定は protocol 記載の統計基準に従う（恣意的な打ち切り禁止）
- ヘッドホン/DAC 等の再生環境をサマリに必ず記録する

## References

- `docs/abx_listening_protocol.md` — プロトコル原典 (#59/#64)
- `docs/templates/abx_session_summary_template.md` / `docs/templates/abx_trial_log_template.csv`
- golden 更新時の必須手順: `regression-golden-update` スキル
