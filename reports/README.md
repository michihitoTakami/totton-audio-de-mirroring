# CAPB reports

- [`release/`](release/README.md): 推奨採用品（run16、`v5b_sharp1023_midflat70` bank、
  `focused_gentle_fraction 0.3`）の受入証跡
- `abx/`: ABX試聴セッションの`session_meta.json` / `answer_key.json` / `trial_log.csv`（未追跡）

推奨ペア以外の証跡は保存しません。非推奨候補run13〜run15の証跡と1535-tap長FIRの研究比較は
2026-09-04に削除しました（commit `1541917`から復元可能）。ABXの音声素材（`stems/`、`trials/`）は
再生成できるため保存せず、判定ログとseed・checkpoint・ゲイン整合を記録したmetaだけを残します。
