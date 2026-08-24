# CAPB ABX Listening Protocol

## Purpose

CAPB出力と固定比較経路を、音量差や提示順のbiasを避けて比較する。
ABXはprobe gateの代替ではなく、gate通過後の知覚差を記録する補助評価とする。

## Assets

各sessionで次を固定し、`session_meta.json`へ記録する。

- 入力音源とそのSHA-256
- CAPB checkpointとそのSHA-256
- 比較backend（通常はBessel reference）
- commit hash、sample rate、再生chain

## Loudness matching

1. CAPBとreferenceを同一区間で測定する。
2. integrated loudness、利用できなければRMSを使う。
3. level差を±0.1 dB以内へ合わせる。
4. 適用gainをtrial logへ保存する。

## Procedure

1. `A`と`B`へ`reference`/`capb`をrandomに割り当てる。
2. `X`を`A`または`B`からrandomに選ぶ。
3. listenerはA/B/Xを自由に再生し、`X == A`または`X == B`を回答する。
4. confidence、harshness、attack、fatigueを記録する。
5. 1 sampleにつき10 trial以上、session全体で20 trial以上を推奨する。

片側binomial test（帰無仮説`p = 0.5`、`p < 0.05`）を用いる。統計的に識別できても、attack低下やmetallic artifactの増加がnoteに現れた場合は採用根拠にしない。

## Records

テンプレート:

- `docs/templates/abx_trial_log_template.csv`
- `docs/templates/abx_session_summary_template.md`

保存先:

```text
reports/abx/<session_id>/
  trial_log.csv
  summary.md
  session_meta.json
```

同じtripletを再生成できるよう、生成command、config、checkpoint hash、適用gainを必ず残す。
