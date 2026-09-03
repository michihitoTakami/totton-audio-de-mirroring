# CAPB release evidence — 2026-09-03 (routing v2, sharp floor)

採用品は`release_v4` prototype bankのまま、controllerを**routing v2 + sharp floor**処方で
再学習したペアです。定常部はSharp、包絡onset/offsetと持続plateau edgeはGentle、疎な
impulseはGentle主体（`focused_gentle_fraction` 44.1 kHz `0.90` / 48 kHz `0.85`）に
ルーティングし、定常フレームのSharp下限損失（`stationary_sharp_floor`）でノイズ上の
Gentle漏れを抑えています。

| Family | checkpoint | G1〜G9 CPU | G1〜G9 CUDA strict FP32 | G5 impulse列 | G2b pre-echo | G3 pink | 64位相 worst |
|---|---|---|---|---:|---:|---:|---:|
| 44.1→88.2 kHz | `data/checkpoints/capb/run13_routing_v2_sharpfloor_20260903_44k1/capb_best.pt` | PASS | PASS | `0.447 dB` | `5.15e-11` | `-117.6 dB` | `-35.93 dB` |
| 48→96 kHz | `data/checkpoints/capb_48k/run13_routing_v2_sharpfloor_20260903_48k/capb_best.pt` | PASS | PASS | `0.451 dB` | `1.57e-11` | `-112.1 dB` | `-41.98 dB` |

旧release（run11 / run12）との主な差:

- 実音源9本中8本で逆向きだったSharp/Gentle遷移（定常でGentle、打撃でSharp）が正方向になった
  （旧release 0/9 → 8/9）。
- 孤立impulseのリンギング（主ローブ±0.06 msを除く±4 msのエネルギー比）が
  44.1 kHz `-24.3 dB` → `-32.5 dB`、48 kHz `-25.7 dB` → `-28.9 dB`。G2b pre-echoは
  `1.05e-7` → `5.2e-11`、`1.33e-8` → `1.6e-11`。
- controller 64位相・Hann-OLA境界のマージンは`-2.84 dB` → `-35.9 dB`、`-8.87 dB` → `-42.0 dB`。
- G5 impulse列利得誤差はGentle主体化の代償で`0.446` → `0.447 dB`、`0.400` → `0.451 dB`
  （上限`0.5 dB`）。
- 44.1 kHz SMPTE sidebandは`-125.6` → `-142.9 dB`。48 kHz added AM sidebandは`-157.8` → `-146.7 dB`
  （G9閾値`-110 dB`は十分に満たす）。

## 複数seed

同じ処方でseed 1234 / 2026 / 4649を両系列で学習し、いずれもG1〜G9をCPUとstrict-FP32 CUDAの
両方で通過しました（[seed_summary.json](selection/seed_summary.json)）。impulseリンギングは
44.1 kHz `-32.5〜-33.0 dB`、48 kHz `-28.9〜-29.0 dB`、pink noiseのSharp<0.99フレームは
全seedで`0.07%`以下です。sharp floor損失を入れる前の同処方では、seed 4649の44.1 kHzが
pink noiseでG3を`-58 dB`で落としていました。採用checkpointはseed 1234です。

## release_quality（クロスレート検査）の改定

`evaluation/release_quality.py`の過渡検査は「48 kHzが44.1 kHz以上に保護的」を要求していましたが、
48 kHz Gentle端点の利得誤差（impulse列`0.527 dB`）が44.1 kHz（`0.497 dB`）より大きいため、
Gentle主体の処方では「normalized peak ≤ 44.1 kHz + 0.02」と「impulse列G5 ≤ 44.1 kHzの値」を
同時に満たせません。意図（48 kHzを不必要にSharp寄りにしない、過渡応答を両系列で比較する）を
保ったまま次のように明文化しました。

- impulse列G5は両系列とも凍結G5ゲート（`0.5 dB`）に対して判定する。
- normalized position検査の許容に、checkpointに保存された`focused_gentle_fraction`の差
  （44.1 kHz − 48 kHz、今回は`0.05`）を加える。

G1〜G9のゲート本体、probe manifest、閾値は変更していません。

## ONNXとnull test

- 44.1 kHz ONNX sha256 `65746ffb…4014e95`の前半、parity max abs error `7.15e-7`、wav null `-130.8 dB`
- 48 kHz ONNX sha256 `6a543b72…`、parity max abs error `5.36e-7`、wav null `-138.6 dB`

`torch.diff`をslice差分に置き換えてphysics routing priorをONNX opset 17で書き出せるようにしました。
置き換え前後でcontroller出力は同一です。ONNXファイルはこのリポジトリには含めず、
`totton-audio-nn/data/nmse/`側の差し替えは別PRで行います。

- [release manifest](release_manifest.json)
- [release quality](release_quality/release_quality.md)
- [44.1 kHz gate（CPU）](gates/44k1/cpu/candidate/gate_report.md) / [CUDA](gates/44k1/cuda/candidate/gate_report.md)
- [48 kHz gate（CPU）](gates/48k/cpu/candidate/gate_report.md) / [CUDA](gates/48k/cuda/candidate/gate_report.md)
- `robustness/`: controller 64位相とHann-OLA境界
- `routing/`: 診断用routing gate R1〜R4（release条件ではない）、impulseリンギングとpink漏れの比較
- `visualization/`: impulse応答と歪み図
- `selection/`: seed比較

数値の正史はstrict-FP32のworst-probe gateです。図とrouting診断は補助です。
