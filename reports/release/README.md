# CAPB release evidence — 2026-09-03（run16）

現在の推奨は両rate familyとも**run16**（`v5b_sharp1023_midflat70` bank、
`focused_gentle_fraction 0.3`、gate spec 6）です。`release_manifest.json`の`recommended`が正史です。

このディレクトリには推奨ペアの受入証跡だけを置きます。2026-09-04に、非推奨となった
run13〜run15の証跡バンドルと1535-tap研究比較（`reports/research/long_fir_1535/`）を削除しました。
必要な場合はcommit `1541917`から復元できます。

## run16 — `run16_v5b_midflat_g03_20260903/`

controllerはrouting v2 + sharp floor処方で学習し、`v5b_sharp1023_midflat70` bank上で
3 seed（1234 / 2026 / 4649）のcontroller-only fine-tuneを行い、seed 1234を採用しました。

| Family | checkpoint | G1〜G9 CPU / CUDA（spec 6） | G5 impulse列 | G2b | G3最悪 | 64位相 worst | ONNX parity / null |
|---|---|---|---:|---:|---:|---:|---|
| 44.1k | `data/checkpoints/capb/run16_v5b_midflat_g03_20260903_44k1/capb_best.pt` | PASS / PASS | `0.164 dB` | `4.6e-11` | `-133.1 dB` | `-37.3 dB` | `1.0e-6` / `-128.2 dB` |
| 48k | `data/checkpoints/capb_48k/run16_v5b_midflat_g03_20260903_48k/capb_best.pt` | PASS / PASS | `0.175 dB` | `3.3e-11` | `-133.5 dB` | `-38.4 dB` | `1.1e-6` / `-128.2 dB` |

3 seedすべてが両系列でCPU・strict-FP32 CUDAのG1〜G9（spec 6）を通過し、G5は`0.16〜0.18 dB`、
pink noiseのSharp<0.99フレームは全seedで0%、64位相worstは`-37〜-38 dB`です。最悪行は2 kHz矩形の
plateau_rms（余裕4.7% / 6.3%）
（[selection/seed_summary.json](run16_v5b_midflat_g03_20260903/selection/seed_summary.json)）。

null testの入力probeは2 s log sweep + 低レベルノイズ + 3クリック（16 bit）で、SHA-256は
各`null_test/*.json`に記録しています。

### bankと打撃時Gentle比率を選んだ理由

bankはSharp 1023 taps（Kaiser 140 dB）、Midは20 kHzまでフラットな短いKaiser（阻止24 kHz、70 dB、
約100 taps）、GentleはBessel6@20 kHzです。

ABX（ハイハット素材）でGentle単体は識別可、Sharp単体とCAPBは識別不能でした。Gentleは15 kHzで
`-4.3 dB`、20 kHzで`-10 dB`落ちるので、打撃の瞬間にGentleへ寄るほど打撃の高域が削れます。
Midをフラットな短いKaiserに置き換えたうえで、打撃時のGentle比率を`0.9` → `0.6` → `0.3`と下げ、
残りをflat Midで受ける構成に到達しました。

比率0.0はMidのGibbsリップルが5 kHz矩形でBessel参照の1.1倍を超えるためG2で不合格（物理的に正しい
判定）で、`0.3`が両系列共通で通る下限です。Gentle自体を20 kHzまでフラットにする案はG7
（Bessel参照に対するイメージ帯の増加）を`+12 dB`超過して不合格でした。

| 指標（比率0.6 → 0.3） | 44.1 kHz | 48 kHz |
|---|---|---|
| G5 impulse列利得誤差 | `0.294` → `0.164 dB` | `0.311` → `0.175 dB` |
| G2b pre-echo | `1.7e-11` → `4.6e-11` | `1.2e-11` → `3.3e-11` |
| 孤立impulseリンギング | `-23.4` → `-19.7 dB` | `-21.9` → `-18.5 dB` |
| ハイハット打撃16–20 kHz（Sharp比） | `-3.8` → `-3.2 dB` | — |
| 64位相 worst | `-32.8` → `-37.3 dB` | `-42.6` → `-38.4 dB` |
| controller@impulse | Gentle 0.58 / Mid 0.42 → Gentle 0.29 / Mid 0.70 | Gentle 0.58 / Mid 0.41 → Gentle 0.30 / Mid 0.70 |

代償は孤立impulseのリンギングが約4 dB戻ること（Sharp単体`-15 dB`より4〜5 dB低い水準は維持）。
打撃時のGentle使用は比率に依存しない包絡onset経路（level_risk → Gentle）が残るため、ハイハットの
16–20 kHzはSharp比`-3.2 dB`が床です。この床を下げるにはonset経路の保護先を変える別のknobが必要です。

比率の走査値は[selection/fraction_sweep_summary.json](run16_v5b_midflat_g03_20260903/selection/fraction_sweep_summary.json)、
打撃時の帯域別measurementは[selection/hit_metrics.json](run16_v5b_midflat_g03_20260903/selection/hit_metrics.json)にあります。

### gate spec 6（2026-09-03）

矩形波 / DC stepのリンギング指標（G1/G2）は、出力とBessel参照を8倍sincオーバーサンプルしてから
エッジ整列と0.1–0.8 msの窓切りを行います。閾値の定義（Bessel参照比1.1倍、overshoot +0.005）は
不変です。spec 5では出力サンプル格子でエッジを検出していたため、48 kHzの2 kHz / 5 kHz矩形
（エッジが96 kHzサンプルの中間に揃う）で窓が1サンプル飛び、5 kHz矩形のovershoot閾値が
44.1 kHz `0.549`に対して48 kHz `0.283`と2倍違っていました。spec 6では`0.539` / `0.567`と一致します。

## タップ数の走査

controllerを学習なしで各bankへ載せ替え、483〜4095 tapsを走査しました
（[selection/long_fir_sweep_summary.json](run16_v5b_midflat_g03_20260903/selection/long_fir_sweep_summary.json)、
構造検査は[selection/long_fir_structural.json](run16_v5b_midflat_g03_20260903/selection/long_fir_structural.json)）。

| Sharp taps | 44.1k G3最悪 | 44.1k 64位相 | 44.1k G9 | 48k G3最悪 | 48k 64位相 | 48k G9 |
|---:|---:|---:|---:|---:|---:|---:|
| 483 / 277（release_v4） | `-116.1` | `-35.9` | `-142.9` | `-107.8` | `-42.0` | `-143.9` |
| 1023 a140 | `-117.9` | `-35.8` | `-137.5` | `-128.0` | `-42.0` | `-138.0` |
| 1535 a120 | `-117.7` | `-30.6` | `-135.7` | `-126.6` | `-42.0` | `-139.5` |
| 2047 a140 | `-117.7` | `-23.9` | `-135.6` | `-126.0` | `-41.9` | `-134.2` |
| 3071 a140 | `-117.4` | `-16.8` | `-136.1` | `-124.6` | `-41.4` | `-137.8` |
| 4095 a140 | `-117.2` | `-14.3` | `-135.2` | `-123.6` | `-41.1` | `-136.9` |

1023 tapsが膝です。それ以上ではイメージが改善せず（48 kHzはむしろ後退）、44.1 kHzの
controller 64位相マージンが`-36` → `-14 dB`へ縮み、G9が悪化します。pre-ringingの長さが
物理priorの先読み（約13 ms）を超えるためです。

## 共通事項

- ゲート仕様はspec 6。閾値の定義とprobe manifestは変更していません。release_qualityのクロスレート
  検査は、impulse列G5を両系列とも凍結ゲート`0.5 dB`に対して判定し、過渡位置の許容に
  checkpointへ保存された`focused_gentle_fraction`の差を加える定義です。
- ONNXファイルはこのリポジトリに含めません。`totton-audio-nn/data/nmse/`はrun16ペアを使います。
- 数値の正史はstrict-FP32のworst-probe gateです。`routing/`の診断R1〜R4と図は補助です。
