# CAPB release evidence — 2026-09-03

4つの候補ペアを並置して保存しています。いずれもrouting v2 + sharp floor処方のcontrollerで、
違いはprototype bankと打撃時のGentle比率です。**推奨は両系列ともrun16**（`v5b_sharp1023_midflat70`
bank、`focused_gentle_fraction 0.3`、gate spec 6）です（`release_manifest.json`の`recommended`）。

| run | bank | Gentle比率 | 位置付け |
|---|---|---|---|
| run13 | `release_v4`（Sharp 483/277 taps、Mid Kaiser 80 dB、Gentle Bessel6@20k） | 0.90 / 0.85 | routing v2 + sharp floorの最初の採用品 |
| run14 | `long_sharp_1023_a140`（Sharp 1023 taps） | 0.90 / 0.85 | Sharp長尺化。48 kHz最悪イメージ`-107.8` → `-128.1 dB` |
| run15 | `v5b_sharp1023_midflat70`（Mid Kaiser 20–24 kHz 70 dB ~100 taps） | 0.6 / 0.6 | flat Mid導入 |
| **run16** | `v5b_sharp1023_midflat70` | **0.3 / 0.3** | 推奨。打撃時のGentle比率を最小限にし可聴域の減衰を抑える |

### run16を推奨する理由

ABX（ハイハット素材）でGentle単体は識別可、Sharp単体とCAPBは識別不能でした。Gentleは15 kHzで
`-4.3 dB`、20 kHzで`-10 dB`落ちるので、打撃の瞬間にGentleへ寄るほど打撃の高域が削れます。run15で
Midを20 kHzまでフラットな短いKaiserに置き換えた上で、run16では打撃時のGentle比率を0.6 → 0.3に下げ、
残りをflat Midで受けます。比率0.0はMidのGibbsリップルが5 kHz矩形でBessel参照の1.1倍を超えるため
G2で不合格（物理的に正しい判定）で、0.3が両系列共通で通る下限です。

| 指標（run15 → run16） | 44.1 kHz | 48 kHz |
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

### gate spec 6（2026-09-03）

矩形波 / DC stepのリンギング指標（G1/G2）は、出力とBessel参照を8倍sincオーバーサンプルしてから
エッジ整列と0.1–0.8 msの窓切りを行います。閾値の定義（Bessel参照比1.1倍、overshoot +0.005）は
不変です。spec 5では出力サンプル格子でエッジを検出していたため、48 kHzの2 kHz / 5 kHz矩形
（エッジが96 kHzサンプルの中間に揃う）で窓が1サンプル飛び、5 kHz矩形のovershoot閾値が
44.1 kHz `0.549`に対して48 kHz `0.283`と2倍違っていました。spec 6では`0.539` / `0.567`と一致します。
run13〜run15のゲートレポートもspec 6で再生成し、いずれも全通過です。
比率0.0は引き続きG2（5 kHz / 4.4 kHz矩形のplateau_rms）で不合格になります。

## run13 — `run13_routing_v2_sharpfloor_20260903/`

| Family | checkpoint | G1〜G9 CPU / CUDA | G5 impulse列 | G2b | G3最悪 | 64位相 worst | ONNX parity / null |
|---|---|---|---:|---:|---:|---:|---|
| 44.1k | `data/checkpoints/capb/run13_routing_v2_sharpfloor_20260903_44k1/capb_best.pt` | PASS / PASS | `0.447 dB` | `5.1e-11` | `-116.1 dB` | `-35.9 dB` | `7.2e-7` / `-130.8 dB` |
| 48k | `data/checkpoints/capb_48k/run13_routing_v2_sharpfloor_20260903_48k/capb_best.pt` | PASS / PASS | `0.451 dB` | `1.6e-11` | `-107.8 dB` | `-42.0 dB` | `5.4e-7` / `-138.6 dB` |

## run14 — `run14_longsharp1023_a140_20260903/`

controllerはrun13から`long_sharp_1023_a140` bank上へcontroller-only転写して3 seed fine-tuneしたもの
（seed 1234を採用）。mid/gentleは`release_v4`と同一係数を対称ゼロ埋めし、共通群遅延511 sample。

| Family | checkpoint | G1〜G9 CPU / CUDA | G5 impulse列 | G2b | G3最悪 | 64位相 worst | ONNX parity / null |
|---|---|---|---:|---:|---:|---:|---|
| 44.1k | `data/checkpoints/capb/run14_longsharp1023_a140_20260903_44k1/capb_best.pt` | PASS / PASS | `0.447 dB` | `5.6e-11` | `-108.8 dB` | `-36.1 dB` | `1.0e-6` / `-127.4 dB` |
| 48k | `data/checkpoints/capb_48k/run14_longsharp1023_a140_20260903_48k/capb_best.pt` | PASS / PASS | `0.457 dB` | `1.3e-11` | `-128.1 dB` | `-42.4 dB` | `1.1e-6` / `-127.5 dB` |

3 seed（1234 / 2026 / 4649）すべてが両系列でCPU・strict-FP32 CUDAのG1〜G9を通過
（[selection/seed_summary.json](run14_longsharp1023_a140_20260903/selection/seed_summary.json)）。
sweep画像は44.1 kHz `-128.7` → `-133.1 dB`、48 kHz `-116.0` → `-133.5 dB`。fp32累積誤差床は
長いFIRで浅くなり、THDは44.1 kHz `-145.5` → `-142.1 dB`、48 kHz `-140.3` → `-133.1 dB`
（rate-local固定FIR床に対して評価するrelease_qualityはPASS）。

## run15 — `run15_v5b_midflat_g06_20260903/`

controllerはrun14からcontroller-only転写して`v5b_sharp1023_midflat70` bank上で3 seed fine-tune
（seed 1234を採用）。

| Family | checkpoint | G1〜G9 CPU / CUDA | G5 impulse列 | G2b | G3最悪 | 64位相 worst | ONNX parity / null |
|---|---|---|---|---:|---:|---:|---:|---|
| 44.1k | `data/checkpoints/capb/run15_v5b_midflat_g06_20260903_44k1/capb_best.pt` | PASS / PASS | `0.294 dB` | `1.7e-11` | `-133.1 dB` | `-32.8 dB` | `1.0e-6` / `-128.2 dB` |
| 48k | `data/checkpoints/capb_48k/run15_v5b_midflat_g06_20260903_48k/capb_best.pt` | PASS / PASS | `0.311 dB` | `1.2e-11` | `-133.5 dB` | `-42.6 dB` | `1.1e-6` / `-128.2 dB` |

3 seed（1234 / 2026 / 4649）すべてが両系列でCPU・strict-FP32 CUDAのG1〜G9を通過し、G5は
`0.29〜0.33 dB`、pink noiseのSharp<0.99フレームは全seedで0%、64位相worstは`-33〜-43 dB`
（[selection/seed_summary.json](run15_v5b_midflat_g06_20260903/selection/seed_summary.json)）。
null testの入力probeは`/tmp/capb_null_inputs_20260903/probe_{44100,48000}.wav`（2 s log sweep +
低レベルノイズ + 3クリック、16 bit）で、SHA-256は各`null_test/*.json`に記録しています。

## run16 — `run16_v5b_midflat_g03_20260903/`

controllerはrun14からcontroller-only転写して`v5b_sharp1023_midflat70` bank上で3 seed fine-tune
（比率0.3、seed 1234を採用）。

| Family | checkpoint | G1〜G9 CPU / CUDA（spec 6） | G5 impulse列 | G2b | G3最悪 | 64位相 worst | ONNX parity / null |
|---|---|---|---:|---:|---:|---:|---|
| 44.1k | `data/checkpoints/capb/run16_v5b_midflat_g03_20260903_44k1/capb_best.pt` | PASS / PASS | `0.164 dB` | `4.6e-11` | `-133.1 dB` | `-37.3 dB` | `1.0e-6` / `-128.2 dB` |
| 48k | `data/checkpoints/capb_48k/run16_v5b_midflat_g03_20260903_48k/capb_best.pt` | PASS / PASS | `0.175 dB` | `3.3e-11` | `-133.5 dB` | `-38.4 dB` | `1.1e-6` / `-128.2 dB` |

3 seed（1234 / 2026 / 4649）すべてが両系列でCPU・strict-FP32 CUDAのG1〜G9（spec 6）を通過し、
G5は`0.16〜0.18 dB`、pink noiseのSharp<0.99フレームは全seedで0%、64位相worstは`-37〜-38 dB`。
最悪行は2 kHz矩形のplateau_rms（余裕4.7% / 6.3%）
（[selection/seed_summary.json](run16_v5b_midflat_g03_20260903/selection/seed_summary.json)）。

## タップ数の走査

run13 controllerを学習なしで各bankへ載せ替え、483〜4095 tapsを走査しました
（[long_fir_sweep_summary.json](run14_longsharp1023_a140_20260903/selection/long_fir_sweep_summary.json)）。

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
物理priorの先読み（約13 ms）を超えるためです。2026-09-01に1535/2047 tapsが44.1 kHz G2bで
落ちた原因（impulseでSharpが約10%混入）はrouting v2で解消しており、今回は全profileが通過しました。

## 共通事項

- ゲート仕様はspec 6（矩形/DC stepのリンギング指標を8倍sincオーバーサンプル上で測定）。閾値の定義とprobe manifestは変更していません。run13〜run15のゲートレポートもspec 6で再生成しています。release_qualityのクロスレート
  検査はrun13採用時に明文化した定義（impulse列G5は凍結ゲートに対して判定、位置許容に
  `focused_gentle_fraction`差を加算）です。
- ONNXファイルはこのリポジトリに含めません。`totton-audio-nn/data/nmse/`はrun16ペアへ差し替えます
  （run13ペアはPR #696、run14ペアはPR #697、run15ペアはPR #698で導入済み）。
- 数値の正史はstrict-FP32のworst-probe gateです。`routing/`の診断R1〜R4と図は補助です。
