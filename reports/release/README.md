# CAPB release evidence — 2026-09-03

3つの候補ペアを並置して保存しています。いずれもrouting v2 + sharp floor処方のcontrollerで、
違いはprototype bankと打撃時のGentle比率です。**推奨は両系列ともrun15**（`v5b_sharp1023_midflat70`
bank、`focused_gentle_fraction 0.6`）です（`release_manifest.json`の`recommended`）。

| run | bank | Gentle比率 | 位置付け |
|---|---|---|---|
| run13 | `release_v4`（Sharp 483/277 taps、Mid Kaiser 80 dB、Gentle Bessel6@20k） | 0.90 / 0.85 | routing v2 + sharp floorの最初の採用品 |
| run14 | `long_sharp_1023_a140`（Sharp 1023 taps、Mid/Gentleはrelease_v4） | 0.90 / 0.85 | Sharp長尺化。48 kHz最悪イメージ`-107.8` → `-128.1 dB` |
| **run15** | `v5b_sharp1023_midflat70`（Sharp 1023 taps、**Mid Kaiser 20–24 kHz 70 dB ~100 taps**、Gentle Bessel6@20k） | **0.6 / 0.6** | 推奨。打撃時の可聴域減衰を減らしつつG1〜G9通過 |

### run15を推奨する理由

ABX（ハイハット素材）でGentle単体は識別可、Sharp単体とCAPBは識別不能でしたが、Gentleは
15 kHzで`-4.3 dB`、20 kHzで`-10 dB`落ちるため、打撃の瞬間にGentleへ寄るほど打撃の高域が削れます。
run15は(1) Midを20 kHzまでフラットな短いKaiser（阻止24 kHz、70 dB、約100 taps）に置き換え、
(2) 打撃時のGentle比率を0.9 → 0.6に下げて、打撃の高域をSharpに近づけました。

| 指標（run14 → run15） | 44.1 kHz | 48 kHz |
|---|---|---|
| G5 impulse列利得誤差 | `0.447` → `0.294 dB` | `0.457` → `0.311 dB` |
| G2b pre-echo | `5.6e-11` → `1.7e-11` | `1.3e-11` → `1.2e-11` |
| 孤立impulseリンギング | `-32.4` → `-23.4 dB` | `-29.7` → `-21.9 dB` |
| ハイハット打撃ピーク（Sharp比） | `-1.47` → `-1.25 dB` | — |
| ハイハット16–20 kHz（Sharp比） | `-4.6` → `-3.8 dB` | — |
| G3最悪 | `-108.8` → `-133.1 dB` | `-128.1` → `-133.5 dB` |
| 64位相 worst | `-36.1` → `-32.8 dB` | `-42.4` → `-42.6 dB` |

代償は孤立impulseのリンギングが約9 dB戻ること（それでもSharp単体の`-15 dB`より7〜8 dB低い）と、
44.1 kHzのadded AM sidebandが`-142.6` → `-132.6 dB`になることです（G9閾値`-110 dB`）。
比率をさらに下げた0.0 / 0.3は44.1 kHzでは全通過しますが、48 kHzでG1（2 kHz矩形）とG2（5 kHz矩形）を
落とすため、両系列共通の値として0.6を採用しました（`run15_.../selection/fraction_sweep_summary.json`）。

Gentle自体を20 kHzまでフラットにする案（Bessel4@24 kHz）も試しましたが、G7（Bessel6@20 kHz参照に
対するイメージ帯の増加 ≤ 3 dB）を両系列で`+12〜14 dB`超過して不合格でした。Gentleがエッジで使われる
限り、その阻止帯はBessel6@20 kHz以上に深くなければならず、フラット化はG7と両立しません。

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

- ゲート本体（G1〜G9、probe manifest、閾値）は変更していません。release_qualityのクロスレート
  検査はrun13採用時に明文化した定義（impulse列G5は凍結ゲートに対して判定、位置許容に
  `focused_gentle_fraction`差を加算）です。
- ONNXファイルはこのリポジトリに含めません。`totton-audio-nn/data/nmse/`はrun15ペアへ差し替えます
  （run13ペアはPR #696、run14ペアはPR #697で導入済み）。
- 数値の正史はstrict-FP32のworst-probe gateです。`routing/`の診断R1〜R4と図は補助です。
