# CAPB release evidence — 2026-09-03

2つの候補ペアを並置して保存しています。どちらもrouting v2 + sharp floor処方のcontrollerで、
違いはSharp prototypeの長さだけです。**推奨は両系列ともrun14（`long_sharp_1023_a140`、Sharp 1023 taps）**
です（`release_manifest.json`の`recommended`）。

| 系列 | 推奨 | 理由 |
|---|---|---|
| 48→96 kHz | **run14** | 最悪イメージ行が`-107.8` → `-128.1 dB`（`+20 dB`）。G1〜G9、ロバストネス、pink漏れは同等。代償はG9 `-143.9` → `-138.2 dB`、群遅延1.4 → 5.3 ms、G5余裕9.7 → 8.6%。 |
| 44.1→88.2 kHz | **run14** | 測定上はrun13と同等。最悪イメージ行（pink noise、controller律速）は`-116.1` → `-108.8 dB`だがseed間ばらつき（run13 `-111〜-120`、run14 `-109〜-115 dB`）の内側で、sweepイメージは`-128.7` → `-133.1 dB`と改善。G9 `-142.9` → `-138.1 dB`、群遅延2.7 → 5.8 ms。両系列でbank・群遅延・ONNXグラフを統一するために採用。 |

44.1 kHzの採用は「最悪イメージ行が`0.5 dB`以上改善」という長尺FIRの採用規則を機械的には満たしません。
この規則はSharp filterが最悪行を決めている場合の判定基準で、44.1 kHzの最悪行はcontrollerのpink noise
漏れが決めており、フィルタ長を変えても動かないためです。run14ペアのクロスレートrelease_qualityは
[run14 release_quality](run14_longsharp1023_a140_20260903/release_quality/release_quality.md)でPASSです。
非推奨側のrun13も同じ処方の有効なペアとして保存しています。

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
- ONNXファイルはこのリポジトリに含めません。`totton-audio-nn/data/nmse/`はrun14ペアへ差し替えます
  （run13ペアはPR #696で導入済み）。
- 数値の正史はstrict-FP32のworst-probe gateです。`routing/`の診断R1〜R4と図は補助です。
