# CAPB release evidence — 2026-09-03（run16）

現在の推奨は両rate familyとも**run16**（`v5b_sharp1023_midflat70` bank、
`focused_gentle_fraction 0.3`、gate spec 7）です。`release_manifest.json`の`recommended`が正史です。

このディレクトリには推奨ペアの受入証跡だけを置きます。2026-09-04に、非推奨となった
run13〜run15の証跡バンドルと1535-tap研究比較（`reports/research/long_fir_1535/`）を削除しました。
必要な場合はcommit `1541917`から復元できます。

## run16 — `run16_v5b_midflat_g03_20260903/`

controllerはrouting v2 + sharp floor処方で学習し、`v5b_sharp1023_midflat70` bank上で
3 seed（1234 / 2026 / 4649）のcontroller-only fine-tuneを行い、seed 1234を採用しました。

| Family | checkpoint | G1〜G9 CPU / CUDA（spec 7） | G5 impulse列 | G2b | G3最悪 | 64位相 worst | ONNX parity / null |
|---|---|---|---:|---:|---:|---:|---|
| 44.1k | `data/checkpoints/capb/run16_v5b_midflat_g03_20260903_44k1/capb_best.pt` | PASS / PASS | `0.164 dB` | `4.6e-11` | `-133.1 dB` | `-37.3 dB` | `1.0e-6` / `-128.2 dB` |
| 48k | `data/checkpoints/capb_48k/run16_v5b_midflat_g03_20260903_48k/capb_best.pt` | PASS / PASS | `0.175 dB` | `3.3e-11` | `-133.5 dB` | `-38.4 dB` | `1.1e-6` / `-128.2 dB` |

採用seed 1234は両系列でCPU・strict-FP32 CUDAのG1〜G9（spec 7）を通過します。
G5は`0.16〜0.18 dB`、pink noiseのSharp<0.99フレームは0%、64位相worstは`-37〜-38 dB`です。

リンギングgateの最悪行は次のとおりで、**余裕は約59〜61%**あります。

| Family | G1最悪行 | 余裕 | G2最悪行 | 余裕 |
|---|---|---:|---|---:|
| 44.1k | `square_73hz_held` overshoot `0.0375` / `0.0962` | **61.0%** | `square_1000hz` plateau_rms `2.04e-4` / `5.0e-4` | **59.1%** |
| 48k | `square_500hz` overshoot `0.0300` / `0.0776` | **61.3%** | `square_1000hz` overshoot `0.0300` / `0.0776` | **61.3%** |

spec 6 まで最悪行として記録していた「2 kHz矩形のplateau_rms、余裕4.7% / 6.3%」は、
窓が半周期を2.8個跨いだ退化した行の残差でした（下記「gate spec 7」）。3 seedの選定記録は
[selection/seed_summary.json](run16_v5b_midflat_g03_20260903/selection/seed_summary.json)
にありますが、これは spec 6 の窓で測った当時の値です（非採用seedのcheckpointは残っていないため再測不能）。

null testの入力probeは2 s log sweep + 低レベルノイズ + 3クリック（16 bit）で、SHA-256は
各`null_test/*.json`に記録しています。

### bankと打撃時Gentle比率を選んだ理由

bankはSharp 1023 taps（Kaiser 140 dB）、Midは20 kHzまでフラットな短いKaiser（阻止24 kHz、70 dB、
約100 taps）、GentleはBessel6@20 kHzです。

ABX（ハイハット素材）でGentle単体は識別可、Sharp単体とCAPBは識別不能でした。Gentleは15 kHzで
`-4.3 dB`、20 kHzで`-10 dB`落ちるので、打撃の瞬間にGentleへ寄るほど打撃の高域が削れます。
Midをフラットな短いKaiserに置き換えたうえで、打撃時のGentle比率を`0.9` → `0.6` → `0.3`と下げ、
残りをflat Midで受ける構成に到達しました。

比率`0.3`が両系列共通で通る下限です。Gentle自体を20 kHzまでフラットにする案はG7
（Bessel参照に対するイメージ帯の増加）を`+12 dB`超過して不合格でした。

なお spec 6 まで「比率0.0はMidのリップルが5 kHz矩形でG2を落とす」と記載していましたが、
**この主張は撤回します**。固定checkpointに対して `focused_gentle_fraction` を 0.3 から 0.0 へ
変えても、13個の矩形 / DC step probe の plateau・overshoot metric は**bit一致**します
（比率は疎な孤立impulseのrisk経路にしか触らないため、矩形probeは比率を見ていません）。
当該の`g0.0`数値は別途fine-tuneした別のcontrollerのもので、そのcheckpointは残っていません。
Midのリップルが実際にBessel参照を超えることは、有効なplateauを持つ全probeで測れます
（100 Hzで`25.99`倍、1000 Hzで`38.25`倍）。

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
そのknob（広帯域HF onsetだけGentleをフラットMidへ振り替えるrouting class）は2026-09-04に
試作し不採用としました。上限は素材次第で1.25〜3.48 dBありますが、gateを通すために必要な制約が
そのまま効果を削るため、実際に取れたのは0.06〜0.47 dBでした。経緯と測定値は
[`docs/hf_onset_routing_experiment.md`](../../docs/hf_onset_routing_experiment.md)にあります。

比率の走査値は[selection/fraction_sweep_summary.json](run16_v5b_midflat_g03_20260903/selection/fraction_sweep_summary.json)、
打撃時の帯域別measurementは[selection/hit_metrics.json](run16_v5b_midflat_g03_20260903/selection/hit_metrics.json)にあります。

### gate spec 7（2026-09-04）

矩形波 / DC stepのリンギング指標（G1/G2）は、出力とBessel参照を8倍sincオーバーサンプルしてから
エッジ整列と窓切りを行います。閾値の定義（Bessel参照比1.1倍、overshoot +0.005）とprobe manifest
（hash `085959477798407d` / `6529deadaab177fe`）は不変で、変わるのは**どのprobeでどの行を、
どの窓で出すか**だけです。

spec 6 まで窓はprobe周波数に依らずエッジ後 0.1–0.8 ms 固定でした。矩形の半周期がこの0.70 msより
短い**625 Hz以上の全probeで窓が次の遷移を跨ぎ**、「plateauリップル」が矩形波自体になっていました。
2 kHzでは窓が半周期を2.8個、5 kHzでは7.0個跨ぎます。結果、リンギングが48倍違う3本のprototypeが
2 kHzで`1.017`〜`1.048`（許容1.10）に収まって判別不能になり、5 kHzでは順序が反転して
**フラットなFIRほど悪い**と出ていました（gentle `1.004` 対 mid `1.094` / sharp `1.095`）。

spec 7 の規則（probeを見ずに先に確定）:

```
guard = 0.1 ms（plateau_start_ms と対称。両エッジから同じだけ離す）
end   = min(0.8 ms, 半周期 − guard)
plateau が解像できる ⇔ (end − 0.1 ms) ≥ 0.1 ms
```

| probe | 半周期 | 窓 | 扱い |
|---|---|---|---|
| 50 / 73 / 100 / 331 / 500 Hz, 500hz_a005, DC step | ≥1.0 ms | `[0.1, 0.8]` | **spec 6と同一** |
| 1000 Hz | 0.500 ms | `[0.1, 0.4]` | 窓を狭める（初めて本物のplateauになる） |
| 1730 / 2000 / 4400 / 5000 Hz | ≤0.289 ms | — | **plateau/overshoot行を出さない** |

規則は窓を狭めるか行を落とすだけで、広げることはありません。既存の有効な窓の値は
bit一致で保たれます（`visualization/distortion/summary.json` の100 / 500 Hz plateau値32個で確認）。
半周期がsharpの整定より短い矩形には**リンギングのないplateauが物理的に存在しない**ため、
行を落とすのは閾値の緩和ではなく未定義量の測定停止です。行を落としたprobeは
`gate_report.json` の `skipped` と `.md` の「NOT a pass」表に理由付きで記録されます。

G1/G2 の境界も規則から導出します（`500 / (0.8 + 0.1) = 555.56 Hz`）。G1 = plateauが完全に
整定する矩形とDC step、G2 = 次のエッジでplateauが切れる矩形（現行probeでは1000 Hzのみ）。
`plateau_rms` の判別力は 2 kHz の3%幅から 1000 Hz の47倍 / 38倍へ回復します。

**未充当の被覆**: 1666.7 Hz を超える矩形（1730 / 2000 / 4400 / 5000 Hz）は、
どのgateでもplateauリンギングを測られません。これらのprobe自体は G5_gain（低域RMS利得誤差
≤0.5 dB）と G7_no_added_hf（イメージ帯のBessel比 +3 dB以下）で引き続き判定され、
15 kHzの倍音は G4_flatness が縛ります。埋めるにはprobe manifestの改訂
（新しいhashと次のspec bump）が必要で、本specの範囲外です。

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

- ゲート仕様はspec 7。閾値の定義とprobe manifestは変更していません。release_qualityのクロスレート
  検査は、impulse列G5を両系列とも凍結ゲート`0.5 dB`に対して判定し、過渡位置の許容に
  checkpointへ保存された`focused_gentle_fraction`の差を加える定義です。
- ONNXファイルはこのリポジトリに含めません。`totton-audio-nn/data/nmse/`はrun16ペアを使います。
- 数値の正史はstrict-FP32のworst-probe gateです。`routing/`の診断R1〜R4と図は補助です。
