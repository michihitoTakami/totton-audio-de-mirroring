# totton-audio-de-mirroring

CAPB（Constrained Adaptive Prototype-Blend）を用いた、波形の立ち上がりや立ち下がりを重視する音声アップサンプラです。

44.1 kHzまたは48 kHzの入力を、3種類の固定FIRフィルタでそれぞれ2倍に補間します。小さなニューラルネットワークが入力音声を観察し、3つの結果をどの割合で混ぜるかだけを決めます。混合比はすべて0以上で、合計は常に1です。ニューラルネットワークがFIR係数や新しい音声波形を生成することはありません。

基本的な考え方は次のとおりです。

1. 定常的な音では、不要な高域の複製を強く除去するフィルタを多く使えるようにする
2. 急な立ち上がりやクリックでは、前後の振動が少ないフィルタを多く使えるようにする
3. その中間では、3つのフィルタを連続的な割合で混ぜる

### はじめに知っておく用語

- FIRフィルタ: 有限個の係数で入力波形を畳み込むデジタルフィルタ。このリポジトリでは係数を左右対称にし、周波数によって遅延が変わらない直線位相にしています。
- ナイキスト周波数: sample rateの半分の周波数です。44.1 kHz音源では22.05 kHz、48 kHz音源では24 kHzです。元のデジタル音声は、この周波数を超える情報を持ちません。
- イメージ成分: sample rateを上げるときに生じる、元のスペクトルの不要な複製です。補間フィルタで十分に減衰させる必要があります。
- リンギング: stepやクリックのような急変の前後に生じる振動です。周波数を急峻に切るフィルタほど長くなりやすい性質があります。
- Besselフィルタ: 立ち上がり・立ち下がりの波形を乱しにくい、緩やかな周波数特性のフィルタです。CAPBでは低リンギング側の比較基準に使います。
- controller: 入力音声から3つのFIRの混合比を計算するニューラルネットワークです。
- probe / gate: probeは評価用の既知信号、gateはその測定値に対する合格条件です。

## 設計目標

- 入力のナイキスト周波数を超える未知の成分を推測・生成しない
- 定常信号ではイメージ成分を十分に抑える
- 不連続点では、低リンギングの比較基準であるBesselフィルタより振動を悪化させない
- 3つのFIRを対称・同一長・同一利得・共通遅延にそろえる
- 評価値の平均ではなく、通常probeと未学習条件を模したheld-out probeを含む最悪値で合否を決める

20 kHzを境に信号を急峻に分割するとGibbsリンギングが生じるため、CAPBは固定的な帯域分割を行いません。可聴帯域の忠実度は、3つのFIRの設計と、利得・波形・位相・群遅延に対する最悪値gateで確認します。

## パイプライン

| 段 | 変換 | 実装 | 役割 |
|---|---:|---|---|
| Stage 1 | 44.1→88.2 kHz / 48→96 kHz | CAPB | リンギングの少なさとイメージ抑制の強さを入力に合わせて調整 |
| Stage 2 | 88.2→705.6 kHz / 96→768 kHz | 固定DSP（2×を3段） | Stage 1の結果をさらに8倍へ補間 |

Stage 2は音作りを行わない高レート搬送段です。各段の入力Nyquistをcutoffとする、
Kaiser窓（β=16）の短い線形位相FIRを`255 → 63 → 39 taps`でカスケードします。
可聴帯域を平坦に保ちながら、各2倍補間で生じるイメージ成分だけを除去します。
両sample-rate系列で正規化した係数形状は共通で、カスケードの群遅延は最終出力上の
589 samples（705.6 kHzで約0.835 ms、768 kHzで約0.767 ms）です。

Stage 1では、3つのFIRフィルタをまとめたprototype bankを使います。

- `sharp`: 定常信号向け。狭い遷移帯域と強いイメージ除去
- `mid`: イメージ除去とリンギングの少なさの中間
- `gentle`: 6次Besselフィルタの振幅応答へ合わせた低リンギング側

短いFIRの左右へ0を追加して3つの長さと中心位置をそろえます。44.1 kHz系列は483 taps（補償前の遅延は88.2 kHzで241 samples）、48 kHz系列は277 taps（96 kHzで138 samples）です。中心を合わせた畳み込みにより、出力上ではこの固定遅延を補償します。

## CAPBのアルゴリズム

CAPBは、特性の異なる3本の固定FIRを並列に動かし、小さなニューラルネットワークが音声の局所的な性質を見て、その混合比だけを時間方向に変える方式です。「3本から1本だけを選ぶ」のではなく、3本の出力を滑らかに混ぜます。

```text
入力波形 x
  ├─ 2倍用の0挿入 → sharp FIR  ─┐
  ├─ 2倍用の0挿入 → mid FIR    ─┼─ 混合比を掛けて加算 → Stage 1出力 y
  └─ 2倍用の0挿入 → gentle FIR ─┘
          │
          └─ 入力波形を解析するcontrollerが
             [sharp, mid, gentle]の混合比を出力
```

入力を`x`、3本のFIRを`h_k`、各FIRの出力を`y_k`とすると、処理は次のように表せます。

```text
x_up[n] = x[n/2]  (nが偶数)、0 (nが奇数)
y_k[n]  = (h_k * x_up)[n]
w_k[m]  = softmax(controller(x))[k, m]
~w_k[n] = linear_interpolate(w_k)[n]
y[n]    = Σ_k ~w_k[n] y_k[n]
```

`*`は畳み込み、`k`は3つのFIRの番号です。`m`は音声sampleより間隔の粗い制御時刻を表します。制御時刻ごとの重み`w_k[m]`を出力sample数まで直線補間した`~w_k[n]`を、3本のFIR出力へ適用します。softmax関数を使うため、各制御時刻で常に次の制約が成立します。

```text
w_k >= 0
Σ_k w_k = 1
```

つまりcontrollerは、各時刻の混合について、検証済みFIRの凸結合の外へ自由なフィルタ係数を出せません。学習されるのは混合比を決めるcontrollerだけで、FIR係数は学習中もcheckpoint読込後も固定です。

ただし、各FIRが線形でも、入力に応じて混合比が時間変化するCAPB全体は厳密なLTI系ではありません。混合比が定常toneへ同期して動くと振幅変調となり、入力にないsidebandを作る可能性があります。そのため定常・周期信号では、同じ信号の時間平均重みによる固定混合との差と、重み自体の時間平均からの偏差を学習lossで抑え、SMPTE two-tone gateで実出力を検査します。この制約は学習時だけに適用し、ランタイムguardやFIRの帯域分割は追加しません。

### 3本のFIRの役割

| FIR | 周波数・時間応答の性格 | 主に使う区間 |
|---|---|---|
| `sharp` | 狭い遷移帯域、90 dB設計の強いイメージ抑制。代わりに不連続点の前後でリンギングが長くなりやすい | 定常音、ノイズ、緩やかに変化する音 |
| `mid` | イメージ抑制とリンギングの中間 | `sharp`と`gentle`の中間的な区間 |
| `gentle` | 6次Besselの緩やかな振幅応答へ合わせた101-tap FIR。イメージ抑制は弱いがedgeのリンギングを抑えやすい | step、矩形波、クリックなどの不連続点 |

Kaiser窓を使う`sharp`と`mid`はsample-rate系列ごとに遷移帯域を変えます。入力のナイキスト周波数が22.05 kHzの場合と24 kHzの場合では利用できる遷移幅が異なるため、単純な周波数スケーリングはしません。一方、`gentle`は両系列ともcutoff 20 kHzの6次Bessel振幅応答を基準にします。

全FIRには次の構造制約があります。

- 左右対称なFIR係数による直線位相
- 低域の基準周波数で、2倍補間に必要な利得へ正規化
- 短いFIRの左右へ0を追加し、最長FIRと中心を一致
- 3つすべてで共通の群遅延

sample間へ0を挿入すると元の帯域の振幅が1/2になるため、FIR側は補間率2に相当する利得を持ちます。また、全FIRの位相と中心が一致しているので、混合比を変えても異なる遅延の波形同士を足して位相キャンセルを起こす構成にはなりません。

### Controllerが混合比を決める方法

controllerは入力波形を直接受け取る5段の小さな1次元畳み込みネットワークです。各段で時間方向の情報を圧縮し、入力64 samplesごとに3個の未正規化スコアを出します。このスコアへsoftmax関数を適用したものが`sharp`、`mid`、`gentle`の重みです。

判定が音量へ依存しないよう、controllerへ入れる波形だけを処理単位内のpeakで正規化します。3本のFIRは正規化前の入力を処理するため、この正規化が出力音量を変えることはありません。

初期混合比は`sharp: 0.85 / mid: 0.10 / gentle: 0.05`です。ニューラルネットワークはこの初期値を基準に、信号内容に応じて重みを増減する方法を学びます。初期値を表すbiasは固定し、学習開始直後に1本のFIRだけを常時使う状態へ偏って学習が止まることを防ぎます。

### 何を学習しているか

教師波形は出力sample rateで生成した帯域制限済み信号です。そこから1 sampleおきに取り出して入力を作るため、`入力 == 教師波形[::2]`が正確に成立します。controllerへ入力のナイキスト周波数より上の原音成分を推測・復元させる学習にはなっていません。一方で、入力依存の混合比が変動すると副帯域を生む可能性があるため、これは定常変調lossとtwo-tone gateで別途制約します。損失関数は次の役割を分担します。

- `waveform L1` / `multi-resolution STFT`: 時間波形と周波数分布を教師へ近づける
- `plateau ripple`: stepや矩形波の平坦部に残る、特に大きなリンギングを減らす
- `quiet energy`: impulse前後など、本来無音の領域へ漏れる前後のechoを減らす
- `edge ringing`: 不連続点の近くで、`gentle`より大きくなったリンギングを罰する
- `weight total variation`: 混合比を時間方向に滑らかにし、重みの急変による副作用を抑える
- `stationary modulation`: 定常・周期信号の出力を時間平均重みによる固定混合へ近づけ、信号同期sidebandを抑える
- `entropy floor`: 1本のFIRへ早期に完全固定され、学習できなくなることを防ぐ

この設計では、ニューラルネットワークは音声sample、周波数ごとの加工量、FIR係数を生成しません。候補となる3つの波形は常に固定FIRで入力から計算され、機械学習は「現在の区間ではイメージ抑制と波形の自然さのどちらをどの程度優先するか」だけを制御します。これは未知の原音成分を復元することを意味しませんが、時変混合による副帯域の有無は実測gateで確認します。最終的な合否は学習中の誤差ではなく、両sample-rate系列のすべての評価信号に対する最悪値で判定します。

## 動作環境

- Python 3.13
- PyTorch 2.5+
- NumPy / SciPy / torchaudio
- `uv`
- 推論ターゲット: Jetson Orin Nano 8GB

セットアップ:

```bash
uv sync --extra dev
```

## 学習

44.1 kHz family:

```bash
uv run python scripts/train_capb.py \
  --data-config configs/data_generation_capb.yaml \
  --config configs/training_stage1_capb.yaml
```

44.1 kHzの最終margin polishは、同梱したpre-polish checkpointから再現できます。

```bash
uv run python scripts/train_capb.py \
  --data-config configs/data_generation_capb.yaml \
  --config configs/training_stage1_capb_44k1_margin.yaml \
  --summary-json /tmp/capb-44k1-margin.json
```

48 kHz familyの通常学習:

```bash
uv run python scripts/train_capb.py \
  --data-config configs/data_generation_capb_48k.yaml \
  --config configs/training_stage1_capb_48k.yaml
```

48 kHzのrelease margin学習は、定常歪みprobeと過渡probeを同じデータ分布へ入れ、採用済みcheckpointを小さい学習率で1 epochだけ更新します。FIR prototypeと受入gateは変更しません。CUDA学習ではTF32を既定で禁止します。controller重みの時間補間にはbit-exactなCUDA backwardがないため、seed、親checkpoint、設定hash、実行環境をcheckpointとsummaryへ記録し、最終選定は学習lossではなく再生成したprobeで行います。

```bash
uv run python scripts/train_capb.py \
  --data-config configs/data_generation_capb_48k_balanced.yaml \
  --config configs/training_stage1_capb_48k_balanced_margin.yaml \
  --init-checkpoint data/checkpoints/capb_48k/run11_48k_optimized_20260830/capb_best.pt \
  --checkpoint-dir data/checkpoints/capb_48k/run12_48k_strictfp32_balanced_20260830 \
  --summary-json /tmp/capb-48k-margin.json
```

通常の学習データは出力sample rateで合成し、入力のナイキスト周波数未満へ直線位相FIRで帯域制限した後、正確に2:1で間引いて入力を作ります。孤立clickとtone burstは、実際のprobeと同じ入力sample rateでeventを生成し、cardinalなFFT zero-paddingで帯域制限済みteacherへ変換します。どちらの経路も`source == target[::2]`を厳密に保ち、入力Nyquistを超える教師情報を作りません。

44.1 kHzデータではclick/tone burstを必ずchunk内へ配置し、70%を無noise・無clipの絶対silence supervision、30%をaugmentation耐性用としました。SMPTE/held-out two-toneを代表する`imd_two_tone`も5%含めます。分布、mask、decimation、image leakageは次で監査できます。

```bash
uv run python scripts/audit_capb_training_data.py \
  --data-config configs/data_generation_capb.yaml \
  --output-dir /tmp/capb-data-audit
```

学習されるのは約10万parameterのcontrollerだけです。3つのFIRは固定され、checkpointにはcontrollerの学習結果と対象sample-rate系列が保存されます。

### Routing v2（Sharp既定・過渡時のみ保護側）

実音源で旧controllerがギターと氷の衝突に対して逆向きへ遷移したため、routing v2では短時間RMSの対称変化、局所微分crest、波形activity densityをcontroller入力と物理priorへ追加しています。定常tone/noiseではSharp、包絡onset/offsetと持続plateau edgeではGentleを選びます。疎なclick/impulse列は`routing_prior.focused_gentle_fraction`でGentleとMidに分配し（44.1 kHz `0.90`、48 kHz `0.85`）、Midは曖昧な常用クラスではなく、48 kHz Gentleのimpulse利得誤差を補償しつつG2b/G2cを守る限定的な役割です。初期の候補はimpulseでMid=1.0を選びリンギングがSharpに近づいたため、Gentle主体へ改めました。`routing_prior.level_change_threshold`（既定`0.15`、routing v2は`0.30`）はpink noiseのRMS揺らぎを過渡と誤検出してGentleが漏れないための下限で、両値はcheckpointに保存され旧checkpointは旧挙動を保ちます。それでも学習ヘッドが定常ノイズ上でGentleを足す残りはseed依存で（3 seed中1 seedがG3を落としました）、定常サンプルの安全フレームでSharp重みが`sharp_floor`（`0.995`）を下回った分を罰する`stationary_sharp_floor`損失（重み`1000`）で解消しています。この処方は3 seed×両系列でG1〜G9を通過し、2026-09-03のrelease pairになりました。20 kHzの帯域分割は導入していません。

2-prototype（Sharp/Gentle）も比較できますが、48 kHzのGentle固定端点自体がimpulseで`0.514 dB`、10 ms impulse列で`0.527 dB`のG5誤差となり、上限`0.5 dB`を超えます。このため両rate family共通の候補は3-prototypeを維持します。

```bash
# 44.1 kHz family（48 kHzは対応する _48k 設定を使う）
uv run python scripts/train_capb.py \
  --data-config configs/data_generation_capb_routing_v2.yaml \
  --config configs/training_stage1_capb_routing_v2_3p.yaml

uv run python scripts/evaluate_capb_routing_gates.py \
  --checkpoint <checkpoint> \
  --data-config configs/data_generation_capb_routing_v2.yaml \
  --report /tmp/capb-routing-gates.json

uv run python scripts/report_capb_real_audio_routing.py \
  --audio <evaluation-only.wav> \
  --checkpoint candidate=<checkpoint> \
  --output-dir /tmp/capb-real-routing
```

実録音は学習へ混ぜず、方向性の診断と最終tie-breakだけに使います。正式採用にはrouting診断だけでなく、両系列・複数seedで変更していないcanonical/held-out G1〜G9、controller位相、Hann-OLA境界をすべて通す必要があります。生成checkpoint、実音源、ad-hoc plotはrepositoryへ追加しません。

### 長尺sharp FIRの実験

`long_sharp_1023_a120`、`long_sharp_1023_a140`、`long_sharp_1535_a120`、`long_sharp_2047_a120`、`long_sharp_2047_a140`、`long_sharp_2047_a160`、`long_sharp_3071_a120`、`long_sharp_3071_a140`、`long_sharp_4095_a120`、`long_sharp_4095_a140`を候補として利用できます。変更するのはsharpだけで、middle/gentleの係数はrelease版をそのまま対称ゼロ埋めします。全prototypeは同じ奇数長・中心sample・群遅延を持ち、採用前に位相差`1e-6 deg`以下、群遅延差`1e-9 sample`以下、対称誤差`1e-12`以下を満たす必要があります。

まず両rate familyの構造、FP32係数量子化、image応答、0.5--4 msと4--12 msのechoを比較します。

```bash
uv run python scripts/report_long_fir_candidates.py \
  --output /tmp/capb-long-fir-candidates.json
```

既存controllerを変更せずに候補bankへ移し、canonical/held-out G1--G9を実行できます。`--fir-compute-dtype float64`は係数設計ではなく固定FIR演算床を分離する診断です。

```bash
uv run python scripts/evaluate_probe_gates.py \
  --backend capb \
  --checkpoint <44k1-checkpoint> \
  --rate-family 44k1 \
  --prototype-profile long_sharp_2047_a120 \
  --fir-compute-dtype float32 \
  --report-dir /tmp/capb-long-fir-gates
```

1 epochのcontroller-only fine-tuneでは、2047 tapの半長に合わせてborder trimを1024 sampleへ広げ、従来G2bと同じ損失係数で4--12 msの長いプリエコーも監視します。

```bash
uv run python scripts/train_capb.py \
  --data-config configs/data_generation_capb.yaml \
  --config configs/training_stage1_capb_44k1_margin.yaml \
  --seed 1234 \
  --init-checkpoint <44k1-checkpoint> \
  --prototype-profile long_sharp_2047_a120 \
  --initial-controller-only \
  --border-trim 1024 \
  --far-pre-echo-window-ms 8.0 \
  --checkpoint-dir <candidate-output-dir>
```

長尺化は自動的な高音質化ではありません。固定FIRのFP32累積誤差は演算床を分離する診断値であり、それだけで採否を決めません。採用には両rate familyの全probe、controller 64位相、Hann-OLA境界64 offset、近距離・長距離echoを通過したうえで、worst image leakageをrelease比0.5 dB以上改善することを要求します。image差が0.5 dB未満ならG2b、G9、歪み、短いtap数の順で決め、どれも満たさなければrelease bankを維持します。

2026-09-01には1535/2047 tapsを両系列それぞれ3 seedでFineTuningしました。48 kHzは両候補とも3/3 seedでG1--G9を通過しましたが、44.1 kHzはG2b pre-echoが1535で`4.96e-7`、2047で`5.11e-7`となり、上限`2.5e-7`を全seedで超えました。2047の48 kHzもoffset robustnessが僅かに不合格です。当時はイメージ抑制が改善してもhard gateを満たさなかったため`release_v4`を維持しました。checkpointを含まない1535-tap研究比較は`reports/research/long_fir_1535/`にあります。

2026-09-03にrouting v2 controllerで再走査しました。impulseでSharpをほぼ混ぜなくなったため全profileがG1〜G9を通過し、483〜4095 tapsの走査で1023 tapsが膝でした（それ以上はイメージが改善せず、44.1 kHzの64位相マージンとG9が悪化）。`long_sharp_1023_a140`を両系列で採用しました（48 kHzはG3最悪`-107.8` → `-128.1 dB`、44.1 kHzは測定上同等でbank統一のため）。詳細は`reports/release/README.md`を参照してください。

### 学習済み成果物

2026-09-03のrelease候補は2ペアを並置しています。どちらもrouting v2 + sharp floor処方のcontrollerで、違いはSharp prototypeの長さです。推奨は両系列とも**run14**（`long_sharp_1023_a140`、Sharp 1023 taps）で、`reports/release/release_manifest.json`の`recommended`に記録しています。

| 系列 | 推奨checkpoint | 主要値 |
|---|---|---|
| 44.1→88.2 kHz | `data/checkpoints/capb/run14_longsharp1023_a140_20260903_44k1/capb_best.pt` | G1〜G9 CPU/CUDA全通過、SMPTE `-138.1 dB`、impulse列利得誤差 `0.447 dB`、G2b `5.6e-11`、sweepイメージ `-133.1 dB`、G3最悪（pink）`-108.8 dB`、64位相worst `-36.1 dB`、群遅延5.8 ms |
| 48→96 kHz | `data/checkpoints/capb_48k/run14_longsharp1023_a140_20260903_48k/capb_best.pt` | G1〜G9 CPU/CUDA全通過、SMPTE `-138.2 dB`、impulse列利得誤差 `0.457 dB`、G2b `1.3e-11`、G3最悪 `-128.1 dB`、64位相worst `-42.4 dB`、群遅延5.3 ms |

48 kHzは最悪イメージ行が`-107.8` → `-128.1 dB`と明確に改善します。44.1 kHzは測定上run13（483 taps）と同等で、最悪行のpink noise（controller律速）はseed間ばらつきの内側、sweepイメージは`4.4 dB`改善、G9は`-142.9` → `-138.1 dB`、群遅延は2.7 → 5.8 msです。両系列のbank・群遅延・ONNXグラフを統一するために44.1 kHzも1023 tapsを採用しました。非推奨側のrun13ペア（`run13_routing_v2_sharpfloor_20260903_*`）も同じ処方の有効な候補として保存しています。

旧release（`run11_44k1_optimized_20260829` / `run12_48k_strictfp32_balanced_20260830`）に対して、実音源で逆向きだったSharp/Gentle遷移が正方向になり、孤立impulseのリンギングは44.1 kHzで`-24.3` → `-32.4 dB`、48 kHzで`-25.7` → `-29.7 dB`に下がりました。代償はimpulse列の利得誤差が48 kHzで`0.400` → `0.457 dB`へ増えたこと（上限`0.5 dB`）と、群遅延が44.1 kHz 2.7 → 5.8 ms、48 kHz 1.4 → 5.3 msになったことです。旧checkpointは履歴として残しています。

release評価はstrict FP32を必須とし、48 kHzの数値床には同じTorch経路のrate-local fixed FIRを使います。クロスレート検査（`evaluation/release_quality.py`）は、impulse列G5を両系列とも凍結ゲート`0.5 dB`に対して判定し、過渡位置の許容にcheckpointへ保存された`focused_gentle_fraction`の差を加えます。G1〜G9本体は変更していません。選定・gate・robustness・可視化・ONNX parity・null test・タップ数走査の根拠は`reports/release/`にあります。

## 受入評価

CAPB checkpointは、44.1→88.2 kHzと48→96 kHzの両系列で、通常評価用とheld-out評価用のprobeをすべて通過するまでリリースできません。

Torch/CUDA評価はstrict FP32が既定です。`--allow-tf32`は原因調査専用で、release reportには使用できません。レポートにはTorch、CUDA、cuDNN、GPU、TF32設定を含む実行metadataが保存されます。

```bash
# 44.1 kHz family
uv run python scripts/evaluate_probe_gates.py \
  --backend capb \
  --checkpoint <44k1-checkpoint> \
  --rate-family 44k1

# 48 kHz family
uv run python scripts/evaluate_probe_gates.py \
  --backend capb \
  --checkpoint <48k-checkpoint> \
  --rate-family 48k

# controller phase / Hann-OLA boundary（48 kHzは--rate-family 48k）
uv run python scripts/evaluate_capb_transient_robustness.py \
  --checkpoint data/checkpoints/capb/run13_routing_v2_sharpfloor_20260903_44k1/capb_best.pt \
  --rate-family 44k1 \
  --output-dir /tmp/capb-robustness-44k1
```

主な合格条件:

- 矩形波とstep: 平坦部の振動、最大振幅差、飛び出し量をBessel基準以下に保つ
- impulse、impulse列、短いtone: 音が始まる前へ漏れるechoの増加を制限する
- sweep、pink noise、複数tone: イメージ成分を`-65 dB`以下にする
- 100 Hz–20 kHz: 周波数特性の平坦性を制限する（100 Hz–18 kHzと18–20 kHzで別の閾値）
- 低域: 利得誤差を制限し、pink noiseと複数toneでは0–20 kHzの位相・群遅延・時間波形の誤差も確認する
- 不要な高域成分: Bessel基準に対する増加を制限する
- SMPTE/held-out two-tone: controller変調によるsidebandを`-110 dBc`以下にする

レポートは指定した`--report-dir`以下の`gate_report.json`と`gate_report.md`へ出力されます。平均値は参考情報にすぎず、各gateの合否は最も悪いprobeで決まります。採用品の固定release evidenceは`reports/release/`へ保存し、不採用候補を残す場合はcheckpointを含まない研究資料として`reports/research/`へ分離します。

固定prototypeだけを検証する場合:

```bash
uv run python scripts/evaluate_probe_gates.py --backend prototype:gentle
uv run python scripts/run_capb_phase0.py --rate-family 44k1
uv run python scripts/run_capb_phase0.py --rate-family 48k
```

## Stage 1 → Stage 2推論

`totton-upsample`はCAPB checkpointを直接ロードできます。`configs/stage1_stage2_pipeline.yaml`の`stage1`を次のように設定してください。

```yaml
stage1:
  mode: capb
  checkpoint_path: <checkpoint>
  device: cpu
```

単一ファイル:

```bash
uv run totton-upsample input.wav \
  -o output.wav \
  -c configs/stage1_stage2_pipeline.yaml
```

バッチ:

```bash
uv run totton-upsample "audio/*.wav" \
  -o reports/batch_out \
  -c configs/stage1_stage2_pipeline.yaml \
  --output-format wav \
  --output-format metadata
```

設定の`stage1.mode=reference`は配線確認用のBessel比較経路です。リリース出力には`capb`を使用してください。Stage 2の既定backendはC++で、テスト用に`pipeline.stage2_backend=python`も選べます。

## テストと品質チェック

```bash
# 高速テスト
uv run pytest -m "not slow and not gpu" -v

# format / lint / type check / test
uv run ruff format --check src tests scripts
uv run ruff check src tests scripts
uv run mypy src
uv run pytest -v
```

## 主要ファイル

```text
configs/
  data_generation_capb.yaml
  data_generation_capb_run9_legacy.yaml
  data_generation_capb_48k.yaml
  data_generation_capb_48k_balanced.yaml
  training_stage1_capb.yaml
  training_stage1_capb_44k1_margin.yaml
  training_stage1_capb_48k.yaml
  training_stage1_capb_48k_balanced_margin.yaml
  training_stage1_capb_48k_relearn_075.yaml
  training_stage1_capb_48k_relearn_085.yaml
  training_stage1_capb_48k_stationary_warmup.yaml
  training_stage1_capb_48k_stationary.yaml
  stage1_stage2_pipeline.yaml
src/totton_audio_de_mirroring/
  data/capb_dataset.py
  data/probe_generators.py
  data/reference.py
  data/transient_supervision.py
  models/capb.py
  models/proto_bank.py
  training/capb_losses.py
  training/capb_trainer.py
  evaluation/gates.py
  evaluation/probe_suite.py
  evaluation/distortion.py
  evaluation/release_quality.py
  inference/pipeline.py
  torch_precision.py
  stage2/
scripts/
  train_capb.py
  evaluate_capb_release_quality.py
  evaluate_probe_gates.py
  evaluate_capb_transient_robustness.py
  audit_capb_training_data.py
  report_capb_distortion.py
  report_capb_impulse.py
  report_capb_precision.py
  run_capb_phase0.py
```

## 非目標

- 22.05 kHzまたは24 kHzを超える原音成分の復元
- GAN等による倍音・超音波成分の生成
- 平均metricだけによるcheckpoint採用
- probe gateを通していない学習loss改善の品質主張

このプロジェクトの優先順位は、帯域を広く見せることではなく、入力にない情報を作らず、イメージ成分と時間領域artifactを測定可能な制約内に収めることです。

## ライセンス

[MIT License](LICENSE) の下で提供します。
