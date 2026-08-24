# totton-audio-de-mirroring

CAPB（Constrained Adaptive Prototype-Blend）を用いた、時間応答優先の音声アップサンプラです。

44.1 kHzまたは48 kHzの入力を、固定された対称FIRプロトタイプの凸結合で2倍に補間します。小さなニューラルコントローラが信号の状態に応じてブレンド比だけを選び、FIR係数そのものや音声波形を自由生成することはありません。

## 設計目標

- 超音波成分を推測・生成しない
- 定常信号ではイメージ成分を十分に抑える
- 不連続点ではBessel基準を超えるリンギングを生じさせない
- 全プロトタイプを対称・同一長・同一利得・共通群遅延にそろえる
- 平均値ではなく、canonical/held-out probeの最悪値で合否を決める

CAPBは旧NMSEのhard 20 kHz band splitを使用しません。急峻な帯域分割自体がGibbsリンギングを戻すためです。低域透明性は構造的な完全バイパスではなく、固定FIR bankの設計とworst-caseの利得・波形・位相・群遅延gateで保証します。

## パイプライン

| 段 | 変換 | 実装 | 役割 |
|---|---:|---|---|
| Stage 1 | 44.1→88.2 kHz / 48→96 kHz | CAPB | リンギングとイメージ抑制のトレードオフを適応選択 |
| Stage 2 | 88.2→705.6 kHz | DSP（2×を3段） | 時間応答を保った高レート補間 |

Stage 1のFIR bankには`sharp`、`mid`、`gentle`があります。

- `sharp`: 定常信号向け。狭い遷移帯域と強いイメージ除去
- `mid`: 抑制量と時間応答の中間点
- `gentle`: Bessel-6の振幅応答へ合わせた低リンギング端点

各プロトタイプは共通中心へパディングされます。44.1 kHz familyは483 taps（未補償遅延241 samples @ 88.2 kHz）、48 kHz familyは277 taps（138 samples @ 96 kHz）です。centered convolutionにより出力タイムライン上の固定遅延を補償します。

## CAPBのアルゴリズム

CAPBを簡単に言うと、特性の異なる3本の固定FIRを並列に動かし、小さなニューラルネットワークが音声の局所的な性質を見て、その混合比だけを時間方向に変える方式です。「3本から1本を選択する」hard switchではなく、3本の出力を連続的にブレンドします。

```text
入力波形 x
  ├─ 2x zero-stuff → sharp FIR  ─┐
  ├─ 2x zero-stuff → mid FIR    ─┼─ 重み付き和 → Stage 1出力 y
  └─ 2x zero-stuff → gentle FIR ─┘
          │
          └─ 入力波形を解析するcontrollerが
             [sharp, mid, gentle]の混合比を出力
```

入力を`x`、3本のFIRを`h_k`、各FIRの出力を`y_k`とすると、処理は次のように表せます。

```text
x_up[n] = x[n/2]  (nが偶数)、0 (nが奇数)
y_k[n]  = (h_k * x_up)[n]
w_k[m]  = softmax(controller(x))[k, m]
y[n]    = Σ_k w_k[n] y_k[n]
```

`m`はaudio sampleより粗いcontrol frameです。frame単位の重み`w_k[m]`を出力sample数まで線形補間してから、3本のFIR出力へ適用します。softmaxを使うため、常に次の制約が成立します。

```text
w_k >= 0
Σ_k w_k = 1
```

つまりcontrollerは、検証済みFIRが作る範囲の外へ自由な係数を出せません。学習されるのは混合比を決めるcontrollerだけで、FIR kernelは学習中もcheckpoint読込後も固定です。

### 3本のFIRの役割

| prototype | 周波数・時間応答の性格 | 主に使いたい信号 |
|---|---|---|
| `sharp` | 狭い遷移帯域、90 dB設計の強いimage抑制。代わりに不連続点の前後でringingが長くなりやすい | 定常tone、noise、緩やかに変化する音 |
| `mid` | image抑制とringingの中間。48 kHz familyでは疎な過渡に対する安全側prototypeでもある | sharpとgentleの中間的な区間 |
| `gentle` | Bessel-6の緩やかな振幅応答へ合わせた101-tap FIR。image抑制は弱いがedge ringingを抑えやすい | step、square wave、クリックなどの不連続点 |

Kaiser型の`sharp`と`mid`はrate familyごとに遷移帯域を変えます。入力Nyquistが22.05 kHzのfamilyと24 kHzのfamilyでは利用できる遷移幅が異なるため、単純な周波数スケーリングはしません。一方、`gentle`は両familyとも20 kHz cutoffのBessel-6振幅応答を基準にします。

全FIRには次の構造制約があります。

- 対称kernelによるlinear phase
- 低域基準周波数で2倍の補間gainへ正規化
- 短いkernelを左右対称にzero paddingし、最長kernelと中心を一致
- 全prototypeで共通のgroup delay

zero-stuffするとbaseband振幅が1/2になるため、FIR側は補間率2のgainを持ちます。また、全FIRの位相と中心が一致しているので、混合比を変えても異なる遅延の波形同士を足して位相キャンセルを起こす構成にはなりません。

### Controllerが混合比を決める方法

controllerは入力波形を直接受け取る5段の小さな1次元畳み込みencoderです。各段で時間方向をdownsampleし、合計stride 64、すなわち入力64 samplesごとに3個のlogitを出します。logitへsoftmaxを適用したものが`sharp`、`mid`、`gentle`の重みです。

判定が音量へ依存しないよう、controllerへ入れる波形だけをchunk内peakで正規化します。3本のFIRは正規化前の入力を処理するため、controller用の正規化が出力音量を変えることはありません。chunk端は実波形をreflect paddingしてcontrollerの受容野を満たし、定常信号の端だけ誤って`gentle`へ寄ることを防ぎます。

初期混合比は`sharp: 0.85 / mid: 0.10 / gentle: 0.05`です。この固定biasから開始し、学習可能なencoderとhead weightが信号内容に応じた差分を作ります。bias自体は固定し、学習初期に常時sharpまたは常時gentleへ飽和してsoftmax勾配を失うことを防ぎます。

学習controllerに加えて、見逃しやすい波形を決定論的に保護する2つのguardがあります。

- sparse transient guard: 局所RMSに対するcrest factorが大きいクリックや単発impulseを検出し、その周辺を安全側prototypeへ寄せます。44.1 kHz familyでは`gentle`、48 kHz familyでは`mid`を使用します。
- discontinuity guard: 大きなsample差分と平坦なplateauの密度を組み合わせ、stepやsquare waveのedgeを検出して`gentle`へ寄せます。単なるGaussian noiseを不連続信号と誤認しないため、傾きだけでは判定しません。

guardもhard switchではなく、検出scoreが0から1へ上がるにつれて通常のcontroller重みから安全側prototypeへ凸結合します。

### 何を学習しているか

教師波形はtarget rateで生成した帯域制限済み信号です。そこから正確な2:1 decimationで入力を作るため、controllerが入力Nyquistより上の未知成分を推測する教師にはなっていません。損失関数は次の役割を分担します。

- waveform L1 / multi-resolution STFT: 帯域制限済み教師との基本的な忠実度
- plateau ripple: stepやsquare waveの平坦部に残る最悪付近のripple
- quiet energy: impulse前後など、本来無音の領域へ漏れるpre/post-echo
- edge ringing: `gentle`より悪化したedge近傍のripple
- prototype selection: 既知の不連続区間では`gentle`、定常区間では強いimage抑制側を選ぶ教師信号
- weight total variation: 混合比を時間方向に滑らかにし、急激な重み変化によるmodulationを抑える
- entropy floor: 完全なone-hotへ早期飽和して学習不能になることを防ぐ

この設計では、ニューラルネットワークは音声sample、スペクトルmask、FIR係数を生成しません。出力に含まれる候補波形は常に固定FIRで入力から計算され、機械学習は「現在の区間ではimage抑制と時間応答のどちらをどの程度優先するか」だけを制御します。最終的な安全性は学習lossではなく、両rate familyのcanonical/held-out probeに対するworst-case gateで判定します。

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

48 kHz family:

```bash
uv run python scripts/train_capb.py \
  --data-config configs/data_generation_capb_48k.yaml \
  --config configs/training_stage1_capb_48k.yaml
```

学習データはtarget rateでネイティブ合成し、入力Nyquist未満へlinear-phase brickwall FIRで帯域制限した後、正確に2:1 decimationして入力を作ります。したがって`source == target[::2]`が成立し、旧Bessel劣化経路は学習に入りません。

学習されるのは約10万parameterのコントローラだけです。FIR prototype bankは固定され、checkpointにはcontroller stateとrate family metadataが保存されます。

### 学習済み成果物

再現確認用に、各rate familyの最新学習候補と対応する学習履歴・probe gate reportを同梱しています。

| rate family | checkpoint | spec v3結果 | 状態 |
|---|---|---|---|
| 44.1→88.2 kHz | `data/checkpoints/capb/run12_context_retrain/capb_best.pt` | 全gate合格 | family候補 |
| 48→96 kHz | `data/checkpoints/capb_48k/run5_context_retrain/capb_best.pt` | G3 image peak不合格 | 未採用・診断用 |

対応する記録は`reports/capb_training/`と`reports/probe_gates/`にあります。CAPBのリリース条件は両familyの全gate合格なので、これらを組み合わせたリリースcheckpoint pairはまだ確定していません。

## 受入評価

CAPB checkpointは、44.1→88.2 kHzと48→96 kHzの両familyでcanonical/held-out probeをすべて通過するまでリリースできません。

```bash
# 44.1 kHz family
uv run python scripts/evaluate_probe_gates.py \
  --backend capb \
  --checkpoint data/checkpoints/capb/run12_context_retrain/capb_best.pt \
  --rate-family 44k1

# 48 kHz family
uv run python scripts/evaluate_probe_gates.py \
  --backend capb \
  --checkpoint data/checkpoints/capb_48k/run5_context_retrain/capb_best.pt \
  --rate-family 48k
```

主なgate:

- square/step: plateau RMS、P2P、overshootをBessel基準以下に保つ
- impulse/impulse train/tone burst: pre-echoの増加を制限する
- sweep/pink noise/multitone: image bandとsweep ridgeを`-65 dB`以下にする
- 100 Hz–20 kHz: flatness、gain、phase、group delay、waveform errorを制限する
- no-added-HF: Bessel基準に対する不要な高域増加を制限する

レポートは`reports/probe_gates/<label>/gate_report.json`と`gate_report.md`へ出力されます。平均値は診断用途のみで、各gateは最悪probeにbindします。

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
  checkpoint_path: data/checkpoints/capb/run12_context_retrain/capb_best.pt
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
  data_generation_capb_48k.yaml
  training_stage1_capb.yaml
  training_stage1_capb_48k.yaml
  stage1_stage2_pipeline.yaml
src/totton_audio_de_mirroring/
  data/capb_dataset.py
  data/probe_generators.py
  data/reference.py
  models/capb.py
  models/proto_bank.py
  training/capb_losses.py
  training/capb_trainer.py
  evaluation/gates.py
  evaluation/probe_suite.py
  inference/pipeline.py
  stage2/
scripts/
  train_capb.py
  evaluate_probe_gates.py
  run_capb_phase0.py
```

## 非目標

- 22.05 kHzまたは24 kHzを超える原音成分の復元
- GAN等による倍音・超音波成分の生成
- 平均metricだけによるcheckpoint採用
- probe gateを通していない学習loss改善の品質主張

このプロジェクトの優先順位は、帯域を広く見せることではなく、入力にない情報を作らず、イメージ成分と時間領域artifactを測定可能な制約内に収めることです。
