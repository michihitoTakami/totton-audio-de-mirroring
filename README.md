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
