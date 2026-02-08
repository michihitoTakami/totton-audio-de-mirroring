# totton-audio-de-mirroring
# Hybrid Neural Bessel SR (HNB-SR) – Updated Specification (Mirror-Removal & Time-Response Preservation)

Target Platform: **Jetson Orin Nano (8GB)**
Target Output: **705.6kHz (16× Upsampling)**
Input: **44.1kHz / 16bit or 24bit PCM**
Latency: **数秒オーダー許容（非リアルタイムでも可）**

---

## 0. Design Intent / Success Criteria

本システムの狙いは「超高域の積極的生成」ではなく、**リンギングを増やさずに時間応答（過渡・位相・群遅延）を維持しつつ、折り返し（ミラー）由来の聴覚的不自然さを除去する**ことに置く。
44.1kHz入力から22.05kHz超の成分は一意に復元できないため、20kHz以上帯域は「復元」ではなく、**不自然成分の抑制と安全な整形（必要ならゼロでも可）**として扱う。

### Hard Requirements（満たせない場合は失敗）

1. **0–20kHzは入力と同一**（波形・位相・群遅延の改変禁止）
2. **折り返し（ミラー）パターンを抑制**し、可聴上の“デジタル臭さ/ジャリつき”を低減
3. **20–44kHzはゼロ近傍でもOK**（無理に倍音を作らない）
4. 20–44kHzの**高域総エネルギーは固定上限（energy cap）**を常に遵守（IMD安全側）
5. **リンギング回帰を禁止**（矩形波プローブで before 比悪化を許容しない）

### Stage1 Quantitative Acceptance Criteria

以下の定量基準を満たさない Stage1 チェックポイントは不採用とする。

1. **Mirror Reduction**: `symmetry_reduction_ratio >= 0.70`
2. **Energy Cap**: `hb_energy_cap_violation_rate == 0.0`
3. **Ringing Regression Gate**（square-wave, edge-aligned）:
   * `plateau_ripple_rms_after / before <= 1.10`
   * `plateau_ripple_p2p_after / before <= 1.10`
   * `overshoot_abs_after - overshoot_abs_before <= 5e-3`
   * `ringing_ratio_after - ringing_ratio_before <= 0.0`

---

## 1. System Overview (Two-Stage Hybrid)

本システムは2段構成とする。

* **Stage 1: Neural Mirror Suppression Engine (NMSE)**
  44.1kHz → 88.2kHz（2×）
  目的：

  * **0–20kHz完全保持**（構造で保証）
  * 20–44kHzに存在する**折り返し/ミラー由来の不自然成分を検出・抑制**
  * **高域エネルギーを固定上限で管理**しIMDリスクを抑える

* **Stage 2: DSP High-Rate Interpolation Engine (HIE)**
  88.2kHz → 705.6kHz（8×）
  目的：

  * Stage 1で得られた“安全な88.2kHz信号”を高レート化し、アナログLPF設計を容易にする
  * 時間応答を壊しにくい補間（最小位相寄り・緩スロープ）

---

## 2. Stage 1: Neural Mirror Suppression Engine (44.1kHz → 88.2kHz)

### 2.1 Core Strategy: Band Split + Low-Band Bypass

Stage 1は全帯域生成を行わず、**帯域分割**して低域を完全バイパスする。

* `x_full`: 44.1kHz入力を基準SRCで88.2kHzへ2×アップサンプルした信号
* `LB_in = LPF(20kHz, x_full)`（0–20kHz）
* `HB_in = HPF(20kHz, x_full)`（20–44.1kHz）

出力は

* `LB_out = LB_in`（固定、改変不可）
* `HB_out = Suppress(HB_in)`（AIが抑制）
* `y_full = LB_out + HB_out`

> LBの同一性は“損失で祈る”のではなく、**構造で保証**する。

### 2.2 Model Output: Suppression Mask (Recommended)

AIは高域を生成するのではなく、**抑制マスク（ゲイン）**を推定する。

* 出力：`M ∈ [0, 1]`（時間-周波数マスク、または時間領域ゲイン系列）
* 適用：`HB_out = HB_in ⊙ M`

意図：

* 折り返しパターンを含む成分を強く抑制
* それ以外は保持
* 必要ならHBはゼロに近くても正解（“創作”を避ける）

### 2.3 Representation Options

本仕様では以下のいずれかで実装可能（推奨はA）。

* **A. STFT Masking (推奨)**

  * HBのみSTFT → マスク推定 → iSTFTでHB_out再合成
  * 折り返しの幾何学パターン（帯状・対称性）に対して学習が安定
* **B. Time-Domain Gain Control**

  * HBを時間領域で直接抑制（TCN等）
  * 周波数選択性を獲得しづらく、学習難度は上がる

### 2.4 Fixed Safety Constraints (Post-Processing)

ネット出力の後に必ず以下を適用し、一般化と安全性を担保する。

1. **Energy Cap（固定上限）**

   * 20–44kHzの総エネルギーが上限を超えたらスケーリング/クランプ
2. **Envelope Target（固定包絡）**

   * 20kHz以降が“なだらかに減衰する”形状へ投影（過剰な山を抑える）
3. **DC/Leak対策**

   * HBがLB側へ漏れないようHPF側で再確認（境界漏れの抑制）

---

## 3. Dataset Pipeline (Ideal Master Not Required)

理想マスターを前提にせず、**合成データ生成と規格化ターゲット**で学習を成立させる。

### 3.0 Stage1 Input/Target Path Spec（固定仕様）

`configs/data_generation.yaml` の `stage1_path` は、実装経路と1:1で対応する固定仕様とする。

* `input_route = source_chunk_44k1_to_x_full_88k2_via_degradation`
  * 意味: 44.1kHzチャンク (`source`) を劣化SRC経路で2xして `x_full` を作る
* `target_route = high_band_to_hb_target_via_mirror_detection`
  * 意味: `high_band = HPF(20kHz, x_full)` から mirror検出+抑制で `hb_target` を作る
* `strict_route_validation = true`
  * 意味: Stage1の学習データ経路を `44.1kHz -> 88.2kHz (2x)` に固定する

この仕様は `MirrorSuppressionDataset` で検証され、経路が一致しない設定はエラーとする。

### 3.1 Source Material (Synthetic / Procedural)

学習に用いる“元音源”は実音源でなくてもよい。一般化目的のため、以下を広く混ぜる：

* マルチトーン（和音/非整数倍を含む）
* 周波数スイープ、インパルス列、パーカッション風トランジェント
* AM/FM変調、ノイズ（白色/ピンク/帯域ノイズ）
* クリップ/ソフトサチュ等の軽微非線形を含む波形（任意）

### 3.2 Degradation Diversity (Key to Generalization)

同一のSourceから、異なる劣化SRCをランダム適用して `x_full` を作る（過適合防止）。

例（混合セット）：

* ZOH / 線形補間 / 短・長窓sinc / IIR系（Bessel/Butter）等
* カットオフ：18–22kHzでランダム
* 位相：線形位相/最小位相/アナログ風群遅延
* 量子化：16/24bit、複数ディザ

### 3.3 Training Target (Normalized “Anti-Mirror” Target)

全帯域の「理想クリーン波形」を教師にせず、**折り返し由来の不自然成分のみ抑制したHBターゲット**を作る。

* まず `HB_in` を得る
* ルールベース/解析ベースで **折り返し特徴を持つ成分**を検出して減衰し、`HB_target` を作る
* 最後に `HB_target` に対して **Energy Cap / Envelope Target** を適用して規格化する

注意:
`HB_target` は「高域抑制のための規格化ターゲット」であり、全帯域の物理的グラウンドトゥルースではない。
0–20kHzの同一性は `LB_out = LB_in` の構造保証で担保する。

このとき、学習は

* 「折り返しっぽい形だけを消す」
* 「それ以外は触らない」
  に寄るため、一般化しやすい。

---

## 4. Loss Strategy (Mirror Removal Oriented)

GANで質感生成を狙うのではなく、**抑制の正確さと“不用意に触らないこと”**を中心にする。

* `L_mask`: マスク/抑制量の教師（HB_target）への一致（L1/L2）
* `L_stft`: MR-STFT（HBのみ、構造変化の最小化）
* `L_preserve`: “触りすぎ”罰則（HB_inとHB_outの差分に対する正則化、ただしミラー検出領域は例外）
* `L_energy`: energy cap違反を強罰（固定上限の厳守）

※ Adversarial（GAN）は原則使用しない（必要なら最終調整で弱く導入可）。

---

## 5. Stage 2: DSP High-Rate Interpolation Engine (88.2kHz → 705.6kHz)

### 5.1 Multi-Stage Upsampling

8×を一発でなく、**2××2××2×**の多段とする。

### 5.2 Filter Policy (Time Response Priority)

* 推奨：**最小位相寄りFIR（短め・緩スロープ）** または **低次IIR多段**
* 仕様上の目標：

  * 可聴帯過渡の悪化を最小化
  * 測定で“プリエコー相当”が無視できるレベル

### 5.3 Cutoff

* Stage 1でHBは安全側に整形されている前提のため、Stage 2では急峻カットを不要とし

  * **40–50kHz付近を緩やかに通す**設計を基本とする（段ごとに最適化可）

---

## 6. Expected Output Characteristics

* **0–20kHz**：入力同一（波形・位相・群遅延）
* **20–44kHz**：折り返し/ミラー由来の不自然成分が抑制され、必要ならゼロ近傍
* **>44kHz**：ノイズ床へ自然減衰
* **聴感**：高域のジャリつき/金属感/不自然なザラつきが減り、過渡が鈍らない

---

## 7. Evaluation (PoC Minimum Set)

### 7.1 Hard Metrics

1. **LB差分（0–20kHz）**：振幅/位相誤差が測定限界近傍
2. **Mirror Pattern Reduction**：STFTで折り返し特徴（対称性・帯状成分）が低減
3. **HB Energy Cap**：20–44kHzエネルギーが常に固定上限以下
4. **Touch-Minimization**：ミラー以外のHBが不要に変形していない

### 7.2 IMD Proxy (Recommended)

* 簡易非線形（軽いクリップ/2次歪等）通過後の可聴帯ノイズ/歪みが増えないこと

### 7.3 Listening

* ABXで“ジャリつき/刺さり/金属感”が減り、アタックが維持されること
* 実施手順・記録様式は `docs/abx_listening_protocol.md` を使用する

### 7.4 Automated Stage 1 Hard Metrics

`scripts/evaluate_stage1.py` で、README 7.1 のHard Metricsを自動評価する。

```bash
uv run python scripts/evaluate_stage1.py \
  --input-dir data/eval/stage1/input \
  --output-dir data/eval/stage1/output \
  --sample-rate 88200 \
  --energy-cap 1e-3 \
  --json reports/stage1_metrics.json \
  --csv reports/stage1_metrics.csv
```

主な出力指標:

1. LB振幅差分（0–20kHz）
2. LB位相差分（0–20kHz）
3. LB群遅延差（0–20kHz）
4. Mirror低減率（STFT対称性ベース）
5. HB energy cap違反率
6. Touch指標（非ミラーHB変形量）

`--json` 出力には `gates` オブジェクトが含まれ、以下の判定根拠（threshold / observed / passed）を追跡できる:

1. `energy_cap`
2. `mirror_reduction`
3. `ringing_regression`

strict判定時の終了コードは固定:

1. `2`: energy cap gate fail (`--strict-energy-cap`)
2. `3`: mirror reduction gate fail (`--strict-mirror-reduction`)
3. `4`: ringing regression gate fail (`--strict-ringing-regression`)
4. `5`: strict複数指定時に2つ以上 fail

回帰テスト（golden samples）:

```bash
uv run --extra dev pytest tests/regression/test_stage1_regression.py -v
```

### 7.5 Issue #63 Workflow (Retrain + Checkpoint Selection)

`scripts/run_issue63_stage1_workflow.py` は以下を一括実行する。

1. 学習条件固定（config hash / seed / gate 設定を `run_manifest.json` へ保存）
2. `scripts/train_stage1.py` による Stage 1 再学習
3. Hard Metrics / Mirror Metrics 評価
4. IMD proxy（naive vs NMSE）比較
5. hard + mirror + IMD + ringing gate通過候補からベストcheckpoint選定とレポート保存

```bash
uv run python scripts/run_issue63_stage1_workflow.py \
  --data-config configs/data_generation.yaml \
  --train-config configs/training_stage1.yaml \
  --eval-input-dir tests/fixtures/golden_samples/stage1/input \
  --imd-naive-dir tests/fixtures/golden_samples/imd/naive \
  --checkpoint-dir data/checkpoints/issue63 \
  --report-dir reports/issue63 \
  --seed 1234
```

主な成果物:

* `reports/issue63/run_manifest.json`
* `reports/issue63/selected/selection_report.json`
* `reports/issue63/selected/stage1_best_selected.pt`

### 7.6 Issue #64 Workflow (Freeze Golden / Regression / ABX Pairs)

Issue #63で確定したcheckpointを基準に、回帰/ABX基準を固定化する。

```bash
uv run python scripts/run_issue63_stage1_workflow.py \
  --data-config configs/data_generation.yaml \
  --train-config configs/training_stage1.yaml \
  --eval-input-dir tests/fixtures/golden_samples/stage1/input \
  --imd-naive-dir tests/fixtures/golden_samples/imd/naive \
  --checkpoint-dir data/checkpoints \
  --report-dir reports/issue64 \
  --seed 1234 \
  --device cuda \
  --energy-cap 1e-3 \
  --skip-training \
  --candidate-checkpoints stage1_best.pt stage1_last.pt stage1_emergency.pt
```

固定化対象（リポジトリ追跡）:

* `tests/fixtures/golden_samples/stage1/output/*.npy`
* `tests/fixtures/golden_samples/imd/nmse/*.npy`
* `tests/fixtures/golden_samples/regression_baseline.json`
* `tests/fixtures/golden_samples/abx_pairs.json`
* `tests/fixtures/golden_samples/issue64_model_selection.json`
* `docs/abx_listening_protocol.md`

### 7.7 Issue #75 Workflow (Frequency/THD+N/Time-Domain Visualization)

Issue #75では、Hard Metricsの数値だけでなく、周波数応答・THD+Nスペクトル・時間領域応答を画像で確認できるようにする。

```bash
uv run python scripts/visualize_audio_quality.py \
  --input-dir tests/fixtures/golden_samples/stage1/input \
  --output-dir tests/fixtures/golden_samples/stage1/output \
  --visual-dir reports/issue75/visualizations \
  --sample-rate 88200 \
  --n-fft 8192 \
  --cutoff-hz 20000 \
  --num-taps 1025 \
  --summary-json reports/issue75/visualization_summary.json
```

主な成果物:

* `reports/issue75/visualizations/*_frequency_response.png`
* `reports/issue75/visualizations/*_thdn_spectrum.png`
* `reports/issue75/visualizations/*_waveform_comparison.png`
* `reports/issue75/visualizations/*_square_wave.png`
* `reports/issue75/visualizations/*_impulse_response.png`
* `reports/issue75/visualization_summary.json`

---

## 8. Jetson Orin Nano Implementation Notes

### 8.1 Training Device Policy (Stage 1)

* **Stage 1学習はGPU実行を必須**とする（CUDA利用可能環境ではCPU学習を許可しない）
* 学習ログに`device`（GPU名/CUDA index）を必ず記録し、再現性を担保する
* CPU実行はデバッグ用途の最小確認のみに限定し、学習結果の評価対象に含めない

### 8.2 Inference / Deployment Notes

* **Chunk Processing**：数秒遅延許容のため、1–4秒程度のチャンク推論を採用可
* **Boundary Handling**：overlap-add / crossfade（HB側で実施）
* **Optimization**：TensorRT（FP16推奨）
* Stage 2は軽量のためCPU/GPUいずれでも可（全体最適で選択）

### 8.3 Stage1->Stage2 統合CLI

```bash
uv run python scripts/run_stage1_stage2_pipeline.py \
  --config configs/stage1_stage2_pipeline.yaml \
  --input-wav path/to/input_44k1.wav \
  --output-wav path/to/output_705k6.wav
```

ベンチマーク/仕様詳細は `docs/stage1_stage2_pipeline_integration.md` を参照。

---

## 9. Implementation Roadmap

1. **Phase 1：合成データ生成基盤**

   * 多様な劣化SRCの実装とサンプリング
   * Mirror検出→抑制→規格化（HB_target生成）
2. **Phase 2：Stage 1 PoC（Masking）**

   * LB同一性、ミラー低減、energy cap遵守を評価
3. **Phase 3：Stage 2統合（2××2××2×）**

   * 過渡保護・IMD proxyで確認
4. **Phase 4：Jetson実装**

   * TensorRT化、チャンク長/オーバーラップ最適化、スループット測定

---

## Explicit Statement of Scope (Non-Hallucination by Design)

* 本システムは **“超音波の創作”を目的としない**。
* 20–44kHzは **折り返し不自然成分の除去と安全整形**を主目的とし、必要ならゼロでも正しい。
* 可聴帯（0–20kHz）の時間応答・位相保護を最優先とする。
