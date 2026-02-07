# totton-audio-de-mirroring
# Hybrid Neural Bessel SR (HNB-SR) – Updated Specification (Mirror-Removal & Time-Response Preservation)

Target Platform: **Jetson Orin Nano (8GB)**
Target Output: **705.6kHz (16× Upsampling)**
Input: **44.1kHz / 16bit or 24bit PCM**
Latency: **数秒オーダー許容（非リアルタイムでも可）**

---

## 0. Design Intent / Success Criteria

本システムの狙いは「超高域の積極的生成」ではなく、**時間応答（過渡・位相・群遅延）を守りつつ、折り返し（ミラー）由来の聴覚的不自然さを除去する**ことに置く。
44.1kHz入力から22.05kHz超の成分は一意に復元できないため、20kHz以上帯域は「復元」ではなく、**不自然成分の抑制と安全な整形（必要ならゼロでも可）**として扱う。

### Hard Requirements（満たせない場合は失敗）

1. **0–20kHzは入力と同一**（波形・位相・群遅延の改変禁止）
2. **折り返し（ミラー）パターンを抑制**し、可聴上の“デジタル臭さ/ジャリつき”を低減
3. **20–44kHzはゼロ近傍でもOK**（無理に倍音を作らない）
4. 20–44kHzの**高域総エネルギーは固定上限（energy cap）**を常に遵守（IMD安全側）

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

「正解HB」を真値として与えず、**折り返し由来の不自然成分のみ抑制したターゲット**を作る。

* まず `HB_in` を得る
* ルールベース/解析ベースで **折り返し特徴を持つ成分**を検出して減衰し、`HB_target` を作る
* 最後に `HB_target` に対して **Energy Cap / Envelope Target** を適用して規格化する

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

回帰テスト（golden samples）:

```bash
uv run --extra dev pytest tests/regression/test_stage1_regression.py -v
```

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
