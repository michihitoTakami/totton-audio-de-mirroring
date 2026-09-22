# Transient dwell projection — 実験記録（2026-09-22）

`sharp` と `gentle` を直接混ぜず `mid` を経由させる「遷移順序の強制」を検討し、
softmax 後の決定的な射影として実装した記録です。Issue #175。

## 問い

Sharp → Gentle の直行を禁止して必ず Mid を挟むと特性が良くなるか。逆向き
（Gentle → Mid → Sharp）だけを強制する非対称案、その両方を強制する案も含めて検討した。

## 物理

混合出力は各 prototype 出力の凸結合なので、周波数応答も impulse 応答も重みに線形である。
release bank（`v5b_sharp1023_midflat70`、44.1 kHz 系列）で測ると性質はきれいに分離する。

| 混合点 | 20 kHz 通過 | 15〜20 kHz 成分のイメージ | 中心 ±0.5 ms 外の最大リンギング |
|---|---:|---:|---:|
| sharp | 0.00 dB | -157 dB | -37.6 dB |
| mid | 0.00 dB | -70 dB | -70.3 dB |
| gentle | -10.1 dB | -19 dB | なし |
| sharp 0.5 / gentle 0.5（辺上） | -3.66 dB | -25 dB | -42.5 dB |
| sharp .45 / mid .05 / gentle .50（run16 のドラム過渡） | -3.66 dB | -25 dB | -43.4 dB |
| sharp .30 / mid .40 / gentle .30 | -2.01 dB | -30 dB | -47.4 dB |

高域の落ち込みとイメージ漏れは `gentle` の重みだけで決まり、長い尾のリンギングは `sharp`
の重みだけで決まる。`mid` はどちらにも寄与しない。sharp–gentle の辺上にいる限り両方の
欠点を同時に払っている。

リンギングの長さも桁が違う。

| prototype | 台 | -60 dB まで | -100 dB まで |
|---|---|---|---|
| sharp | ±5.79 ms | 2.1 ms | 4.4 ms |
| mid | ±0.54 ms | 0.44 ms | 0.54 ms |
| gentle | ±0.46 ms | 0.09 ms | 0.18 ms |

したがってエッジの周囲で `sharp` を消すべき窓は ±5.8 ms、`mid` の Gibbs リップルを消すべき窓は
±1 ms 弱である。`gentle` が必要なのは中心 ±1 ms だけで、1〜5.8 ms は `mid` で受けられ、
5.8 ms より外は何も守る必要がない。

## run16 の実際の軌道

- DC step / impulse の保護窓は ±26 ms（36 フレーム）。sharp の台の 4 倍以上で、幅を決めて
  いるのは prior の 25 ms RMS 平滑化である。外周部は DC step では sharp 0.45 / gentle 0.55
  の辺上、impulse では gentle 1.0 に張り付く。
- ハイハットでは打撃の瞬間に gentle+mid を入れた後、打撃の 25〜30 ms 後の減衰部にもう一度
  gentle 100% の谷を作る。ここにエッジはない（prior の offset 検出の誤反応）。
- L1 の TV 損失では sharp→gentle 直行の総変動は 2、mid 頂点経由は 4 なので、学習は辺経路を
  積極的に好む。

## 学習なしの射影実験

run16 の重みへ softmax 後の射影をかけ、凍結 prototype 出力で再合成した。

- **kappa1**: `w_mid ≥ min(w_sharp, w_gentle)` を強制（状態なしの辺除外）
- **dwell_mid**: 短窓イベント検出から ±1.5 ms は controller の混合を保持、それ以外は gentle 質量を mid へ
- **dwell_tight**: 同上、ただし sharp の台 5.8 ms の外は sharp へ戻す
- **asym**: 接近側は dwell_tight、復帰側は 2.9 ms 以降で sharp へ早戻し

全変種が両系列で G1〜G9 を通過した。16〜20 kHz（all-sharp 描画基準、先頭 8 秒モノラル）:

| 素材 | run16 | kappa1 | dwell_mid / tight / asym |
|---|---:|---:|---:|
| ハイハット shuffle 863823 | -3.20 dB | -3.16 | **-2.62** |
| among-us hihat 863856 | -2.12 dB | -1.84 | **-0.98** |
| kitchen foley 864857 | -2.42 dB | -2.32 | **-1.73** |
| foley drum 864855 | -1.47 dB | -1.39 | **-1.07** |
| glitch drum 863857 | -1.08 dB | -1.03 | -0.94 |
| pop beat 684120 | -0.56 dB | -0.49 | -0.47 |
| microtonic 865041（48k） | -0.74 dB | -0.67 | **-0.33** |

イメージ帯は 0.2〜4.9 dB 改善し、G5 impulse 列の利得誤差は 0.375 → 0.164 dB になった。
氷と水の録音は 16 kHz 以上が -126 dBr で存在しないため除外した。

読み取り:

- 「sharp→gentle 禁止で mid 経由」は効く。効果の主因は遷移経路そのものではなく、run16 が
  減衰部に作る不要な gentle 区間が mid か sharp に置き換わることである。
- 「gentle→mid→sharp の復帰強制」は高域では何も変わらない。mid も sharp も 20 kHz まで
  平坦なので、復帰先の違いはイメージ帯と G5 にしか現れない。非対称規則を独立に追う理由はない。
- kappa1 は gentle 質量の削減が小さく 0.05〜0.3 dB。
- 残る -2.6 dB は打撃の瞬間（±1.5 ms）を gentle にするか mid にするかの問題で、
  `docs/hf_onset_routing_experiment.md` が扱った到達点の問題である。

## 実装

`models/transient_dwell.py`。`CAPB.controller_weights` の softmax 直後に適用するので、
波形 ONNX と controller-only ONNX の両方に同じ演算が入る。損失ではなく射影にしたのは、
HF-onset 実験で「prior は正しく、NN が矩形エッジへ過剰般化して壊す」を 3 試行で確定させた
ためである。

1. **イベント検出**（フレーム = 64 samples、中心 65 sample 窓）
   - 微分 crest（フレーム内 max|x'| / rms(x')）> 4: step、click、疎な矩形エッジ。
     ノイズは 2.5〜3.5、矩形は 5.7〜8。
   - 平坦部波形: 波形の rms/peak > 0.97 かつ微分 peak > 0.5 × 波形 peak。1.2 kHz 以上の
     矩形は 1 フレームに複数エッジが入り crest が 3.3〜3.6 に落ちるため、この規則で拾う。
     矩形は 1.00、正弦は 0.71、実録音は 0.9 未満。0.5 の振幅条件で帯域制限リップル（約 0.3）と
     ノイズ付き DC 平坦部を除外する。
   - レベル変化: ±1 フレームのエネルギー差 > 6 dB、かつ当該フレームが大きい側。
     孤立 spike の隣接フレームが誤って広がらない。
2. **ゾーン**（イベントからの距離 d、フレーム単位）
   - d ≤ 1（1.5 ms）: controller の混合をそのまま保持（孤立 click の mid 0.7 も保持）
   - 1 < d ≤ 4（5.8 ms = sharp の台）: gentle 質量を mid へ
   - d > 4: 非 sharp 質量を sharp へ
3. **ramp**: 既定 0。フレーム重みはサンプルレートへ線形補間されるので、ゾーン境界は
   すでに 1.45 ms の傾斜を持つ。1 フレームの追加 ramp は高域回復を半分にした
   （ハイハット -2.61 → -2.93 dB）一方、帯域制限素材の変調床は -96 → -105 dBr にしか動かない。
   G9 は両方通過する。

周期矩形は全フレームがイベントになるので射影は恒等であり、前回 G2 を落とした「矩形エッジへの
mid 漏れ」の経路は構造的に閉じている。定数は checkpoint に保存し、旧 checkpoint は無効
（従来挙動）。ONNX parity は controller 誤差 0.0、波形 1.0e-6。

## 学習

run16 から 20 epochs / lr 1e-4、bank 変更なし、3 seed（1234 / 2026 / 4649）× 両系列。
設定は `configs/training_stage1_capb_transient_dwell_3p.yaml` と `_48k_` 版。

### 第 1 回（検出器が crest とレベル変化のみ）

5/6 が CPU・strict-FP32 CUDA の G1〜G9 を通過。44.1 kHz seed 4649 が held-out の 331 Hz 矩形で
G1 を落とした（plateau_rms 3.0e-3 / 5.0e-4）。原因は NN 自身が 442/690 フレームで sharp を
0.63 漏らしたことで、射影はそこでは恒等（全フレームがイベント）。学習矩形は 40〜5000 Hz を
覆うので分布外ではなく、seed 依存の drift である。

同時に 1730 Hz 矩形で検出器が edge を見落として sharp へ戻す穴が見つかった（gate は plateau 行を
持たないため通過）。1000〜1667 Hz の矩形では G1/G2 を落としうるので、平坦部波形の規則を追加した
（上記 1.）。

### 第 2 回（平坦部規則を追加）

**6/6 が CPU・strict-FP32 CUDA の G1〜G9（spec 7）を通過した。** リンギング gate の最悪行は
run16 と同じ probe・同じ値で、余裕は 44.1k G1 63.8% / G2 59.1%、48k G1 65.2% / G2 64.5%。
G5 の最悪は両系列とも `square_5000hz` の 0.34〜0.38 dB（run16 と同値。平坦部規則が 5 kHz 矩形を
run16 と同じ gentle に保つため）。

16〜20 kHz（all-sharp 描画基準）:

| 素材 | run16 | seed 1234 | seed 2026 | seed 4649 | 学習なし射影 |
|---|---:|---:|---:|---:|---:|
| ハイハット shuffle 863823 | -3.20 | -2.80 | -2.71 | -2.76 | -2.61 |
| among-us hihat 863856 | -2.12 | -1.34 | -1.40 | -1.35 | -1.19 |
| kitchen foley 864857 | -2.42 | -1.78 | -1.67 | -1.79 | -1.63 |
| foley drum 864855 | -1.47 | -1.23 | -1.18 | -1.28 | -1.09 |
| glitch drum 863857 | -1.08 | -1.09 | -1.06 | -1.12 | -0.88 |
| pop beat 684120 | -0.56 | -0.44 | -0.49 | -0.42 | -0.45 |
| microtonic 865041（48k） | -0.74 | -0.31 | -0.30 | -0.28 | -0.33 |

イメージ帯は 44.1k で 0.1〜0.5 dB、48k で 5〜6 dB 改善（microtonic -48.3 → -53.3〜-54.5 dB）。
fine-tune 後の NN は学習なし射影より 0.1〜0.2 dB だけ gentle 側へ戻るが、seed 間の差は 0.1 dB
以内で、第 1 回で見た drift は再現しなかった。

候補ペアは **seed 1234**（両系列。全 seed が通過し差が 0.1 dB 以内のため参照 seed を採用）。
補助検査:

| 検査 | 44.1k | 48k |
|---|---|---|
| transient robustness（64 位相 / OLA 境界） | PASS、worst -37.1 / -37.1 dB | PASS、worst -37.4 / -38.2 dB |
| ONNX parity（波形 / controller） | 1.0e-6 / 0.0 | 1.1e-6 / 0.0 |

run16 の 64 位相 worst は -37.3 / -38.4 dB で同等。

checkpoint は `data/checkpoints/experiments/transient_dwell2_{44k1,48k}_seed{1234,2026,4649}/`
（未追跡）。**推奨 checkpoint は run16 のまま**である。差し替えには release_quality、null test、
可視化を含む証跡バンドルの再生成と ABX が必要で、それは別 PR で行う。

## 判断

- 高域回復は 0.3〜0.8 dB で、run16 の ABX で識別不能だった 2.3〜2.9 dB の差の一部にとどまる。
  可聴改善を主張する根拠はない。
- 一方で構造としては正しい方向で、gate 余裕を一切失わず、イメージ帯を改善し、run16 が減衰部に
  作っていた物理的根拠のない gentle 区間を除いた。前回の HF-onset 実験が学習込みで 0.06〜0.47 dB
  だった同じ目標を、決定的な射影で 3 seed × 両系列を通して達成している。
- 残る高域差の主因は打撃の瞬間（±1.5 ms）の到達点であり、遷移経路ではない。

## 再現に必要な情報

- 基準 checkpoint: run16 両系列（`reports/release/release_manifest.json` の `recommended_checkpoints`）
- 学習: `scripts/train_capb.py --data-config configs/data_generation_capb{,_48k}_routing_v2.yaml
  --config configs/training_stage1_capb{,_48k}_transient_dwell_3p.yaml --init-checkpoint <run16>
  --seed {1234,2026,4649}`
- 評価: `scripts/evaluate_probe_gates.py --backend capb --rate-family {44k1,48k} --device {cpu,cuda}`、
  strict FP32
- 素材: `863823`（ハイハット shuffle）、`863856`（among-us hihat）、`864857`（kitchen foley）、
  `865041`（microtonic、48k）。いずれも評価専用で学習には混ぜない
