# 音源ごとのprototype選択

推奨checkpoint（run16、44.1 kHz系列と48 kHz系列）でcontrollerが実際に選ぶ混合比です。CAPB全体の概要は[README](../README.md)の図を参照してください。

![音源ごとの混合比](images/routing_by_source.png)

左は定常フレーム（過渡強度が下位50%）、右は過渡強度が上位5%のフレームの平均です。いずれも無音フレームを除いています。実録音は先頭8秒をモノラル化して入力しました。

## 読み取り

- 定常toneとpink noiseは`sharp` 100%です。矩形波は`gentle` 100%、孤立clickは`mid` 0.69 / `gentle` 0.31です。
- 実録音の定常部は`sharp`が79〜99%を占めます。
- 打楽器系（ドラム、ハイハット、フォーリー）の過渡フレームでは`gentle`が42〜60%まで上がります。
- 水音やアイスキューブのように打点が緩い素材は、過渡フレームでも`sharp`が77〜85%を保ちます。

## 図の再生成

実録音は評価専用の素材で、リポジトリには含めません。対象ファイル名は`scripts/plot_readme_routing_figures.py`の`REAL_SOURCES`にあります。

```bash
uv run python scripts/plot_readme_routing_figures.py \
  --audio-dir <評価用録音のディレクトリ> \
  --output-dir docs/images
```

## さらに詳しい測定

| 内容 | 場所 |
|---|---|
| sweep spectrogram、THD、IMD、AMサイドバンド | `reports/release/run16_v5b_midflat_g03_20260903/visualization/distortion/` |
| impulse応答と長い尾 | `reports/release/run16_v5b_midflat_g03_20260903/visualization/impulse/` |
| gate結果一式 | `reports/release/run16_v5b_midflat_g03_20260903/gates/` |
