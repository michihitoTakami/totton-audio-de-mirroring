# Stage1 Data Path Spec (Issue #82)

## Purpose

Stage1 の学習データにおける `input` (`x_full`) と `target` (`hb_target`) の物理的意味を固定し、
`configs/data_generation.yaml` と実装経路の 1:1 対応を維持する。

## Fixed Definitions

1. Input (`x_full`)
   - Route ID: `source_chunk_44k1_to_x_full_88k2_via_degradation`
   - Definition: `source` (44.1kHz chunk) に劣化SRC経路を適用して `x_full` (88.2kHz) を生成する。
   - `raw_88k2` 教師時は、教師88.2kHzチャンクを44.1kHzへダウンサンプルした `source` を起点にする。
2. Target (`hb_target`)
   - Route ID: `high_band_to_hb_target_via_mirror_detection`
   - Definition: `high_band = HPF(20kHz, x_full)` から mirror 検出 + 抑制 + energy/envelope 規格化で生成する。

`hb_target` は高域抑制目標であり、全帯域クリーンGTではない。0-20kHzの同一性は
band-split 構造 (`LB_out = LB_in`) で保証する。

## Config <-> Code Mapping

`configs/data_generation.yaml`:

```yaml
stage1_path:
  input_route: source_chunk_44k1_to_x_full_88k2_via_degradation
  target_route: high_band_to_hb_target_via_mirror_detection
  strict_route_validation: true
```

`src/totton_audio_de_mirroring/data/pipeline_config.py`:
- `Stage1PathConfig` が route ID と strict mode を検証する。
- strict mode では `target_sample_rate = source_sample_rate * 2` を必須にする。

`src/totton_audio_de_mirroring/data/dataset.py`:
- `apply_degradation_profile(...)` で `x_full` を生成する。
- `generate_hb_target(...)` で `hb_target` を生成する。
- 生成サンプルに `input_route` / `target_route` を保持する。

## Regression Guard

`tests/test_dataset.py`:

1. `test_stage1_strict_path_requires_2x_ratio`
   - strict route で 2x 以外を拒否することを検証。
2. `test_stage1_path_roundtrip_in_serialized_config`
   - 設定ファイル roundtrip で route spec が保持されることを検証。
3. `test_dataset_pipeline_route_mapping`
   - `apply_degradation_profile` / `generate_hb_target` 呼び出しを監視し、
     config 値がそのまま実装経路に渡されることを検証。
