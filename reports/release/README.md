# CAPB release evidence — 2026-09-01

最終採用品は従来の `release_v4` です。1535/2047-tap候補は両rate family・3 seedで
controller-only FineTuningを行いましたが、44.1 kHz系列のG2b pre-echoが全seedで
不合格となったため採用していません。

| Profile | 44.1k passing | 48k passing | 44.1k G2b | 48k G2b | 判断 |
|---|---:|---:|---:|---:|---|
| `release_v4` | PASS | PASS | `1.047e-7` | `1.326e-8` | 採用維持 |
| `long_sharp_1535_a120` | 0/3 | 3/3 | `4.959e-7` | `7.913e-8` | 不採用 |
| `long_sharp_2047_a120` | 0/3 | 3/3 | `5.112e-7` | `8.492e-8` | 不採用 |

長FIRはイメージ抑制を改善しました。2047 tapsの44.1 kHz SMPTE sidebandも
`-125.60`から`-126.34 dB`へ改善しています。一方、48 kHz SMPTE sidebandは1535で
`-139.26 dB`、2047で`-136.77 dB`へ悪化し、2047は48 kHz offset robustnessも僅かに
不合格でした。sidebandは依然G9閾値を大幅に下回りますが、pre-echo hard gateを
相殺できないためfail-closedとしました。

- [選定レポート](selection/selection.md)
- [改善・悪化の比較図](selection/tradeoff_comparison.png)
- [最終release quality](release_quality/release_quality.md)
- [44.1 kHz gate](gates/44k1/cpu/candidate/gate_report.md)
- [48 kHz gate](gates/48k/cpu/candidate/gate_report.md)
- `visualization/`: 採用品の既存形式可視化
- `comparison/`: 非採用理由を示す1535/2047の比較図のみ

数値の正史はstrict-FP32のworst-probe gateです。図は診断用です。
