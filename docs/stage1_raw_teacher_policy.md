# Stage1 Raw Teacher Policy (EPIC #103 / Issue #111)

## Purpose

Stage1教師方針を `raw 88.2kHz` に統一し、実験運用での誤読と成果物混在を防止する。

## Policy

1. Default teacher is `raw88` (raw 88.2kHz reference path).
2. Legacy `bessel` teacher remains valid only for baseline comparison.
3. 0-20kHz preservation, mirror suppression, and 20-44kHz energy-cap safety gates are mandatory regardless of teacher type.

`raw88` (`raw_88k2`) の生成経路は「44.1kHzを88.2kHzへ補間」ではなく、88.2kHzでネイティブ生成した教師信号を基準とする。
そのうえで入力側は教師信号を44.1kHzへダウンサンプルし、劣化SRC経路で `x_full` を合成する。

学習ターゲット（HB）は raw 教師の波形をそのまま回帰するのではなく、劣化入力 HB の位相を保持したまま
「教師 HB の振幅スペクトルを入力に投影（入力振幅を上限にクランプ）」して構成する。
これにより、入力に存在しない高域成分を“加算で生成”することを避けつつ、ミラー除去で生じやすい
鋭いノッチ（ギブス由来のリンギング）を最小化する方向へ学習が誘導される。

また mirror 検出マスクは教師 HB ではなく劣化入力の HB から算出し、preserve 損失がミラー成分を
誤って保護して学習を阻害しないようにする。

## Baseline Positioning

- `raw88`: production-direction teacher policy for Stage1.
- `bessel`: historical baseline for A/B comparison and regression context.
- Reports must explicitly state comparison as `raw88 vs bessel` under matched conditions (seed/config/gates).

## Experiment Naming Convention

Use this run ID format:

- `stage1_<teacher>_<model>_<yyyymmdd>_s<seed>`

Examples:

- `stage1_raw88_nmse_20260210_s1234`
- `stage1_bessel_nmse_20260210_s1234`

`teacher` must be one of:

- `raw88`
- `bessel`

## Artifact Storage Convention

Store artifacts under teacher-scoped directories:

- Checkpoints: `data/checkpoints/stage1/<teacher>/<run_id>/`
- Reports: `reports/stage1/<teacher>/<run_id>/`
- Candidate outputs for fixture updates:
  `reports/stage1/<teacher>/<run_id>/candidate_outputs/`

Recommended minimum metadata in `run_manifest.json` (or equivalent):

- `teacher_type`
- `teacher_tag`
- `run_id`
- `seed`
- `config_hash`
- `checkpoint_paths`
- `gate_thresholds`

Reference workflow:

- Training/eval run: `scripts/run_issue63_stage1_workflow.py`
- Matched-condition comparison report: `scripts/report_raw_teacher_comparison.py`

## Migration Checklist

- [ ] `README.md`, `CLAUDE.md`, and `AGENTS.md` state that `raw88` is default and `bessel` is baseline-only.
- [ ] Stage1 run IDs include teacher type and avoid ambiguous naming.
- [ ] `checkpoint_dir` / `report_dir` use teacher-scoped paths.
- [ ] Comparison reports use matched conditions across `raw88` and `bessel`.
- [ ] PR includes gate outcomes (hard/mirror/ringing/IMD) and rationale when updating fixtures.
- [ ] Open implementation issues are tracked and linked: #104, #105, #106, #107, #108, #109, #110.
