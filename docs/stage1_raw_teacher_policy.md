# Stage1 Raw Teacher Policy (EPIC #103 / Issue #111)

## Purpose

Stage1教師方針を `raw 88.2kHz` に統一し、実験運用での誤読と成果物混在を防止する。

## Policy

1. Default teacher is `raw88` (raw 88.2kHz reference path).
2. Legacy `bessel` teacher remains valid only for baseline comparison.
3. 0-20kHz preservation, mirror suppression, and 20-44kHz energy-cap safety gates are mandatory regardless of teacher type.

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
