# ABX Session Summary Template

## Session Metadata

- session_id:
- date_utc:
- listener_id:
- repository_commit:
- playback_chain:
- listening_environment:
- sample_rate_hz:
- loudness_matching_method: `LUFS` or `RMS`

## Frozen Assets

- abx_pairs_json:
- selected_checkpoint_name:
- selected_checkpoint_sha256:

## Quantitative Result

- total_trials:
- correct_trials:
- hit_rate_percent:
- binomial_test_p_value:
- significance_threshold: `0.05`
- statistically_significant: `true` or `false`

## Qualitative Notes

### Harshness / Graininess

- sample_a:
- sample_b:

### Transient Attack

- sample_a:
- sample_b:

### Fatigue / Other

- sample_a:
- sample_b:

## Verdict

- README_7_3_harshness_reduced: `pass` or `fail`
- README_7_3_attack_preserved: `pass` or `fail`
- overall_status: `pass`, `fail`, or `needs_follow_up`
- follow_up_actions:
