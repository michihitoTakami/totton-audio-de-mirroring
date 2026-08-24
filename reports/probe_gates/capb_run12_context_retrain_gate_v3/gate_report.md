# Stage 1 probe gate report

- all_passed: **True**
- spec_version: 3
- manifest_hash: c15ecd1149d409e3

## G1_lf_ringing: PASS (worst: square_1730hz_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | plateau_rms_after | 2.534e-05 | 0.0005 | absolute | PASS |
| square_50hz | canonical | plateau_p2p_after | 0.0001935 | 0.00158 | absolute | PASS |
| square_50hz | canonical | overshoot_after | 0.03068 | 0.09352 | relative | PASS |
| square_100hz | canonical | plateau_rms_after | 2.534e-05 | 0.0005 | absolute | PASS |
| square_100hz | canonical | plateau_p2p_after | 0.0001935 | 0.00158 | absolute | PASS |
| square_100hz | canonical | overshoot_after | 0.03068 | 0.09352 | relative | PASS |
| square_500hz | canonical | plateau_rms_after | 5.26e-05 | 0.0005 | absolute | PASS |
| square_500hz | canonical | plateau_p2p_after | 0.0005155 | 0.002907 | relative | PASS |
| square_500hz | canonical | overshoot_after | 0.03068 | 0.09352 | relative | PASS |
| square_1000hz | canonical | plateau_rms_after | 0.6498 | 0.7178 | relative | PASS |
| square_1000hz | canonical | plateau_p2p_after | 1.063 | 1.198 | relative | PASS |
| square_1000hz | canonical | overshoot_after | 0.03114 | 0.09367 | relative | PASS |
| square_2000hz | canonical | plateau_rms_after | 0.598 | 0.6707 | relative | PASS |
| square_2000hz | canonical | plateau_p2p_after | 1.063 | 1.295 | relative | PASS |
| square_2000hz | canonical | overshoot_after | 0.0973 | 0.1452 | relative | PASS |
| square_500hz_a005 | canonical | plateau_rms_after | 2.534e-06 | 5e-05 | absolute | PASS |
| square_500hz_a005 | canonical | plateau_p2p_after | 1.935e-05 | 0.0002907 | relative | PASS |
| square_500hz_a005 | canonical | overshoot_after | 0.003068 | 0.01385 | relative | PASS |
| dc_step_up | canonical | plateau_rms_after | 2.534e-05 | 0.0005 | absolute | PASS |
| dc_step_up | canonical | plateau_p2p_after | 0.0001935 | 0.002907 | relative | PASS |
| dc_step_up | canonical | overshoot_after | 0.03068 | 0.09352 | relative | PASS |
| dc_step_down | canonical | plateau_rms_after | 2.534e-05 | 0.0005 | absolute | PASS |
| dc_step_down | canonical | plateau_p2p_after | 0.0001935 | 0.002907 | relative | PASS |
| dc_step_down | canonical | overshoot_after | 0.03068 | 0.09352 | relative | PASS |
| square_73hz_held | held_out | plateau_rms_after | 5.26e-05 | 0.0005 | absolute | PASS |
| square_73hz_held | held_out | plateau_p2p_after | 0.0005155 | 0.002907 | relative | PASS |
| square_73hz_held | held_out | overshoot_after | 0.03068 | 0.09352 | relative | PASS |
| square_331hz_held | held_out | plateau_rms_after | 5.26e-05 | 0.0005 | absolute | PASS |
| square_331hz_held | held_out | plateau_p2p_after | 0.0005155 | 0.002907 | relative | PASS |
| square_331hz_held | held_out | overshoot_after | 0.03068 | 0.09352 | relative | PASS |
| square_1730hz_held | held_out | plateau_rms_after | 0.6353 | 0.6892 | relative | PASS |
| square_1730hz_held | held_out | plateau_p2p_after | 1.063 | 1.295 | relative | PASS |
| square_1730hz_held | held_out | overshoot_after | 0.03634 | 0.1111 | relative | PASS |

## G2_hf_ringing: PASS (worst: square_5000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_5000hz | canonical | plateau_rms_after | 0.4554 | 0.4994 | relative | PASS |
| square_5000hz | canonical | plateau_p2p_after | 1.068 | 1.294 | relative | PASS |
| square_5000hz | canonical | overshoot_after | 0.5345 | 0.5485 | relative | PASS |
| square_4400hz_held | held_out | plateau_rms_after | 0.4607 | 0.5075 | relative | PASS |
| square_4400hz_held | held_out | plateau_p2p_after | 1.063 | 1.295 | relative | PASS |
| square_4400hz_held | held_out | overshoot_after | 0.532 | 0.5485 | relative | PASS |

## G2b_pre_echo: PASS (worst: tone_burst_19000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| impulse | canonical | pre_echo_energy_after | 3.274e-13 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | pre_echo_energy_after | 3.274e-13 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | pre_echo_energy_after | 1.55e-15 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | pre_echo_energy_after | 3.133e-15 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | pre_echo_energy_after | 1.93e-12 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | pre_echo_energy_after | 1.727e-15 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | pre_echo_energy_after | 1.226e-14 | 2.5e-07 | absolute | PASS |

## G3_mirror: PASS (worst: sweep_log_20_20k)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| sweep_log_20_20k | canonical | image_rel_db | -124.6 | -65 | absolute | PASS |
| sweep_log_20_20k | canonical | image_peak_rel_db | -71.85 | -65 | absolute | PASS |
| pink_noise_s1234 | canonical | image_rel_db | -85.84 | -65 | absolute | PASS |
| multitone_60_s20260704 | canonical | image_rel_db | -95.93 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_rel_db | -124.9 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_peak_rel_db | -75.25 | -65 | absolute | PASS |
| pink_noise_s5678_held | held_out | image_rel_db | -89.58 | -65 | absolute | PASS |

## G4_flatness: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | flatness_dip_db | 0.006955 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_boost_db | 1.997e-05 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_hf_dip_db | 0.03847 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_dip_db | 0.002848 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_boost_db | 1.76e-05 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_hf_dip_db | 0.007171 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_dip_db | 0.003759 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_boost_db | 2.292e-05 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_hf_dip_db | 0.03232 | 3 | absolute | PASS |

## G5_gain: PASS (worst: impulse_train_10ms)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | abs_gain_error_db | 0.0124 | 0.5 | absolute | PASS |
| square_100hz | canonical | abs_gain_error_db | 0.01002 | 0.5 | absolute | PASS |
| square_500hz | canonical | abs_gain_error_db | 0.01649 | 0.5 | absolute | PASS |
| square_1000hz | canonical | abs_gain_error_db | 0.04896 | 0.5 | absolute | PASS |
| square_2000hz | canonical | abs_gain_error_db | 0.106 | 0.5 | absolute | PASS |
| square_5000hz | canonical | abs_gain_error_db | 0.3742 | 0.5 | absolute | PASS |
| square_500hz_a005 | canonical | abs_gain_error_db | 0.01649 | 0.5 | absolute | PASS |
| dc_step_up | canonical | abs_gain_error_db | 0.01447 | 0.5 | absolute | PASS |
| dc_step_down | canonical | abs_gain_error_db | 0.01448 | 0.5 | absolute | PASS |
| impulse | canonical | abs_gain_error_db | 0.4844 | 0.5 | absolute | PASS |
| impulse_train_10ms | canonical | abs_gain_error_db | 0.4969 | 0.5 | absolute | PASS |
| tone_burst_1000hz | canonical | abs_gain_error_db | 0.0007953 | 0.5 | absolute | PASS |
| sweep_log_20_20k | canonical | abs_gain_error_db | 0.004068 | 0.5 | absolute | PASS |
| pink_noise_s1234 | canonical | abs_gain_error_db | 0.0003658 | 0.5 | absolute | PASS |
| multitone_60_s20260704 | canonical | abs_gain_error_db | 0.0009217 | 0.5 | absolute | PASS |
| square_73hz_held | held_out | abs_gain_error_db | 0.01174 | 0.5 | absolute | PASS |
| square_331hz_held | held_out | abs_gain_error_db | 0.005608 | 0.5 | absolute | PASS |
| square_1730hz_held | held_out | abs_gain_error_db | 0.1031 | 0.5 | absolute | PASS |
| square_4400hz_held | held_out | abs_gain_error_db | 0.2726 | 0.5 | absolute | PASS |
| tone_burst_3700hz_held | held_out | abs_gain_error_db | 0.0008157 | 0.5 | absolute | PASS |
| sweep_log_30_19k_held | held_out | abs_gain_error_db | 0.003329 | 0.5 | absolute | PASS |
| pink_noise_s5678_held | held_out | abs_gain_error_db | 0.000267 | 0.5 | absolute | PASS |

## G7_no_added_hf: PASS (worst: tone_burst_1000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | added_hf_db | 0.1356 | 3 | relative | PASS |
| square_100hz | canonical | added_hf_db | 0.136 | 3 | relative | PASS |
| square_500hz | canonical | added_hf_db | 0.134 | 3 | relative | PASS |
| square_1000hz | canonical | added_hf_db | 0.1275 | 3 | relative | PASS |
| square_2000hz | canonical | added_hf_db | 0.1939 | 3 | relative | PASS |
| square_5000hz | canonical | added_hf_db | 0.126 | 3 | relative | PASS |
| square_500hz_a005 | canonical | added_hf_db | 0.134 | 3 | relative | PASS |
| dc_step_up | canonical | added_hf_db | 0.1361 | 3 | relative | PASS |
| dc_step_down | canonical | added_hf_db | 0.1361 | 3 | relative | PASS |
| impulse | canonical | added_hf_db | 0.1343 | 3 | relative | PASS |
| impulse_train_10ms | canonical | added_hf_db | 0.1343 | 3 | relative | PASS |
| tone_burst_1000hz | canonical | added_hf_db | 31.31 | 3 | absolute | PASS |
| tone_burst_10000hz | canonical | added_hf_db | -65.28 | 3 | absolute | PASS |
| tone_burst_19000hz | canonical | added_hf_db | -79.52 | 3 | absolute | PASS |
| sweep_log_20_20k | canonical | added_hf_db | -44.18 | 3 | absolute | PASS |
| pink_noise_s1234 | canonical | added_hf_db | -51 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | added_hf_db | -59.94 | 3 | absolute | PASS |
| square_73hz_held | held_out | added_hf_db | 0.1347 | 3 | relative | PASS |
| square_331hz_held | held_out | added_hf_db | 0.1326 | 3 | relative | PASS |
| square_1730hz_held | held_out | added_hf_db | 0.1662 | 3 | relative | PASS |
| square_4400hz_held | held_out | added_hf_db | 0.1236 | 3 | relative | PASS |
| tone_burst_3700hz_held | held_out | added_hf_db | -13.86 | 3 | absolute | PASS |
| tone_burst_14300hz_held | held_out | added_hf_db | -72.45 | 3 | absolute | PASS |
| sweep_log_30_19k_held | held_out | added_hf_db | -42.96 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | added_hf_db | -54.44 | 3 | absolute | PASS |

## G8_lb_preservation: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | lb_phase_error_deg | 0.01962 | 15 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_group_delay_error_samples | 7.861 | 600 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_waveform_error_db | -49.45 | -20 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_phase_error_deg | 0.01851 | 15 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_group_delay_error_samples | 6.501 | 600 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_waveform_error_db | -44.6 | -20 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_phase_error_deg | 0.008647 | 15 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_group_delay_error_samples | 3.475 | 600 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_waveform_error_db | -48.83 | -20 | absolute | PASS |
