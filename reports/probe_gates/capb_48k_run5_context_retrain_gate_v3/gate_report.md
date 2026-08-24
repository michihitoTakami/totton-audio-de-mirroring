# Stage 1 probe gate report

- all_passed: **False**
- spec_version: 3
- manifest_hash: dcce6b6fbfef3ee6

## G1_lf_ringing: PASS (worst: square_1730hz_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | plateau_rms_after | 1.503e-05 | 0.0005 | absolute | PASS |
| square_50hz | canonical | plateau_p2p_after | 0.0001187 | 0.00158 | absolute | PASS |
| square_50hz | canonical | overshoot_after | 0.02403 | 0.07179 | relative | PASS |
| square_100hz | canonical | plateau_rms_after | 1.503e-05 | 0.0005 | absolute | PASS |
| square_100hz | canonical | plateau_p2p_after | 0.0001187 | 0.00158 | absolute | PASS |
| square_100hz | canonical | overshoot_after | 0.02403 | 0.07179 | relative | PASS |
| square_500hz | canonical | plateau_rms_after | 1.504e-05 | 0.0005 | absolute | PASS |
| square_500hz | canonical | plateau_p2p_after | 0.0001187 | 0.00158 | absolute | PASS |
| square_500hz | canonical | overshoot_after | 0.02403 | 0.07179 | relative | PASS |
| square_1000hz | canonical | plateau_rms_after | 0.6593 | 0.7237 | relative | PASS |
| square_1000hz | canonical | plateau_p2p_after | 1.05 | 1.174 | relative | PASS |
| square_1000hz | canonical | overshoot_after | 0.02448 | 0.07187 | relative | PASS |
| square_2000hz | canonical | plateau_rms_after | 0.5546 | 0.6204 | relative | PASS |
| square_2000hz | canonical | plateau_p2p_after | 1.05 | 1.247 | relative | PASS |
| square_2000hz | canonical | overshoot_after | 0.1812 | 0.2082 | relative | PASS |
| square_500hz_a005 | canonical | plateau_rms_after | 1.503e-06 | 5e-05 | absolute | PASS |
| square_500hz_a005 | canonical | plateau_p2p_after | 1.187e-05 | 0.000158 | absolute | PASS |
| square_500hz_a005 | canonical | overshoot_after | 0.002403 | 0.01168 | relative | PASS |
| dc_step_up | canonical | plateau_rms_after | 1.503e-05 | 0.0005 | absolute | PASS |
| dc_step_up | canonical | plateau_p2p_after | 0.0001187 | 0.001951 | relative | PASS |
| dc_step_up | canonical | overshoot_after | 0.02403 | 0.07179 | relative | PASS |
| dc_step_down | canonical | plateau_rms_after | 1.503e-05 | 0.0005 | absolute | PASS |
| dc_step_down | canonical | plateau_p2p_after | 0.0001187 | 0.001951 | relative | PASS |
| dc_step_down | canonical | overshoot_after | 0.02403 | 0.07179 | relative | PASS |
| square_73hz_held | held_out | plateau_rms_after | 1.503e-05 | 0.0005 | absolute | PASS |
| square_73hz_held | held_out | plateau_p2p_after | 0.0001187 | 0.00158 | absolute | PASS |
| square_73hz_held | held_out | overshoot_after | 0.02403 | 0.07179 | relative | PASS |
| square_331hz_held | held_out | plateau_rms_after | 1.503e-05 | 0.0005 | absolute | PASS |
| square_331hz_held | held_out | plateau_p2p_after | 0.0001187 | 0.00158 | absolute | PASS |
| square_331hz_held | held_out | overshoot_after | 0.02403 | 0.07179 | relative | PASS |
| square_1730hz_held | held_out | plateau_rms_after | 0.6367 | 0.694 | relative | PASS |
| square_1730hz_held | held_out | plateau_p2p_after | 1.05 | 1.247 | relative | PASS |
| square_1730hz_held | held_out | overshoot_after | 0.02453 | 0.07952 | relative | PASS |

## G2_hf_ringing: PASS (worst: square_4400hz_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_5000hz | canonical | plateau_rms_after | 0.5403 | 0.563 | relative | PASS |
| square_5000hz | canonical | plateau_p2p_after | 1.054 | 1.247 | relative | PASS |
| square_5000hz | canonical | overshoot_after | 0.1833 | 0.283 | relative | PASS |
| square_4400hz_held | held_out | plateau_rms_after | 0.541 | 0.5631 | relative | PASS |
| square_4400hz_held | held_out | plateau_p2p_after | 1.051 | 1.247 | relative | PASS |
| square_4400hz_held | held_out | overshoot_after | 0.1805 | 0.283 | relative | PASS |

## G2b_pre_echo: PASS (worst: impulse)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| impulse | canonical | pre_echo_energy_after | 6.865e-10 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | pre_echo_energy_after | 6.865e-10 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | pre_echo_energy_after | 7.918e-17 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | pre_echo_energy_after | 1.532e-16 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | pre_echo_energy_after | 1.977e-14 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | pre_echo_energy_after | 7.534e-17 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | pre_echo_energy_after | 5.463e-16 | 2.5e-07 | absolute | PASS |

## G3_mirror: FAIL (worst: sweep_log_20_20k)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| sweep_log_20_20k | canonical | image_peak_rel_db | -41.54 | -65 | absolute | FAIL |
| sweep_log_30_19k_held | held_out | image_peak_rel_db | -46.46 | -65 | absolute | FAIL |
| sweep_log_20_20k | canonical | image_rel_db | -110.5 | -65 | absolute | PASS |
| pink_noise_s1234 | canonical | image_rel_db | -96.14 | -65 | absolute | PASS |
| multitone_60_s20260704 | canonical | image_rel_db | -107 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_rel_db | -115.7 | -65 | absolute | PASS |
| pink_noise_s5678_held | held_out | image_rel_db | -98.49 | -65 | absolute | PASS |

## G4_flatness: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | flatness_dip_db | 0.00537 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_boost_db | 1.055e-05 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_hf_dip_db | 0.01488 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_dip_db | 0.00364 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_boost_db | 4.319e-06 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_hf_dip_db | 0.004372 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_dip_db | 0.004165 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_boost_db | 6.272e-06 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_hf_dip_db | 0.01159 | 3 | absolute | PASS |

## G5_gain: PASS (worst: square_5000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | abs_gain_error_db | 0.01114 | 0.5 | absolute | PASS |
| square_100hz | canonical | abs_gain_error_db | 0.007267 | 0.5 | absolute | PASS |
| square_500hz | canonical | abs_gain_error_db | 0.01882 | 0.5 | absolute | PASS |
| square_1000hz | canonical | abs_gain_error_db | 0.05332 | 0.5 | absolute | PASS |
| square_2000hz | canonical | abs_gain_error_db | 0.1163 | 0.5 | absolute | PASS |
| square_5000hz | canonical | abs_gain_error_db | 0.3877 | 0.5 | absolute | PASS |
| square_500hz_a005 | canonical | abs_gain_error_db | 0.01882 | 0.5 | absolute | PASS |
| dc_step_up | canonical | abs_gain_error_db | 0.01545 | 0.5 | absolute | PASS |
| dc_step_down | canonical | abs_gain_error_db | 0.01545 | 0.5 | absolute | PASS |
| impulse | canonical | abs_gain_error_db | 0.009124 | 0.5 | absolute | PASS |
| impulse_train_10ms | canonical | abs_gain_error_db | 0.02193 | 0.5 | absolute | PASS |
| tone_burst_1000hz | canonical | abs_gain_error_db | 0.001176 | 0.5 | absolute | PASS |
| sweep_log_20_20k | canonical | abs_gain_error_db | 0.002151 | 0.5 | absolute | PASS |
| pink_noise_s1234 | canonical | abs_gain_error_db | 0.001507 | 0.5 | absolute | PASS |
| multitone_60_s20260704 | canonical | abs_gain_error_db | 0.001545 | 0.5 | absolute | PASS |
| square_73hz_held | held_out | abs_gain_error_db | 0.009318 | 0.5 | absolute | PASS |
| square_331hz_held | held_out | abs_gain_error_db | 0.007093 | 0.5 | absolute | PASS |
| square_1730hz_held | held_out | abs_gain_error_db | 0.1103 | 0.5 | absolute | PASS |
| square_4400hz_held | held_out | abs_gain_error_db | 0.3088 | 0.5 | absolute | PASS |
| tone_burst_3700hz_held | held_out | abs_gain_error_db | 0.001212 | 0.5 | absolute | PASS |
| sweep_log_30_19k_held | held_out | abs_gain_error_db | 0.001263 | 0.5 | absolute | PASS |
| pink_noise_s5678_held | held_out | abs_gain_error_db | 0.001904 | 0.5 | absolute | PASS |

## G7_no_added_hf: PASS (worst: tone_burst_1000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | added_hf_db | 0.1827 | 3 | relative | PASS |
| square_100hz | canonical | added_hf_db | 0.1821 | 3 | relative | PASS |
| square_500hz | canonical | added_hf_db | 0.179 | 3 | relative | PASS |
| square_1000hz | canonical | added_hf_db | 0.1805 | 3 | relative | PASS |
| square_2000hz | canonical | added_hf_db | 0.1843 | 3 | relative | PASS |
| square_5000hz | canonical | added_hf_db | 0.133 | 3 | relative | PASS |
| square_500hz_a005 | canonical | added_hf_db | 0.179 | 3 | relative | PASS |
| dc_step_up | canonical | added_hf_db | 0.1827 | 3 | relative | PASS |
| dc_step_down | canonical | added_hf_db | 0.1827 | 3 | relative | PASS |
| impulse | canonical | added_hf_db | -63.14 | 3 | absolute | PASS |
| impulse_train_10ms | canonical | added_hf_db | -63.11 | 3 | absolute | PASS |
| tone_burst_1000hz | canonical | added_hf_db | 54.21 | 3 | absolute | PASS |
| tone_burst_10000hz | canonical | added_hf_db | -44.83 | 3 | absolute | PASS |
| tone_burst_19000hz | canonical | added_hf_db | -3.499 | 3 | relative | PASS |
| sweep_log_20_20k | canonical | added_hf_db | -18.94 | 3 | absolute | PASS |
| pink_noise_s1234 | canonical | added_hf_db | -56.16 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | added_hf_db | -60.26 | 3 | absolute | PASS |
| square_73hz_held | held_out | added_hf_db | 0.183 | 3 | relative | PASS |
| square_331hz_held | held_out | added_hf_db | 0.185 | 3 | relative | PASS |
| square_1730hz_held | held_out | added_hf_db | 0.1832 | 3 | relative | PASS |
| square_4400hz_held | held_out | added_hf_db | 0.1662 | 3 | relative | PASS |
| tone_burst_3700hz_held | held_out | added_hf_db | 5.179 | 3 | absolute | PASS |
| tone_burst_14300hz_held | held_out | added_hf_db | -63.82 | 3 | absolute | PASS |
| sweep_log_30_19k_held | held_out | added_hf_db | -22.56 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | added_hf_db | -58.11 | 3 | absolute | PASS |

## G8_lb_preservation: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | lb_phase_error_deg | 0.01547 | 15 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_group_delay_error_samples | 6.543 | 600 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_waveform_error_db | -65.22 | -20 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_phase_error_deg | 0.01397 | 15 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_group_delay_error_samples | 3.916 | 600 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_waveform_error_db | -64.53 | -20 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_phase_error_deg | 0.004536 | 15 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_group_delay_error_samples | 2.01 | 600 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_waveform_error_db | -65.96 | -20 | absolute | PASS |
