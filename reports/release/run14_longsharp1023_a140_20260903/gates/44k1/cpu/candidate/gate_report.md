# Stage 1 probe gate report

- all_passed: **True**
- spec_version: 6
- manifest_hash: 085959477798407d

## G1_lf_ringing: PASS (worst: square_2000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | plateau_rms_after | 0.0001809 | 0.0005 | absolute | PASS |
| square_50hz | canonical | plateau_p2p_after | 0.0007605 | 0.002909 | relative | PASS |
| square_50hz | canonical | overshoot_after | 0.03753 | 0.0962 | relative | PASS |
| square_100hz | canonical | plateau_rms_after | 0.0001807 | 0.0005 | absolute | PASS |
| square_100hz | canonical | plateau_p2p_after | 0.0007605 | 0.002251 | relative | PASS |
| square_100hz | canonical | overshoot_after | 0.03753 | 0.0962 | relative | PASS |
| square_500hz | canonical | plateau_rms_after | 0.000181 | 0.0005 | absolute | PASS |
| square_500hz | canonical | plateau_p2p_after | 0.0007605 | 0.003282 | relative | PASS |
| square_500hz | canonical | overshoot_after | 0.03753 | 0.0962 | relative | PASS |
| square_1000hz | canonical | plateau_rms_after | 0.6512 | 0.7174 | relative | PASS |
| square_1000hz | canonical | plateau_p2p_after | 1.077 | 1.201 | relative | PASS |
| square_1000hz | canonical | overshoot_after | 0.03771 | 0.09648 | relative | PASS |
| square_2000hz | canonical | plateau_rms_after | 0.6305 | 0.6615 | relative | PASS |
| square_2000hz | canonical | plateau_p2p_after | 1.077 | 1.301 | relative | PASS |
| square_2000hz | canonical | overshoot_after | 0.05471 | 0.1636 | relative | PASS |
| square_500hz_a005 | canonical | plateau_rms_after | 1.809e-05 | 5e-05 | absolute | PASS |
| square_500hz_a005 | canonical | plateau_p2p_after | 7.604e-05 | 0.0003282 | relative | PASS |
| square_500hz_a005 | canonical | overshoot_after | 0.003753 | 0.01412 | relative | PASS |
| dc_step_up | canonical | plateau_rms_after | 0.0001809 | 0.0005 | absolute | PASS |
| dc_step_up | canonical | plateau_p2p_after | 0.0007605 | 0.003546 | relative | PASS |
| dc_step_up | canonical | overshoot_after | 0.03753 | 0.0962 | relative | PASS |
| dc_step_down | canonical | plateau_rms_after | 0.0001809 | 0.0005 | absolute | PASS |
| dc_step_down | canonical | plateau_p2p_after | 0.0007605 | 0.003546 | relative | PASS |
| dc_step_down | canonical | overshoot_after | 0.03753 | 0.0962 | relative | PASS |
| square_73hz_held | held_out | plateau_rms_after | 0.000181 | 0.0005 | absolute | PASS |
| square_73hz_held | held_out | plateau_p2p_after | 0.0007605 | 0.003495 | relative | PASS |
| square_73hz_held | held_out | overshoot_after | 0.03753 | 0.0962 | relative | PASS |
| square_331hz_held | held_out | plateau_rms_after | 0.000181 | 0.0005 | absolute | PASS |
| square_331hz_held | held_out | plateau_p2p_after | 0.0007605 | 0.003282 | relative | PASS |
| square_331hz_held | held_out | overshoot_after | 0.03753 | 0.0962 | relative | PASS |
| square_1730hz_held | held_out | plateau_rms_after | 0.637 | 0.6953 | relative | PASS |
| square_1730hz_held | held_out | plateau_p2p_after | 1.077 | 1.301 | relative | PASS |
| square_1730hz_held | held_out | overshoot_after | 0.04372 | 0.1083 | relative | PASS |

## G2_hf_ringing: PASS (worst: square_5000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_5000hz | canonical | plateau_rms_after | 0.4616 | 0.5057 | relative | PASS |
| square_5000hz | canonical | plateau_p2p_after | 1.08 | 1.3 | relative | PASS |
| square_5000hz | canonical | overshoot_after | 0.4345 | 0.4973 | relative | PASS |
| square_4400hz_held | held_out | plateau_rms_after | 0.4645 | 0.5093 | relative | PASS |
| square_4400hz_held | held_out | plateau_p2p_after | 1.077 | 1.301 | relative | PASS |
| square_4400hz_held | held_out | overshoot_after | 0.4856 | 0.54 | relative | PASS |

## G2b_pre_echo: PASS (worst: impulse_train_10ms)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| impulse | canonical | pre_echo_energy_after | 5.598e-11 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | pre_echo_energy_after | 5.598e-11 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | pre_echo_energy_after | 2.877e-24 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | pre_echo_energy_after | 1.987e-22 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | pre_echo_energy_after | 1.09e-18 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | pre_echo_energy_after | 1.284e-23 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | pre_echo_energy_after | 3.258e-21 | 2.5e-07 | absolute | PASS |

## G2c_post_echo: PASS (worst: impulse_train_10ms)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| impulse | canonical | post_echo_energy_after | 5.492e-11 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | post_echo_energy_after | 5.492e-11 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | post_echo_energy_after | 3.233e-25 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | post_echo_energy_after | 5.902e-27 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | post_echo_energy_after | 9.806e-22 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | post_echo_energy_after | 1.023e-24 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | post_echo_energy_after | 1.32e-24 | 2.5e-07 | absolute | PASS |

## G3_mirror: PASS (worst: pink_noise_s5678_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| sweep_log_20_20k | canonical | image_rel_db | -133.1 | -65 | absolute | PASS |
| sweep_log_20_20k | canonical | image_peak_rel_db | -118.7 | -65 | absolute | PASS |
| pink_noise_s1234 | canonical | image_rel_db | -111.3 | -65 | absolute | PASS |
| multitone_60_s20260704 | canonical | image_rel_db | -119.5 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_rel_db | -133.1 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_peak_rel_db | -123.3 | -65 | absolute | PASS |
| pink_noise_s5678_held | held_out | image_rel_db | -108.8 | -65 | absolute | PASS |

## G4_flatness: PASS (worst: pink_noise_s5678_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | flatness_dip_db | 0.0003004 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_boost_db | 1.627e-06 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_hf_dip_db | 0.0004158 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_dip_db | 6.74e-05 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_boost_db | 3.96e-07 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_hf_dip_db | 7.91e-05 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_dip_db | 0.0003855 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_boost_db | 2.346e-06 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_hf_dip_db | 0.0005222 | 3 | absolute | PASS |

## G5_gain: PASS (worst: impulse_train_10ms)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | abs_gain_error_db | 0.01211 | 0.5 | absolute | PASS |
| square_100hz | canonical | abs_gain_error_db | 0.01001 | 0.5 | absolute | PASS |
| square_500hz | canonical | abs_gain_error_db | 0.01649 | 0.5 | absolute | PASS |
| square_1000hz | canonical | abs_gain_error_db | 0.04896 | 0.5 | absolute | PASS |
| square_2000hz | canonical | abs_gain_error_db | 0.106 | 0.5 | absolute | PASS |
| square_5000hz | canonical | abs_gain_error_db | 0.3754 | 0.5 | absolute | PASS |
| square_500hz_a005 | canonical | abs_gain_error_db | 0.01649 | 0.5 | absolute | PASS |
| dc_step_up | canonical | abs_gain_error_db | 0.0004284 | 0.5 | absolute | PASS |
| dc_step_down | canonical | abs_gain_error_db | 0.0004284 | 0.5 | absolute | PASS |
| impulse | canonical | abs_gain_error_db | 0.4343 | 0.5 | absolute | PASS |
| impulse_train_10ms | canonical | abs_gain_error_db | 0.4465 | 0.5 | absolute | PASS |
| tone_burst_1000hz | canonical | abs_gain_error_db | 0.000789 | 0.5 | absolute | PASS |
| sweep_log_20_20k | canonical | abs_gain_error_db | 0.0003682 | 0.5 | absolute | PASS |
| pink_noise_s1234 | canonical | abs_gain_error_db | 0.0002761 | 0.5 | absolute | PASS |
| multitone_60_s20260704 | canonical | abs_gain_error_db | 0.001001 | 0.5 | absolute | PASS |
| imd_60hz_7000hz | canonical | abs_gain_error_db | 0.001426 | 0.5 | absolute | PASS |
| square_73hz_held | held_out | abs_gain_error_db | 0.01173 | 0.5 | absolute | PASS |
| square_331hz_held | held_out | abs_gain_error_db | 0.005608 | 0.5 | absolute | PASS |
| square_1730hz_held | held_out | abs_gain_error_db | 0.1031 | 0.5 | absolute | PASS |
| square_4400hz_held | held_out | abs_gain_error_db | 0.2732 | 0.5 | absolute | PASS |
| tone_burst_3700hz_held | held_out | abs_gain_error_db | 0.06288 | 0.5 | absolute | PASS |
| sweep_log_30_19k_held | held_out | abs_gain_error_db | 0.0004059 | 0.5 | absolute | PASS |
| pink_noise_s5678_held | held_out | abs_gain_error_db | 0.0002012 | 0.5 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | abs_gain_error_db | 0.001975 | 0.5 | absolute | PASS |

## G7_no_added_hf: PASS (worst: tone_burst_1000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | added_hf_db | 0.1356 | 3 | relative | PASS |
| square_100hz | canonical | added_hf_db | 0.136 | 3 | relative | PASS |
| square_500hz | canonical | added_hf_db | 0.134 | 3 | relative | PASS |
| square_1000hz | canonical | added_hf_db | 0.1275 | 3 | relative | PASS |
| square_2000hz | canonical | added_hf_db | 0.1939 | 3 | relative | PASS |
| square_5000hz | canonical | added_hf_db | 0.1551 | 3 | relative | PASS |
| square_500hz_a005 | canonical | added_hf_db | 0.134 | 3 | relative | PASS |
| dc_step_up | canonical | added_hf_db | 0.1361 | 3 | relative | PASS |
| dc_step_down | canonical | added_hf_db | 0.1361 | 3 | relative | PASS |
| impulse | canonical | added_hf_db | -0.8206 | 3 | relative | PASS |
| impulse_train_10ms | canonical | added_hf_db | -0.8281 | 3 | relative | PASS |
| tone_burst_1000hz | canonical | added_hf_db | 26.65 | 3 | absolute | PASS |
| tone_burst_10000hz | canonical | added_hf_db | -5.262 | 3 | relative | PASS |
| tone_burst_19000hz | canonical | added_hf_db | -5.693 | 3 | relative | PASS |
| sweep_log_20_20k | canonical | added_hf_db | -52.69 | 3 | absolute | PASS |
| pink_noise_s1234 | canonical | added_hf_db | -76.47 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | added_hf_db | -83.5 | 3 | absolute | PASS |
| imd_60hz_7000hz | canonical | added_hf_db | -40.39 | 3 | absolute | PASS |
| square_73hz_held | held_out | added_hf_db | 0.1347 | 3 | relative | PASS |
| square_331hz_held | held_out | added_hf_db | 0.1326 | 3 | relative | PASS |
| square_1730hz_held | held_out | added_hf_db | 0.1662 | 3 | relative | PASS |
| square_4400hz_held | held_out | added_hf_db | 0.1422 | 3 | relative | PASS |
| tone_burst_3700hz_held | held_out | added_hf_db | 1.242 | 3 | absolute | PASS |
| tone_burst_14300hz_held | held_out | added_hf_db | -5.527 | 3 | relative | PASS |
| sweep_log_30_19k_held | held_out | added_hf_db | -51.15 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | added_hf_db | -73.67 | 3 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | added_hf_db | -35.47 | 3 | absolute | PASS |

## G8_lb_preservation: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | lb_phase_error_deg | 0.01385 | 15 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_group_delay_error_samples | 5.32 | 600 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_waveform_error_db | -49.35 | -20 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_phase_error_deg | 0.01015 | 15 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_group_delay_error_samples | 0.5498 | 600 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_waveform_error_db | -44.53 | -20 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_phase_error_deg | 0.00477 | 15 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_group_delay_error_samples | 1.836 | 600 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_waveform_error_db | -48.75 | -20 | absolute | PASS |

## G9_no_modulation_sidebands: PASS (worst: imd_60hz_7000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| imd_60hz_7000hz | canonical | modulation_sideband_db | -138.1 | -110 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | modulation_sideband_db | -144.1 | -110 | absolute | PASS |
