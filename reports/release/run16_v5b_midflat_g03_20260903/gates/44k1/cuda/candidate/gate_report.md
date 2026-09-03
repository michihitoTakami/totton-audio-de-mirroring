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
| square_500hz_a005 | canonical | plateau_rms_after | 1.81e-05 | 5e-05 | absolute | PASS |
| square_500hz_a005 | canonical | plateau_p2p_after | 7.605e-05 | 0.0003282 | relative | PASS |
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
| impulse | canonical | pre_echo_energy_after | 4.62e-11 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | pre_echo_energy_after | 4.653e-11 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | pre_echo_energy_after | 2.407e-23 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | pre_echo_energy_after | 8.816e-22 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | pre_echo_energy_after | 4.493e-18 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | pre_echo_energy_after | 8.048e-23 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | pre_echo_energy_after | 1.345e-20 | 2.5e-07 | absolute | PASS |

## G2c_post_echo: PASS (worst: impulse_train_10ms)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| impulse | canonical | post_echo_energy_after | 4.742e-11 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | post_echo_energy_after | 4.742e-11 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | post_echo_energy_after | 6.51e-25 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | post_echo_energy_after | 1.323e-24 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | post_echo_energy_after | 4.025e-20 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | post_echo_energy_after | 1.596e-24 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | post_echo_energy_after | 3.944e-23 | 2.5e-07 | absolute | PASS |

## G3_mirror: PASS (worst: pink_noise_s5678_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| sweep_log_20_20k | canonical | image_rel_db | -133.1 | -65 | absolute | PASS |
| sweep_log_20_20k | canonical | image_peak_rel_db | -118.8 | -65 | absolute | PASS |
| pink_noise_s1234 | canonical | image_rel_db | -115.9 | -65 | absolute | PASS |
| multitone_60_s20260704 | canonical | image_rel_db | -126.3 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_rel_db | -133.1 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_peak_rel_db | -123.3 | -65 | absolute | PASS |
| pink_noise_s5678_held | held_out | image_rel_db | -114 | -65 | absolute | PASS |

## G4_flatness: PASS (worst: pink_noise_s5678_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | flatness_dip_db | 0.0001775 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_boost_db | 1.094e-06 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_hf_dip_db | 0.0002512 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_dip_db | 3.139e-05 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_boost_db | 9.641e-08 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_hf_dip_db | 3.652e-05 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_dip_db | 0.0002083 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_boost_db | 1.351e-06 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_hf_dip_db | 0.0002859 | 3 | absolute | PASS |

## G5_gain: PASS (worst: square_5000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | abs_gain_error_db | 0.01231 | 0.5 | absolute | PASS |
| square_100hz | canonical | abs_gain_error_db | 0.01001 | 0.5 | absolute | PASS |
| square_500hz | canonical | abs_gain_error_db | 0.01649 | 0.5 | absolute | PASS |
| square_1000hz | canonical | abs_gain_error_db | 0.04896 | 0.5 | absolute | PASS |
| square_2000hz | canonical | abs_gain_error_db | 0.106 | 0.5 | absolute | PASS |
| square_5000hz | canonical | abs_gain_error_db | 0.3754 | 0.5 | absolute | PASS |
| square_500hz_a005 | canonical | abs_gain_error_db | 0.01649 | 0.5 | absolute | PASS |
| dc_step_up | canonical | abs_gain_error_db | 0.0004376 | 0.5 | absolute | PASS |
| dc_step_down | canonical | abs_gain_error_db | 0.0004376 | 0.5 | absolute | PASS |
| impulse | canonical | abs_gain_error_db | 0.1497 | 0.5 | absolute | PASS |
| impulse_train_10ms | canonical | abs_gain_error_db | 0.1643 | 0.5 | absolute | PASS |
| tone_burst_1000hz | canonical | abs_gain_error_db | 0.0007896 | 0.5 | absolute | PASS |
| sweep_log_20_20k | canonical | abs_gain_error_db | 0.0003204 | 0.5 | absolute | PASS |
| pink_noise_s1234 | canonical | abs_gain_error_db | 0.0002747 | 0.5 | absolute | PASS |
| multitone_60_s20260704 | canonical | abs_gain_error_db | 0.001002 | 0.5 | absolute | PASS |
| imd_60hz_7000hz | canonical | abs_gain_error_db | 0.001426 | 0.5 | absolute | PASS |
| square_73hz_held | held_out | abs_gain_error_db | 0.01173 | 0.5 | absolute | PASS |
| square_331hz_held | held_out | abs_gain_error_db | 0.005608 | 0.5 | absolute | PASS |
| square_1730hz_held | held_out | abs_gain_error_db | 0.1031 | 0.5 | absolute | PASS |
| square_4400hz_held | held_out | abs_gain_error_db | 0.2732 | 0.5 | absolute | PASS |
| tone_burst_3700hz_held | held_out | abs_gain_error_db | 0.06268 | 0.5 | absolute | PASS |
| sweep_log_30_19k_held | held_out | abs_gain_error_db | 0.0003546 | 0.5 | absolute | PASS |
| pink_noise_s5678_held | held_out | abs_gain_error_db | 0.0001987 | 0.5 | absolute | PASS |
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
| impulse | canonical | added_hf_db | -4.225 | 3 | relative | PASS |
| impulse_train_10ms | canonical | added_hf_db | -4.217 | 3 | relative | PASS |
| tone_burst_1000hz | canonical | added_hf_db | 26.4 | 3 | absolute | PASS |
| tone_burst_10000hz | canonical | added_hf_db | -5.303 | 3 | relative | PASS |
| tone_burst_19000hz | canonical | added_hf_db | -5.745 | 3 | relative | PASS |
| sweep_log_20_20k | canonical | added_hf_db | -52.73 | 3 | absolute | PASS |
| pink_noise_s1234 | canonical | added_hf_db | -81.03 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | added_hf_db | -90.36 | 3 | absolute | PASS |
| imd_60hz_7000hz | canonical | added_hf_db | -40.43 | 3 | absolute | PASS |
| square_73hz_held | held_out | added_hf_db | 0.1347 | 3 | relative | PASS |
| square_331hz_held | held_out | added_hf_db | 0.1326 | 3 | relative | PASS |
| square_1730hz_held | held_out | added_hf_db | 0.1662 | 3 | relative | PASS |
| square_4400hz_held | held_out | added_hf_db | 0.1422 | 3 | relative | PASS |
| tone_burst_3700hz_held | held_out | added_hf_db | 1.273 | 3 | absolute | PASS |
| tone_burst_14300hz_held | held_out | added_hf_db | -5.578 | 3 | relative | PASS |
| sweep_log_30_19k_held | held_out | added_hf_db | -51.2 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | added_hf_db | -78.88 | 3 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | added_hf_db | -35.49 | 3 | absolute | PASS |

## G8_lb_preservation: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | lb_phase_error_deg | 0.01384 | 15 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_group_delay_error_samples | 5.318 | 600 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_waveform_error_db | -49.35 | -20 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_phase_error_deg | 0.01002 | 15 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_group_delay_error_samples | 0.2597 | 600 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_waveform_error_db | -44.53 | -20 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_phase_error_deg | 0.004651 | 15 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_group_delay_error_samples | 1.776 | 600 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_waveform_error_db | -48.75 | -20 | absolute | PASS |

## G9_no_modulation_sidebands: PASS (worst: imd_60hz_7000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| imd_60hz_7000hz | canonical | modulation_sideband_db | -137.5 | -110 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | modulation_sideband_db | -144.1 | -110 | absolute | PASS |
