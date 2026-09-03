# Stage 1 probe gate report

- all_passed: **True**
- spec_version: 6
- manifest_hash: 6529deadaab177fe

## G1_lf_ringing: PASS (worst: square_2000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | plateau_rms_after | 0.0001742 | 0.0005 | absolute | PASS |
| square_50hz | canonical | plateau_p2p_after | 0.0006528 | 0.00187 | relative | PASS |
| square_50hz | canonical | overshoot_after | 0.03001 | 0.07761 | relative | PASS |
| square_100hz | canonical | plateau_rms_after | 0.0001742 | 0.0005 | absolute | PASS |
| square_100hz | canonical | plateau_p2p_after | 0.0006528 | 0.001952 | relative | PASS |
| square_100hz | canonical | overshoot_after | 0.03001 | 0.07761 | relative | PASS |
| square_500hz | canonical | plateau_rms_after | 0.000174 | 0.0005 | absolute | PASS |
| square_500hz | canonical | plateau_p2p_after | 0.0006528 | 0.001952 | relative | PASS |
| square_500hz | canonical | overshoot_after | 0.03001 | 0.07761 | relative | PASS |
| square_1000hz | canonical | plateau_rms_after | 0.6511 | 0.7175 | relative | PASS |
| square_1000hz | canonical | plateau_p2p_after | 1.062 | 1.181 | relative | PASS |
| square_1000hz | canonical | overshoot_after | 0.03015 | 0.07776 | relative | PASS |
| square_2000hz | canonical | plateau_rms_after | 0.6225 | 0.6645 | relative | PASS |
| square_2000hz | canonical | plateau_p2p_after | 1.062 | 1.26 | relative | PASS |
| square_2000hz | canonical | overshoot_after | 0.05796 | 0.1367 | relative | PASS |
| square_500hz_a005 | canonical | plateau_rms_after | 1.74e-05 | 5e-05 | absolute | PASS |
| square_500hz_a005 | canonical | plateau_p2p_after | 6.528e-05 | 0.0001952 | relative | PASS |
| square_500hz_a005 | canonical | overshoot_after | 0.003001 | 0.01226 | relative | PASS |
| dc_step_up | canonical | plateau_rms_after | 0.0001749 | 0.0005 | absolute | PASS |
| dc_step_up | canonical | plateau_p2p_after | 0.0006528 | 0.002288 | relative | PASS |
| dc_step_up | canonical | overshoot_after | 0.03001 | 0.07761 | relative | PASS |
| dc_step_down | canonical | plateau_rms_after | 0.0001749 | 0.0005 | absolute | PASS |
| dc_step_down | canonical | plateau_p2p_after | 0.0006528 | 0.002288 | relative | PASS |
| dc_step_down | canonical | overshoot_after | 0.03001 | 0.07761 | relative | PASS |
| square_73hz_held | held_out | plateau_rms_after | 0.000174 | 0.0005 | absolute | PASS |
| square_73hz_held | held_out | plateau_p2p_after | 0.0006528 | 0.002184 | relative | PASS |
| square_73hz_held | held_out | overshoot_after | 0.03001 | 0.07761 | relative | PASS |
| square_331hz_held | held_out | plateau_rms_after | 0.000174 | 0.0005 | absolute | PASS |
| square_331hz_held | held_out | plateau_p2p_after | 0.0006528 | 0.002184 | relative | PASS |
| square_331hz_held | held_out | overshoot_after | 0.03001 | 0.07761 | relative | PASS |
| square_1730hz_held | held_out | plateau_rms_after | 0.6339 | 0.6928 | relative | PASS |
| square_1730hz_held | held_out | plateau_p2p_after | 1.062 | 1.26 | relative | PASS |
| square_1730hz_held | held_out | overshoot_after | 0.0341 | 0.08627 | relative | PASS |

## G2_hf_ringing: PASS (worst: square_4400hz_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_5000hz | canonical | plateau_rms_after | 0.4734 | 0.513 | relative | PASS |
| square_5000hz | canonical | plateau_p2p_after | 1.065 | 1.261 | relative | PASS |
| square_5000hz | canonical | overshoot_after | 0.3464 | 0.415 | relative | PASS |
| square_4400hz_held | held_out | plateau_rms_after | 0.4889 | 0.5272 | relative | PASS |
| square_4400hz_held | held_out | plateau_p2p_after | 1.064 | 1.261 | relative | PASS |
| square_4400hz_held | held_out | overshoot_after | 0.3015 | 0.3732 | relative | PASS |

## G2b_pre_echo: PASS (worst: impulse_train_10ms)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| impulse | canonical | pre_echo_energy_after | 1.33e-11 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | pre_echo_energy_after | 1.339e-11 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | pre_echo_energy_after | 1.902e-25 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | pre_echo_energy_after | 3.309e-22 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | pre_echo_energy_after | 3.355e-19 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | pre_echo_energy_after | 8.776e-24 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | pre_echo_energy_after | 3.965e-21 | 2.5e-07 | absolute | PASS |

## G2c_post_echo: PASS (worst: impulse)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| impulse | canonical | post_echo_energy_after | 1.259e-11 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | post_echo_energy_after | 1.259e-11 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | post_echo_energy_after | 3.872e-27 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | post_echo_energy_after | 8.148e-23 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | post_echo_energy_after | 3.641e-17 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | post_echo_energy_after | 9.122e-25 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | post_echo_energy_after | 1.335e-20 | 2.5e-07 | absolute | PASS |

## G3_mirror: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| sweep_log_20_20k | canonical | image_rel_db | -133.6 | -65 | absolute | PASS |
| sweep_log_20_20k | canonical | image_peak_rel_db | -128.3 | -65 | absolute | PASS |
| pink_noise_s1234 | canonical | image_rel_db | -128.1 | -65 | absolute | PASS |
| multitone_60_s20260704 | canonical | image_rel_db | -131.8 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_rel_db | -133.5 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_peak_rel_db | -133.3 | -65 | absolute | PASS |
| pink_noise_s5678_held | held_out | image_rel_db | -129.5 | -65 | absolute | PASS |

## G4_flatness: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | flatness_dip_db | 5.09e-05 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_boost_db | 1.178e-07 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_hf_dip_db | 6.339e-05 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_dip_db | 8.567e-06 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_boost_db | 0 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_hf_dip_db | 9.886e-06 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_dip_db | 4.057e-05 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_boost_db | 8.11e-09 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_hf_dip_db | 5.211e-05 | 3 | absolute | PASS |

## G5_gain: PASS (worst: impulse_train_10ms)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | abs_gain_error_db | 0.0111 | 0.5 | absolute | PASS |
| square_100hz | canonical | abs_gain_error_db | 0.007267 | 0.5 | absolute | PASS |
| square_500hz | canonical | abs_gain_error_db | 0.01882 | 0.5 | absolute | PASS |
| square_1000hz | canonical | abs_gain_error_db | 0.05332 | 0.5 | absolute | PASS |
| square_2000hz | canonical | abs_gain_error_db | 0.1163 | 0.5 | absolute | PASS |
| square_5000hz | canonical | abs_gain_error_db | 0.389 | 0.5 | absolute | PASS |
| square_500hz_a005 | canonical | abs_gain_error_db | 0.01882 | 0.5 | absolute | PASS |
| dc_step_up | canonical | abs_gain_error_db | 0.0007058 | 0.5 | absolute | PASS |
| dc_step_down | canonical | abs_gain_error_db | 0.0007058 | 0.5 | absolute | PASS |
| impulse | canonical | abs_gain_error_db | 0.445 | 0.5 | absolute | PASS |
| impulse_train_10ms | canonical | abs_gain_error_db | 0.4568 | 0.5 | absolute | PASS |
| tone_burst_1000hz | canonical | abs_gain_error_db | 0.001182 | 0.5 | absolute | PASS |
| sweep_log_20_20k | canonical | abs_gain_error_db | 0.002363 | 0.5 | absolute | PASS |
| pink_noise_s1234 | canonical | abs_gain_error_db | 0.001407 | 0.5 | absolute | PASS |
| multitone_60_s20260704 | canonical | abs_gain_error_db | 0.001417 | 0.5 | absolute | PASS |
| imd_60hz_7000hz | canonical | abs_gain_error_db | 0.001638 | 0.5 | absolute | PASS |
| square_73hz_held | held_out | abs_gain_error_db | 0.009322 | 0.5 | absolute | PASS |
| square_331hz_held | held_out | abs_gain_error_db | 0.007093 | 0.5 | absolute | PASS |
| square_1730hz_held | held_out | abs_gain_error_db | 0.1103 | 0.5 | absolute | PASS |
| square_4400hz_held | held_out | abs_gain_error_db | 0.3095 | 0.5 | absolute | PASS |
| tone_burst_3700hz_held | held_out | abs_gain_error_db | 0.08028 | 0.5 | absolute | PASS |
| sweep_log_30_19k_held | held_out | abs_gain_error_db | 0.002493 | 0.5 | absolute | PASS |
| pink_noise_s5678_held | held_out | abs_gain_error_db | 0.001814 | 0.5 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | abs_gain_error_db | 0.00193 | 0.5 | absolute | PASS |

## G7_no_added_hf: PASS (worst: tone_burst_1000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | added_hf_db | 0.1827 | 3 | relative | PASS |
| square_100hz | canonical | added_hf_db | 0.1821 | 3 | relative | PASS |
| square_500hz | canonical | added_hf_db | 0.179 | 3 | relative | PASS |
| square_1000hz | canonical | added_hf_db | 0.1805 | 3 | relative | PASS |
| square_2000hz | canonical | added_hf_db | 0.1843 | 3 | relative | PASS |
| square_5000hz | canonical | added_hf_db | 0.1624 | 3 | relative | PASS |
| square_500hz_a005 | canonical | added_hf_db | 0.179 | 3 | relative | PASS |
| dc_step_up | canonical | added_hf_db | 0.1827 | 3 | relative | PASS |
| dc_step_down | canonical | added_hf_db | 0.1827 | 3 | relative | PASS |
| impulse | canonical | added_hf_db | -1.101 | 3 | relative | PASS |
| impulse_train_10ms | canonical | added_hf_db | -1.126 | 3 | relative | PASS |
| tone_burst_1000hz | canonical | added_hf_db | 33.51 | 3 | absolute | PASS |
| tone_burst_10000hz | canonical | added_hf_db | -6.896 | 3 | absolute | PASS |
| tone_burst_19000hz | canonical | added_hf_db | -9.24 | 3 | relative | PASS |
| sweep_log_20_20k | canonical | added_hf_db | -41.95 | 3 | absolute | PASS |
| pink_noise_s1234 | canonical | added_hf_db | -88.11 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | added_hf_db | -85.08 | 3 | absolute | PASS |
| imd_60hz_7000hz | canonical | added_hf_db | -29.84 | 3 | absolute | PASS |
| square_73hz_held | held_out | added_hf_db | 0.183 | 3 | relative | PASS |
| square_331hz_held | held_out | added_hf_db | 0.185 | 3 | relative | PASS |
| square_1730hz_held | held_out | added_hf_db | 0.1832 | 3 | relative | PASS |
| square_4400hz_held | held_out | added_hf_db | 0.1867 | 3 | relative | PASS |
| tone_burst_3700hz_held | held_out | added_hf_db | 5.169 | 3 | absolute | PASS |
| tone_burst_14300hz_held | held_out | added_hf_db | -7.715 | 3 | relative | PASS |
| sweep_log_30_19k_held | held_out | added_hf_db | -40.41 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | added_hf_db | -89.15 | 3 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | added_hf_db | -24.9 | 3 | absolute | PASS |

## G8_lb_preservation: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | lb_phase_error_deg | 0.01777 | 15 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_group_delay_error_samples | 7.37 | 600 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_waveform_error_db | -65.93 | -20 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_phase_error_deg | 0.009651 | 15 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_group_delay_error_samples | 0.1904 | 600 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_waveform_error_db | -65.23 | -20 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_phase_error_deg | 0.002958 | 15 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_group_delay_error_samples | 1.241 | 600 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_waveform_error_db | -66.98 | -20 | absolute | PASS |

## G9_no_modulation_sidebands: PASS (worst: imd_60hz_7000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| imd_60hz_7000hz | canonical | modulation_sideband_db | -137.9 | -110 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | modulation_sideband_db | -145.1 | -110 | absolute | PASS |
