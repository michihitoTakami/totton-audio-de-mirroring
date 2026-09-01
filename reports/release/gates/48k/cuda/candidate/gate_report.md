# Stage 1 probe gate report

- all_passed: **True**
- spec_version: 4
- manifest_hash: 6529deadaab177fe

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
| square_500hz_a005 | canonical | plateau_rms_after | 2.702e-06 | 5e-05 | absolute | PASS |
| square_500hz_a005 | canonical | plateau_p2p_after | 2.767e-05 | 0.000158 | absolute | PASS |
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
| square_5000hz | canonical | plateau_rms_after | 0.5401 | 0.563 | relative | PASS |
| square_5000hz | canonical | plateau_p2p_after | 1.053 | 1.247 | relative | PASS |
| square_5000hz | canonical | overshoot_after | 0.1833 | 0.283 | relative | PASS |
| square_4400hz_held | held_out | plateau_rms_after | 0.5408 | 0.5631 | relative | PASS |
| square_4400hz_held | held_out | plateau_p2p_after | 1.051 | 1.247 | relative | PASS |
| square_4400hz_held | held_out | overshoot_after | 0.1808 | 0.283 | relative | PASS |

## G2b_pre_echo: PASS (worst: impulse)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| impulse | canonical | pre_echo_energy_after | 1.326e-08 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | pre_echo_energy_after | 1.326e-08 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | pre_echo_energy_after | 3.786e-18 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | pre_echo_energy_after | 1.856e-17 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | pre_echo_energy_after | 2.458e-15 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | pre_echo_energy_after | 8.81e-18 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | pre_echo_energy_after | 8e-18 | 2.5e-07 | absolute | PASS |

## G3_mirror: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| sweep_log_20_20k | canonical | image_rel_db | -116.1 | -65 | absolute | PASS |
| sweep_log_20_20k | canonical | image_peak_rel_db | -112.4 | -65 | absolute | PASS |
| pink_noise_s1234 | canonical | image_rel_db | -111.4 | -65 | absolute | PASS |
| multitone_60_s20260704 | canonical | image_rel_db | -114.7 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_rel_db | -116 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_peak_rel_db | -112.4 | -65 | absolute | PASS |
| pink_noise_s5678_held | held_out | image_rel_db | -112.2 | -65 | absolute | PASS |

## G4_flatness: PASS (worst: multitone_60_s20260704)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | flatness_dip_db | 2.52e-05 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_boost_db | 1.122e-06 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_hf_dip_db | 1.614e-05 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_dip_db | 5.064e-05 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_boost_db | 2.732e-05 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_hf_dip_db | 0.0001041 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_dip_db | 2.518e-05 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_boost_db | 1.054e-06 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_hf_dip_db | 1.738e-05 | 3 | absolute | PASS |

## G5_gain: PASS (worst: impulse_train_10ms)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | abs_gain_error_db | 0.0112 | 0.5 | absolute | PASS |
| square_100hz | canonical | abs_gain_error_db | 0.007267 | 0.5 | absolute | PASS |
| square_500hz | canonical | abs_gain_error_db | 0.01882 | 0.5 | absolute | PASS |
| square_1000hz | canonical | abs_gain_error_db | 0.05332 | 0.5 | absolute | PASS |
| square_2000hz | canonical | abs_gain_error_db | 0.1163 | 0.5 | absolute | PASS |
| square_5000hz | canonical | abs_gain_error_db | 0.389 | 0.5 | absolute | PASS |
| square_500hz_a005 | canonical | abs_gain_error_db | 0.01882 | 0.5 | absolute | PASS |
| dc_step_up | canonical | abs_gain_error_db | 0.01561 | 0.5 | absolute | PASS |
| dc_step_down | canonical | abs_gain_error_db | 0.01561 | 0.5 | absolute | PASS |
| impulse | canonical | abs_gain_error_db | 0.3958 | 0.5 | absolute | PASS |
| impulse_train_10ms | canonical | abs_gain_error_db | 0.3998 | 0.5 | absolute | PASS |
| tone_burst_1000hz | canonical | abs_gain_error_db | 0.001175 | 0.5 | absolute | PASS |
| sweep_log_20_20k | canonical | abs_gain_error_db | 7.34e-05 | 0.5 | absolute | PASS |
| pink_noise_s1234 | canonical | abs_gain_error_db | 0.001425 | 0.5 | absolute | PASS |
| multitone_60_s20260704 | canonical | abs_gain_error_db | 0.001428 | 0.5 | absolute | PASS |
| imd_60hz_7000hz | canonical | abs_gain_error_db | 0.001661 | 0.5 | absolute | PASS |
| square_73hz_held | held_out | abs_gain_error_db | 0.009323 | 0.5 | absolute | PASS |
| square_331hz_held | held_out | abs_gain_error_db | 0.007093 | 0.5 | absolute | PASS |
| square_1730hz_held | held_out | abs_gain_error_db | 0.1103 | 0.5 | absolute | PASS |
| square_4400hz_held | held_out | abs_gain_error_db | 0.3095 | 0.5 | absolute | PASS |
| tone_burst_3700hz_held | held_out | abs_gain_error_db | 0.001067 | 0.5 | absolute | PASS |
| sweep_log_30_19k_held | held_out | abs_gain_error_db | 0.0009011 | 0.5 | absolute | PASS |
| pink_noise_s5678_held | held_out | abs_gain_error_db | 0.001831 | 0.5 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | abs_gain_error_db | 0.001952 | 0.5 | absolute | PASS |

## G7_no_added_hf: PASS (worst: tone_burst_1000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | added_hf_db | 0.1827 | 3 | relative | PASS |
| square_100hz | canonical | added_hf_db | 0.1821 | 3 | relative | PASS |
| square_500hz | canonical | added_hf_db | 0.179 | 3 | relative | PASS |
| square_1000hz | canonical | added_hf_db | 0.1805 | 3 | relative | PASS |
| square_2000hz | canonical | added_hf_db | 0.1843 | 3 | relative | PASS |
| square_5000hz | canonical | added_hf_db | 0.1625 | 3 | relative | PASS |
| square_500hz_a005 | canonical | added_hf_db | 0.179 | 3 | relative | PASS |
| dc_step_up | canonical | added_hf_db | 0.1827 | 3 | relative | PASS |
| dc_step_down | canonical | added_hf_db | 0.1827 | 3 | relative | PASS |
| impulse | canonical | added_hf_db | -2.141 | 3 | relative | PASS |
| impulse_train_10ms | canonical | added_hf_db | -2.228 | 3 | relative | PASS |
| tone_burst_1000hz | canonical | added_hf_db | 56.25 | 3 | absolute | PASS |
| tone_burst_10000hz | canonical | added_hf_db | -41.74 | 3 | absolute | PASS |
| tone_burst_19000hz | canonical | added_hf_db | -56.26 | 3 | absolute | PASS |
| sweep_log_20_20k | canonical | added_hf_db | -24.48 | 3 | absolute | PASS |
| pink_noise_s1234 | canonical | added_hf_db | -71.46 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | added_hf_db | -67.96 | 3 | absolute | PASS |
| imd_60hz_7000hz | canonical | added_hf_db | -11.43 | 3 | absolute | PASS |
| square_73hz_held | held_out | added_hf_db | 0.183 | 3 | relative | PASS |
| square_331hz_held | held_out | added_hf_db | 0.185 | 3 | relative | PASS |
| square_1730hz_held | held_out | added_hf_db | 0.1832 | 3 | relative | PASS |
| square_4400hz_held | held_out | added_hf_db | 0.1867 | 3 | relative | PASS |
| tone_burst_3700hz_held | held_out | added_hf_db | 7.326 | 3 | absolute | PASS |
| tone_burst_14300hz_held | held_out | added_hf_db | -35.91 | 3 | absolute | PASS |
| sweep_log_30_19k_held | held_out | added_hf_db | -22.84 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | added_hf_db | -71.78 | 3 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | added_hf_db | -7.884 | 3 | absolute | PASS |

## G8_lb_preservation: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | lb_phase_error_deg | 0.01524 | 15 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_group_delay_error_samples | 6.324 | 600 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_waveform_error_db | -66.3 | -20 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_phase_error_deg | 0.009708 | 15 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_group_delay_error_samples | 0.1834 | 600 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_waveform_error_db | -65.21 | -20 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_phase_error_deg | 0.002228 | 15 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_group_delay_error_samples | 0.9318 | 600 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_waveform_error_db | -67 | -20 | absolute | PASS |

## G9_no_modulation_sidebands: PASS (worst: imd_60hz_7000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| imd_60hz_7000hz | canonical | modulation_sideband_db | -144.7 | -110 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | modulation_sideband_db | -147.3 | -110 | absolute | PASS |
