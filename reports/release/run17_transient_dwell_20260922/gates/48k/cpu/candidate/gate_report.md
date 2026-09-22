# Stage 1 probe gate report

- all_passed: **True**
- spec_version: 7
- manifest_hash: 6529deadaab177fe

## G1_lf_ringing: PASS (worst: square_500hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | plateau_rms_after | 0.0001742 | 0.0005 | absolute | PASS |
| square_50hz | canonical | plateau_p2p_after | 0.0006528 | 0.00187 | relative | PASS |
| square_50hz | canonical | overshoot_after | 0.03001 | 0.07761 | relative | PASS |
| square_100hz | canonical | plateau_rms_after | 0.0001742 | 0.0005 | absolute | PASS |
| square_100hz | canonical | plateau_p2p_after | 0.0006528 | 0.001952 | relative | PASS |
| square_100hz | canonical | overshoot_after | 0.03001 | 0.07761 | relative | PASS |
| square_500hz | canonical | plateau_rms_after | 0.000174 | 0.0005 | absolute | PASS |
| square_500hz | canonical | plateau_p2p_after | 0.0006529 | 0.001952 | relative | PASS |
| square_500hz | canonical | overshoot_after | 0.03001 | 0.07761 | relative | PASS |
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
| square_331hz_held | held_out | plateau_p2p_after | 0.0006529 | 0.002184 | relative | PASS |
| square_331hz_held | held_out | overshoot_after | 0.03001 | 0.07761 | relative | PASS |

## G2_hf_ringing: PASS (worst: square_1000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_1000hz | canonical | plateau_rms_after | 0.0001777 | 0.0005 | absolute | PASS |
| square_1000hz | canonical | plateau_p2p_after | 0.0006678 | 0.001952 | relative | PASS |
| square_1000hz | canonical | overshoot_after | 0.03002 | 0.07761 | relative | PASS |

Not gated (no row emitted - this is NOT a pass):

| probe | tier | metric group | half period | reason |
|---|---|---|---|---|
| square_2000hz | canonical | plateau_ripple_and_overshoot | 0.25 ms | no settled plateau exists: half period 0.25 ms leaves less than 0.1 ms between the 0.1 ms settling start and the 0.1 ms next-edge guard |
| square_5000hz | canonical | plateau_ripple_and_overshoot | 0.1 ms | no settled plateau exists: half period 0.1 ms leaves less than 0.1 ms between the 0.1 ms settling start and the 0.1 ms next-edge guard |
| square_1730hz_held | held_out | plateau_ripple_and_overshoot | 0.289 ms | no settled plateau exists: half period 0.289 ms leaves less than 0.1 ms between the 0.1 ms settling start and the 0.1 ms next-edge guard |
| square_4400hz_held | held_out | plateau_ripple_and_overshoot | 0.1136 ms | no settled plateau exists: half period 0.1136 ms leaves less than 0.1 ms between the 0.1 ms settling start and the 0.1 ms next-edge guard |

## G2b_pre_echo: PASS (worst: impulse)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| impulse | canonical | pre_echo_energy_after | 3.993e-11 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | pre_echo_energy_after | 3.854e-11 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | pre_echo_energy_after | 1.149e-20 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | pre_echo_energy_after | 1.462e-20 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | pre_echo_energy_after | 9.64e-18 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | pre_echo_energy_after | 1.732e-21 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | pre_echo_energy_after | 1.283e-19 | 2.5e-07 | absolute | PASS |

## G2c_post_echo: PASS (worst: impulse_train_10ms)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| impulse | canonical | post_echo_energy_after | 3.25e-11 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | post_echo_energy_after | 3.25e-11 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | post_echo_energy_after | 9.826e-23 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | post_echo_energy_after | 2.167e-28 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | post_echo_energy_after | 2.063e-25 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | post_echo_energy_after | 1.287e-26 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | post_echo_energy_after | 1.587e-27 | 2.5e-07 | absolute | PASS |

## G3_mirror: PASS (worst: sweep_log_20_20k)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| sweep_log_20_20k | canonical | image_rel_db | -133.5 | -65 | absolute | PASS |
| sweep_log_20_20k | canonical | image_peak_rel_db | -128.2 | -65 | absolute | PASS |
| pink_noise_s1234 | canonical | image_rel_db | -130 | -65 | absolute | PASS |
| multitone_60_s20260704 | canonical | image_rel_db | -131.9 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_rel_db | -133.5 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_peak_rel_db | -133.4 | -65 | absolute | PASS |
| pink_noise_s5678_held | held_out | image_rel_db | -130.8 | -65 | absolute | PASS |

## G4_flatness: PASS (worst: pink_noise_s5678_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | flatness_dip_db | 7.203e-07 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_boost_db | 0 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_hf_dip_db | 4.258e-07 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_dip_db | 5.688e-07 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_boost_db | 0 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_hf_dip_db | 4.453e-08 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_dip_db | 7.334e-07 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_boost_db | 0 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_hf_dip_db | 2.514e-07 | 3 | absolute | PASS |

## G5_gain: PASS (worst: square_5000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | abs_gain_error_db | 0.002303 | 0.5 | absolute | PASS |
| square_100hz | canonical | abs_gain_error_db | 0.004801 | 0.5 | absolute | PASS |
| square_500hz | canonical | abs_gain_error_db | 0.01882 | 0.5 | absolute | PASS |
| square_1000hz | canonical | abs_gain_error_db | 0.05331 | 0.5 | absolute | PASS |
| square_2000hz | canonical | abs_gain_error_db | 0.1141 | 0.5 | absolute | PASS |
| square_5000hz | canonical | abs_gain_error_db | 0.3784 | 0.5 | absolute | PASS |
| square_500hz_a005 | canonical | abs_gain_error_db | 0.01882 | 0.5 | absolute | PASS |
| dc_step_up | canonical | abs_gain_error_db | 1.347e-05 | 0.5 | absolute | PASS |
| dc_step_down | canonical | abs_gain_error_db | 1.347e-05 | 0.5 | absolute | PASS |
| impulse | canonical | abs_gain_error_db | 0.1473 | 0.5 | absolute | PASS |
| impulse_train_10ms | canonical | abs_gain_error_db | 0.1622 | 0.5 | absolute | PASS |
| tone_burst_1000hz | canonical | abs_gain_error_db | 0.001179 | 0.5 | absolute | PASS |
| sweep_log_20_20k | canonical | abs_gain_error_db | 0.002428 | 0.5 | absolute | PASS |
| pink_noise_s1234 | canonical | abs_gain_error_db | 0.001406 | 0.5 | absolute | PASS |
| multitone_60_s20260704 | canonical | abs_gain_error_db | 0.001417 | 0.5 | absolute | PASS |
| imd_60hz_7000hz | canonical | abs_gain_error_db | 0.001638 | 0.5 | absolute | PASS |
| square_73hz_held | held_out | abs_gain_error_db | 0.002403 | 0.5 | absolute | PASS |
| square_331hz_held | held_out | abs_gain_error_db | 0.007093 | 0.5 | absolute | PASS |
| square_1730hz_held | held_out | abs_gain_error_db | 0.1089 | 0.5 | absolute | PASS |
| square_4400hz_held | held_out | abs_gain_error_db | 0.3024 | 0.5 | absolute | PASS |
| tone_burst_3700hz_held | held_out | abs_gain_error_db | 0.07428 | 0.5 | absolute | PASS |
| sweep_log_30_19k_held | held_out | abs_gain_error_db | 0.002565 | 0.5 | absolute | PASS |
| pink_noise_s5678_held | held_out | abs_gain_error_db | 0.001814 | 0.5 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | abs_gain_error_db | 0.00193 | 0.5 | absolute | PASS |

## G7_no_added_hf: PASS (worst: tone_burst_1000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | added_hf_db | 0.1476 | 3 | relative | PASS |
| square_100hz | canonical | added_hf_db | 0.1644 | 3 | relative | PASS |
| square_500hz | canonical | added_hf_db | 0.179 | 3 | relative | PASS |
| square_1000hz | canonical | added_hf_db | 0.1775 | 3 | relative | PASS |
| square_2000hz | canonical | added_hf_db | 0.004462 | 3 | relative | PASS |
| square_5000hz | canonical | added_hf_db | -0.07576 | 3 | relative | PASS |
| square_500hz_a005 | canonical | added_hf_db | 0.179 | 3 | relative | PASS |
| dc_step_up | canonical | added_hf_db | 0.1832 | 3 | relative | PASS |
| dc_step_down | canonical | added_hf_db | 0.1832 | 3 | relative | PASS |
| impulse | canonical | added_hf_db | -11.07 | 3 | relative | PASS |
| impulse_train_10ms | canonical | added_hf_db | -10.9 | 3 | relative | PASS |
| tone_burst_1000hz | canonical | added_hf_db | 45.22 | 3 | absolute | PASS |
| tone_burst_10000hz | canonical | added_hf_db | -5.988 | 3 | absolute | PASS |
| tone_burst_19000hz | canonical | added_hf_db | -7.364 | 3 | relative | PASS |
| sweep_log_20_20k | canonical | added_hf_db | -41.92 | 3 | absolute | PASS |
| pink_noise_s1234 | canonical | added_hf_db | -90.04 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | added_hf_db | -85.14 | 3 | absolute | PASS |
| imd_60hz_7000hz | canonical | added_hf_db | -29.84 | 3 | absolute | PASS |
| square_73hz_held | held_out | added_hf_db | 0.189 | 3 | relative | PASS |
| square_331hz_held | held_out | added_hf_db | 0.1851 | 3 | relative | PASS |
| square_1730hz_held | held_out | added_hf_db | 0.0653 | 3 | relative | PASS |
| square_4400hz_held | held_out | added_hf_db | -0.01407 | 3 | relative | PASS |
| tone_burst_3700hz_held | held_out | added_hf_db | 5.427 | 3 | absolute | PASS |
| tone_burst_14300hz_held | held_out | added_hf_db | -7.118 | 3 | relative | PASS |
| sweep_log_30_19k_held | held_out | added_hf_db | -40.36 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | added_hf_db | -90.45 | 3 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | added_hf_db | -24.88 | 3 | absolute | PASS |

## G8_lb_preservation: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | lb_phase_error_deg | 0.01776 | 15 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_group_delay_error_samples | 7.363 | 600 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_waveform_error_db | -65.93 | -20 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_phase_error_deg | 0.009631 | 15 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_group_delay_error_samples | 0.1845 | 600 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_waveform_error_db | -65.23 | -20 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_phase_error_deg | 0.002958 | 15 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_group_delay_error_samples | 1.241 | 600 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_waveform_error_db | -66.99 | -20 | absolute | PASS |

## G9_no_modulation_sidebands: PASS (worst: imd_60hz_7000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| imd_60hz_7000hz | canonical | modulation_sideband_db | -138.2 | -110 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | modulation_sideband_db | -144.9 | -110 | absolute | PASS |
