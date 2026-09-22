# Stage 1 probe gate report

- all_passed: **True**
- spec_version: 7
- manifest_hash: 085959477798407d

## G1_lf_ringing: PASS (worst: square_73hz_held)

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

## G2_hf_ringing: PASS (worst: square_1000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_1000hz | canonical | plateau_rms_after | 0.0002043 | 0.0005 | absolute | PASS |
| square_1000hz | canonical | plateau_p2p_after | 0.0007605 | 0.003495 | relative | PASS |
| square_1000hz | canonical | overshoot_after | 0.03752 | 0.0962 | relative | PASS |

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
| impulse | canonical | pre_echo_energy_after | 4.579e-11 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | pre_echo_energy_after | 4.578e-11 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | pre_echo_energy_after | 3.776e-21 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | pre_echo_energy_after | 2.864e-21 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | pre_echo_energy_after | 1.025e-16 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | pre_echo_energy_after | 1.936e-20 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | pre_echo_energy_after | 3.656e-19 | 2.5e-07 | absolute | PASS |

## G2c_post_echo: PASS (worst: impulse)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| impulse | canonical | post_echo_energy_after | 4.421e-11 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | post_echo_energy_after | 4.421e-11 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | post_echo_energy_after | 2.317e-21 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | post_echo_energy_after | 4.692e-21 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | post_echo_energy_after | 9.465e-17 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | post_echo_energy_after | 2.75e-25 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | post_echo_energy_after | 1.021e-19 | 2.5e-07 | absolute | PASS |

## G3_mirror: PASS (worst: sweep_log_20_20k)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| sweep_log_20_20k | canonical | image_rel_db | -133.1 | -65 | absolute | PASS |
| sweep_log_20_20k | canonical | image_peak_rel_db | -118.7 | -65 | absolute | PASS |
| pink_noise_s1234 | canonical | image_rel_db | -129.5 | -65 | absolute | PASS |
| multitone_60_s20260704 | canonical | image_rel_db | -131.5 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_rel_db | -133.1 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_peak_rel_db | -123.3 | -65 | absolute | PASS |
| pink_noise_s5678_held | held_out | image_rel_db | -130.4 | -65 | absolute | PASS |

## G4_flatness: PASS (worst: pink_noise_s5678_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | flatness_dip_db | 6.788e-07 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_boost_db | 0 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_hf_dip_db | 9.619e-06 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_dip_db | 7.371e-07 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_boost_db | 0 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_hf_dip_db | 4.178e-07 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_dip_db | 1.357e-06 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_boost_db | 0 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_hf_dip_db | 1.292e-05 | 3 | absolute | PASS |

## G5_gain: PASS (worst: square_5000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | abs_gain_error_db | 0.003795 | 0.5 | absolute | PASS |
| square_100hz | canonical | abs_gain_error_db | 0.008028 | 0.5 | absolute | PASS |
| square_500hz | canonical | abs_gain_error_db | 0.01649 | 0.5 | absolute | PASS |
| square_1000hz | canonical | abs_gain_error_db | 0.04896 | 0.5 | absolute | PASS |
| square_2000hz | canonical | abs_gain_error_db | 0.106 | 0.5 | absolute | PASS |
| square_5000hz | canonical | abs_gain_error_db | 0.3754 | 0.5 | absolute | PASS |
| square_500hz_a005 | canonical | abs_gain_error_db | 0.01649 | 0.5 | absolute | PASS |
| dc_step_up | canonical | abs_gain_error_db | 1.878e-05 | 0.5 | absolute | PASS |
| dc_step_down | canonical | abs_gain_error_db | 1.878e-05 | 0.5 | absolute | PASS |
| impulse | canonical | abs_gain_error_db | 0.1562 | 0.5 | absolute | PASS |
| impulse_train_10ms | canonical | abs_gain_error_db | 0.1663 | 0.5 | absolute | PASS |
| tone_burst_1000hz | canonical | abs_gain_error_db | 0.0007888 | 0.5 | absolute | PASS |
| sweep_log_20_20k | canonical | abs_gain_error_db | 0.0003742 | 0.5 | absolute | PASS |
| pink_noise_s1234 | canonical | abs_gain_error_db | 0.0002707 | 0.5 | absolute | PASS |
| multitone_60_s20260704 | canonical | abs_gain_error_db | 0.001003 | 0.5 | absolute | PASS |
| imd_60hz_7000hz | canonical | abs_gain_error_db | 0.001426 | 0.5 | absolute | PASS |
| square_73hz_held | held_out | abs_gain_error_db | 0.006029 | 0.5 | absolute | PASS |
| square_331hz_held | held_out | abs_gain_error_db | 0.005608 | 0.5 | absolute | PASS |
| square_1730hz_held | held_out | abs_gain_error_db | 0.1031 | 0.5 | absolute | PASS |
| square_4400hz_held | held_out | abs_gain_error_db | 0.2732 | 0.5 | absolute | PASS |
| tone_burst_3700hz_held | held_out | abs_gain_error_db | 0.06263 | 0.5 | absolute | PASS |
| sweep_log_30_19k_held | held_out | abs_gain_error_db | 0.0004126 | 0.5 | absolute | PASS |
| pink_noise_s5678_held | held_out | abs_gain_error_db | 0.0001952 | 0.5 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | abs_gain_error_db | 0.001976 | 0.5 | absolute | PASS |

## G7_no_added_hf: PASS (worst: tone_burst_1000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | added_hf_db | 0.136 | 3 | relative | PASS |
| square_100hz | canonical | added_hf_db | 0.1361 | 3 | relative | PASS |
| square_500hz | canonical | added_hf_db | 0.134 | 3 | relative | PASS |
| square_1000hz | canonical | added_hf_db | 0.1275 | 3 | relative | PASS |
| square_2000hz | canonical | added_hf_db | 0.1939 | 3 | relative | PASS |
| square_5000hz | canonical | added_hf_db | 0.1551 | 3 | relative | PASS |
| square_500hz_a005 | canonical | added_hf_db | 0.134 | 3 | relative | PASS |
| dc_step_up | canonical | added_hf_db | 0.1366 | 3 | relative | PASS |
| dc_step_down | canonical | added_hf_db | 0.1366 | 3 | relative | PASS |
| impulse | canonical | added_hf_db | -4.15 | 3 | relative | PASS |
| impulse_train_10ms | canonical | added_hf_db | -4.192 | 3 | relative | PASS |
| tone_burst_1000hz | canonical | added_hf_db | 41.91 | 3 | absolute | PASS |
| tone_burst_10000hz | canonical | added_hf_db | -5.314 | 3 | relative | PASS |
| tone_burst_19000hz | canonical | added_hf_db | -5.763 | 3 | relative | PASS |
| sweep_log_20_20k | canonical | added_hf_db | -52.69 | 3 | absolute | PASS |
| pink_noise_s1234 | canonical | added_hf_db | -94.61 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | added_hf_db | -95.55 | 3 | absolute | PASS |
| imd_60hz_7000hz | canonical | added_hf_db | -40.4 | 3 | absolute | PASS |
| square_73hz_held | held_out | added_hf_db | 0.1349 | 3 | relative | PASS |
| square_331hz_held | held_out | added_hf_db | 0.1326 | 3 | relative | PASS |
| square_1730hz_held | held_out | added_hf_db | 0.1662 | 3 | relative | PASS |
| square_4400hz_held | held_out | added_hf_db | 0.1422 | 3 | relative | PASS |
| tone_burst_3700hz_held | held_out | added_hf_db | 1.701 | 3 | absolute | PASS |
| tone_burst_14300hz_held | held_out | added_hf_db | -5.597 | 3 | relative | PASS |
| sweep_log_30_19k_held | held_out | added_hf_db | -51.15 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | added_hf_db | -95.28 | 3 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | added_hf_db | -35.48 | 3 | absolute | PASS |

## G8_lb_preservation: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | lb_phase_error_deg | 0.01386 | 15 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_group_delay_error_samples | 5.323 | 600 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_waveform_error_db | -49.35 | -20 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_phase_error_deg | 0.01003 | 15 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_group_delay_error_samples | 0.1358 | 600 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_waveform_error_db | -44.53 | -20 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_phase_error_deg | 0.004621 | 15 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_group_delay_error_samples | 1.761 | 600 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_waveform_error_db | -48.74 | -20 | absolute | PASS |

## G9_no_modulation_sidebands: PASS (worst: imd_60hz_7000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| imd_60hz_7000hz | canonical | modulation_sideband_db | -137.5 | -110 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | modulation_sideband_db | -144.3 | -110 | absolute | PASS |
