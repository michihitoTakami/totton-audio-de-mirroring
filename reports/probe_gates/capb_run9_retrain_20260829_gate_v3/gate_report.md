# Stage 1 probe gate report

- all_passed: **False**
- spec_version: 3
- manifest_hash: c15ecd1149d409e3

## G1_lf_ringing: PASS (worst: square_1730hz_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | plateau_rms_after | 0.0001042 | 0.0005 | absolute | PASS |
| square_50hz | canonical | plateau_p2p_after | 0.0009046 | 0.00158 | absolute | PASS |
| square_50hz | canonical | overshoot_after | 0.0317 | 0.09352 | relative | PASS |
| square_100hz | canonical | plateau_rms_after | 8.085e-05 | 0.0005 | absolute | PASS |
| square_100hz | canonical | plateau_p2p_after | 0.0006022 | 0.00158 | absolute | PASS |
| square_100hz | canonical | overshoot_after | 0.0317 | 0.09352 | relative | PASS |
| square_500hz | canonical | plateau_rms_after | 9.112e-05 | 0.0005 | absolute | PASS |
| square_500hz | canonical | plateau_p2p_after | 0.0007987 | 0.002907 | relative | PASS |
| square_500hz | canonical | overshoot_after | 0.03142 | 0.09352 | relative | PASS |
| square_1000hz | canonical | plateau_rms_after | 0.6497 | 0.7178 | relative | PASS |
| square_1000hz | canonical | plateau_p2p_after | 1.064 | 1.198 | relative | PASS |
| square_1000hz | canonical | overshoot_after | 0.03189 | 0.09367 | relative | PASS |
| square_2000hz | canonical | plateau_rms_after | 0.5982 | 0.6707 | relative | PASS |
| square_2000hz | canonical | plateau_p2p_after | 1.065 | 1.295 | relative | PASS |
| square_2000hz | canonical | overshoot_after | 0.0981 | 0.1452 | relative | PASS |
| square_500hz_a005 | canonical | plateau_rms_after | 9.113e-06 | 5e-05 | absolute | PASS |
| square_500hz_a005 | canonical | plateau_p2p_after | 7.987e-05 | 0.0002907 | relative | PASS |
| square_500hz_a005 | canonical | overshoot_after | 0.003142 | 0.01385 | relative | PASS |
| dc_step_up | canonical | plateau_rms_after | 8.085e-05 | 0.0005 | absolute | PASS |
| dc_step_up | canonical | plateau_p2p_after | 0.0006022 | 0.002907 | relative | PASS |
| dc_step_up | canonical | overshoot_after | 0.0317 | 0.09352 | relative | PASS |
| dc_step_down | canonical | plateau_rms_after | 8.895e-05 | 0.0005 | absolute | PASS |
| dc_step_down | canonical | plateau_p2p_after | 0.0006559 | 0.002907 | relative | PASS |
| dc_step_down | canonical | overshoot_after | 0.03183 | 0.09352 | relative | PASS |
| square_73hz_held | held_out | plateau_rms_after | 0.0001117 | 0.0005 | absolute | PASS |
| square_73hz_held | held_out | plateau_p2p_after | 0.0009529 | 0.002907 | relative | PASS |
| square_73hz_held | held_out | overshoot_after | 0.03182 | 0.09352 | relative | PASS |
| square_331hz_held | held_out | plateau_rms_after | 9.324e-05 | 0.0005 | absolute | PASS |
| square_331hz_held | held_out | plateau_p2p_after | 0.0008305 | 0.002907 | relative | PASS |
| square_331hz_held | held_out | overshoot_after | 0.03151 | 0.09352 | relative | PASS |
| square_1730hz_held | held_out | plateau_rms_after | 0.6352 | 0.6892 | relative | PASS |
| square_1730hz_held | held_out | plateau_p2p_after | 1.064 | 1.295 | relative | PASS |
| square_1730hz_held | held_out | overshoot_after | 0.03724 | 0.1111 | relative | PASS |

## G2_hf_ringing: FAIL (worst: square_5000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_5000hz | canonical | plateau_rms_after | 0.5007 | 0.4994 | relative | FAIL |
| square_5000hz | canonical | plateau_p2p_after | 1.385 | 1.294 | relative | FAIL |
| square_5000hz | canonical | overshoot_after | 0.6592 | 0.5485 | relative | FAIL |
| square_4400hz_held | held_out | plateau_rms_after | 0.4632 | 0.5075 | relative | PASS |
| square_4400hz_held | held_out | plateau_p2p_after | 1.083 | 1.295 | relative | PASS |
| square_4400hz_held | held_out | overshoot_after | 0.5438 | 0.5485 | relative | PASS |

## G2b_pre_echo: FAIL (worst: impulse)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| impulse | canonical | pre_echo_energy_after | 9.328e-07 | 2.5e-07 | absolute | FAIL |
| impulse_train_10ms | canonical | pre_echo_energy_after | 9.328e-07 | 2.5e-07 | absolute | FAIL |
| tone_burst_1000hz | canonical | pre_echo_energy_after | 1.549e-15 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | pre_echo_energy_after | 3.092e-15 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | pre_echo_energy_after | 1.841e-12 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | pre_echo_energy_after | 1.728e-15 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | pre_echo_energy_after | 1.207e-14 | 2.5e-07 | absolute | PASS |

## G3_mirror: PASS (worst: sweep_log_20_20k)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| sweep_log_20_20k | canonical | image_rel_db | -73.87 | -65 | absolute | PASS |
| sweep_log_20_20k | canonical | image_peak_rel_db | -68.75 | -65 | absolute | PASS |
| pink_noise_s1234 | canonical | image_rel_db | -91.98 | -65 | absolute | PASS |
| multitone_60_s20260704 | canonical | image_rel_db | -121.8 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_rel_db | -73.9 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_peak_rel_db | -70.79 | -65 | absolute | PASS |
| pink_noise_s5678_held | held_out | image_rel_db | -91.87 | -65 | absolute | PASS |

## G4_flatness: PASS (worst: pink_noise_s5678_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | flatness_dip_db | 2.156e-05 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_boost_db | 2.042e-05 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_hf_dip_db | 0.07935 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_dip_db | 4.853e-05 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_boost_db | 2.246e-05 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_hf_dip_db | 0.01273 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_dip_db | 2.162e-05 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_boost_db | 2.059e-05 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_hf_dip_db | 0.07966 | 3 | absolute | PASS |

## G5_gain: PASS (worst: square_4400hz_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | abs_gain_error_db | 0.01231 | 0.5 | absolute | PASS |
| square_100hz | canonical | abs_gain_error_db | 0.009918 | 0.5 | absolute | PASS |
| square_500hz | canonical | abs_gain_error_db | 0.01638 | 0.5 | absolute | PASS |
| square_1000hz | canonical | abs_gain_error_db | 0.04867 | 0.5 | absolute | PASS |
| square_2000hz | canonical | abs_gain_error_db | 0.1049 | 0.5 | absolute | PASS |
| square_5000hz | canonical | abs_gain_error_db | 0.0002714 | 0.5 | absolute | PASS |
| square_500hz_a005 | canonical | abs_gain_error_db | 0.01638 | 0.5 | absolute | PASS |
| dc_step_up | canonical | abs_gain_error_db | 0.01445 | 0.5 | absolute | PASS |
| dc_step_down | canonical | abs_gain_error_db | 0.01445 | 0.5 | absolute | PASS |
| impulse | canonical | abs_gain_error_db | 0.006474 | 0.5 | absolute | PASS |
| impulse_train_10ms | canonical | abs_gain_error_db | 0.01867 | 0.5 | absolute | PASS |
| tone_burst_1000hz | canonical | abs_gain_error_db | 0.0007954 | 0.5 | absolute | PASS |
| sweep_log_20_20k | canonical | abs_gain_error_db | 0.001511 | 0.5 | absolute | PASS |
| pink_noise_s1234 | canonical | abs_gain_error_db | 0.0002678 | 0.5 | absolute | PASS |
| multitone_60_s20260704 | canonical | abs_gain_error_db | 0.001006 | 0.5 | absolute | PASS |
| square_73hz_held | held_out | abs_gain_error_db | 0.01162 | 0.5 | absolute | PASS |
| square_331hz_held | held_out | abs_gain_error_db | 0.005564 | 0.5 | absolute | PASS |
| square_1730hz_held | held_out | abs_gain_error_db | 0.1025 | 0.5 | absolute | PASS |
| square_4400hz_held | held_out | abs_gain_error_db | 0.2476 | 0.5 | absolute | PASS |
| tone_burst_3700hz_held | held_out | abs_gain_error_db | 0.0009038 | 0.5 | absolute | PASS |
| sweep_log_30_19k_held | held_out | abs_gain_error_db | 0.0009581 | 0.5 | absolute | PASS |
| pink_noise_s5678_held | held_out | abs_gain_error_db | 0.0001922 | 0.5 | absolute | PASS |

## G7_no_added_hf: PASS (worst: tone_burst_1000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | added_hf_db | 0.04472 | 3 | relative | PASS |
| square_100hz | canonical | added_hf_db | 0.04522 | 3 | relative | PASS |
| square_500hz | canonical | added_hf_db | 0.07547 | 3 | relative | PASS |
| square_1000hz | canonical | added_hf_db | 0.07664 | 3 | relative | PASS |
| square_2000hz | canonical | added_hf_db | 0.1109 | 3 | relative | PASS |
| square_5000hz | canonical | added_hf_db | -63.62 | 3 | absolute | PASS |
| square_500hz_a005 | canonical | added_hf_db | 0.07548 | 3 | relative | PASS |
| dc_step_up | canonical | added_hf_db | 0.05078 | 3 | relative | PASS |
| dc_step_down | canonical | added_hf_db | 0.04082 | 3 | relative | PASS |
| impulse | canonical | added_hf_db | -56.94 | 3 | absolute | PASS |
| impulse_train_10ms | canonical | added_hf_db | -57.26 | 3 | absolute | PASS |
| tone_burst_1000hz | canonical | added_hf_db | 30.09 | 3 | absolute | PASS |
| tone_burst_10000hz | canonical | added_hf_db | -64.71 | 3 | absolute | PASS |
| tone_burst_19000hz | canonical | added_hf_db | -82.44 | 3 | absolute | PASS |
| sweep_log_20_20k | canonical | added_hf_db | 6.537 | 3 | absolute | PASS |
| pink_noise_s1234 | canonical | added_hf_db | -57.13 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | added_hf_db | -85.85 | 3 | absolute | PASS |
| square_73hz_held | held_out | added_hf_db | 0.04409 | 3 | relative | PASS |
| square_331hz_held | held_out | added_hf_db | 0.06147 | 3 | relative | PASS |
| square_1730hz_held | held_out | added_hf_db | 0.1131 | 3 | relative | PASS |
| square_4400hz_held | held_out | added_hf_db | -0.6011 | 3 | relative | PASS |
| tone_burst_3700hz_held | held_out | added_hf_db | -11.33 | 3 | absolute | PASS |
| tone_burst_14300hz_held | held_out | added_hf_db | -79.17 | 3 | absolute | PASS |
| sweep_log_30_19k_held | held_out | added_hf_db | 8.009 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | added_hf_db | -56.73 | 3 | absolute | PASS |

## G8_lb_preservation: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | lb_phase_error_deg | 0.01428 | 15 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_group_delay_error_samples | 5.489 | 600 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_waveform_error_db | -49.39 | -20 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_phase_error_deg | 0.009977 | 15 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_group_delay_error_samples | 0.1316 | 600 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_waveform_error_db | -44.67 | -20 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_phase_error_deg | 0.003711 | 15 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_group_delay_error_samples | 1.414 | 600 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_waveform_error_db | -48.79 | -20 | absolute | PASS |
