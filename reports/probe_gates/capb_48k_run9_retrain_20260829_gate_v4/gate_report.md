# Stage 1 probe gate report

- all_passed: **False**
- spec_version: 4
- manifest_hash: 6529deadaab177fe

## G1_lf_ringing: FAIL (worst: square_2000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_2000hz | canonical | plateau_rms_after | 0.6339 | 0.6204 | relative | FAIL |
| square_50hz | canonical | plateau_rms_after | 7.116e-05 | 0.0005 | absolute | PASS |
| square_50hz | canonical | plateau_p2p_after | 0.0006306 | 0.00158 | absolute | PASS |
| square_50hz | canonical | overshoot_after | 0.02422 | 0.07179 | relative | PASS |
| square_100hz | canonical | plateau_rms_after | 7.116e-05 | 0.0005 | absolute | PASS |
| square_100hz | canonical | plateau_p2p_after | 0.0006306 | 0.00158 | absolute | PASS |
| square_100hz | canonical | overshoot_after | 0.02422 | 0.07179 | relative | PASS |
| square_500hz | canonical | plateau_rms_after | 4.321e-05 | 0.0005 | absolute | PASS |
| square_500hz | canonical | plateau_p2p_after | 0.0003605 | 0.00158 | absolute | PASS |
| square_500hz | canonical | overshoot_after | 0.02412 | 0.07179 | relative | PASS |
| square_1000hz | canonical | plateau_rms_after | 0.6593 | 0.7237 | relative | PASS |
| square_1000hz | canonical | plateau_p2p_after | 1.05 | 1.174 | relative | PASS |
| square_1000hz | canonical | overshoot_after | 0.02467 | 0.07187 | relative | PASS |
| square_2000hz | canonical | plateau_p2p_after | 1.05 | 1.247 | relative | PASS |
| square_2000hz | canonical | overshoot_after | 0.02993 | 0.2082 | relative | PASS |
| square_500hz_a005 | canonical | plateau_rms_after | 4.611e-06 | 5e-05 | absolute | PASS |
| square_500hz_a005 | canonical | plateau_p2p_after | 3.772e-05 | 0.000158 | absolute | PASS |
| square_500hz_a005 | canonical | overshoot_after | 0.002412 | 0.01168 | relative | PASS |
| dc_step_up | canonical | plateau_rms_after | 7.116e-05 | 0.0005 | absolute | PASS |
| dc_step_up | canonical | plateau_p2p_after | 0.0006306 | 0.001951 | relative | PASS |
| dc_step_up | canonical | overshoot_after | 0.02422 | 0.07179 | relative | PASS |
| dc_step_down | canonical | plateau_rms_after | 7.406e-05 | 0.0005 | absolute | PASS |
| dc_step_down | canonical | plateau_p2p_after | 0.0006562 | 0.001951 | relative | PASS |
| dc_step_down | canonical | overshoot_after | 0.02423 | 0.07179 | relative | PASS |
| square_73hz_held | held_out | plateau_rms_after | 7.293e-05 | 0.0005 | absolute | PASS |
| square_73hz_held | held_out | plateau_p2p_after | 0.0006468 | 0.00158 | absolute | PASS |
| square_73hz_held | held_out | overshoot_after | 0.02422 | 0.07179 | relative | PASS |
| square_331hz_held | held_out | plateau_rms_after | 4.705e-05 | 0.0005 | absolute | PASS |
| square_331hz_held | held_out | plateau_p2p_after | 0.0004225 | 0.00158 | absolute | PASS |
| square_331hz_held | held_out | overshoot_after | 0.02414 | 0.07179 | relative | PASS |
| square_1730hz_held | held_out | plateau_rms_after | 0.6366 | 0.694 | relative | PASS |
| square_1730hz_held | held_out | plateau_p2p_after | 1.05 | 1.247 | relative | PASS |
| square_1730hz_held | held_out | overshoot_after | 0.02481 | 0.07952 | relative | PASS |

## G2_hf_ringing: PASS (worst: square_4400hz_held)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_5000hz | canonical | plateau_rms_after | 0.5413 | 0.563 | relative | PASS |
| square_5000hz | canonical | plateau_p2p_after | 1.057 | 1.247 | relative | PASS |
| square_5000hz | canonical | overshoot_after | 0.1842 | 0.283 | relative | PASS |
| square_4400hz_held | held_out | plateau_rms_after | 0.5477 | 0.5631 | relative | PASS |
| square_4400hz_held | held_out | plateau_p2p_after | 1.053 | 1.247 | relative | PASS |
| square_4400hz_held | held_out | overshoot_after | 0.1719 | 0.283 | relative | PASS |

## G2b_pre_echo: PASS (worst: impulse)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| impulse | canonical | pre_echo_energy_after | 2.42e-07 | 2.5e-07 | absolute | PASS |
| impulse_train_10ms | canonical | pre_echo_energy_after | 2.42e-07 | 2.5e-07 | absolute | PASS |
| tone_burst_1000hz | canonical | pre_echo_energy_after | 1.195e-16 | 2.5e-07 | absolute | PASS |
| tone_burst_10000hz | canonical | pre_echo_energy_after | 2.299e-16 | 2.5e-07 | absolute | PASS |
| tone_burst_19000hz | canonical | pre_echo_energy_after | 2.948e-14 | 2.5e-07 | absolute | PASS |
| tone_burst_3700hz_held | held_out | pre_echo_energy_after | 1.132e-16 | 2.5e-07 | absolute | PASS |
| tone_burst_14300hz_held | held_out | pre_echo_energy_after | 8.109e-16 | 2.5e-07 | absolute | PASS |

## G3_mirror: PASS (worst: sweep_log_20_20k)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| sweep_log_20_20k | canonical | image_rel_db | -116.1 | -65 | absolute | PASS |
| sweep_log_20_20k | canonical | image_peak_rel_db | -105.9 | -65 | absolute | PASS |
| pink_noise_s1234 | canonical | image_rel_db | -111.7 | -65 | absolute | PASS |
| multitone_60_s20260704 | canonical | image_rel_db | -114.8 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_rel_db | -116 | -65 | absolute | PASS |
| sweep_log_30_19k_held | held_out | image_peak_rel_db | -108.1 | -65 | absolute | PASS |
| pink_noise_s5678_held | held_out | image_rel_db | -112.4 | -65 | absolute | PASS |

## G4_flatness: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | flatness_dip_db | 2.593e-05 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_boost_db | 1.28e-08 | 1 | absolute | PASS |
| pink_noise_s1234 | canonical | flatness_hf_dip_db | 0.001387 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_dip_db | 5.142e-05 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_boost_db | 2.371e-05 | 1 | absolute | PASS |
| multitone_60_s20260704 | canonical | flatness_hf_dip_db | 0.0001025 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_dip_db | 2.57e-05 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_boost_db | 0 | 1 | absolute | PASS |
| pink_noise_s5678_held | held_out | flatness_hf_dip_db | 0.001268 | 3 | absolute | PASS |

## G5_gain: PASS (worst: square_5000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | abs_gain_error_db | 0.01101 | 0.5 | absolute | PASS |
| square_100hz | canonical | abs_gain_error_db | 0.00714 | 0.5 | absolute | PASS |
| square_500hz | canonical | abs_gain_error_db | 0.01872 | 0.5 | absolute | PASS |
| square_1000hz | canonical | abs_gain_error_db | 0.05308 | 0.5 | absolute | PASS |
| square_2000hz | canonical | abs_gain_error_db | 0.1159 | 0.5 | absolute | PASS |
| square_5000hz | canonical | abs_gain_error_db | 0.3793 | 0.5 | absolute | PASS |
| square_500hz_a005 | canonical | abs_gain_error_db | 0.01872 | 0.5 | absolute | PASS |
| dc_step_up | canonical | abs_gain_error_db | 0.01536 | 0.5 | absolute | PASS |
| dc_step_down | canonical | abs_gain_error_db | 0.01536 | 0.5 | absolute | PASS |
| impulse | canonical | abs_gain_error_db | 0.009052 | 0.5 | absolute | PASS |
| impulse_train_10ms | canonical | abs_gain_error_db | 0.02154 | 0.5 | absolute | PASS |
| tone_burst_1000hz | canonical | abs_gain_error_db | 0.001175 | 0.5 | absolute | PASS |
| sweep_log_20_20k | canonical | abs_gain_error_db | 0.001054 | 0.5 | absolute | PASS |
| pink_noise_s1234 | canonical | abs_gain_error_db | 0.001426 | 0.5 | absolute | PASS |
| multitone_60_s20260704 | canonical | abs_gain_error_db | 0.001429 | 0.5 | absolute | PASS |
| imd_60hz_7000hz | canonical | abs_gain_error_db | 0.001692 | 0.5 | absolute | PASS |
| square_73hz_held | held_out | abs_gain_error_db | 0.009162 | 0.5 | absolute | PASS |
| square_331hz_held | held_out | abs_gain_error_db | 0.007057 | 0.5 | absolute | PASS |
| square_1730hz_held | held_out | abs_gain_error_db | 0.1099 | 0.5 | absolute | PASS |
| square_4400hz_held | held_out | abs_gain_error_db | 0.2885 | 0.5 | absolute | PASS |
| tone_burst_3700hz_held | held_out | abs_gain_error_db | 0.001056 | 0.5 | absolute | PASS |
| sweep_log_30_19k_held | held_out | abs_gain_error_db | 0.001759 | 0.5 | absolute | PASS |
| pink_noise_s5678_held | held_out | abs_gain_error_db | 0.001833 | 0.5 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | abs_gain_error_db | 0.001955 | 0.5 | absolute | PASS |

## G7_no_added_hf: PASS (worst: tone_burst_1000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| square_50hz | canonical | added_hf_db | 0.08412 | 3 | relative | PASS |
| square_100hz | canonical | added_hf_db | 0.08361 | 3 | relative | PASS |
| square_500hz | canonical | added_hf_db | 0.1301 | 3 | relative | PASS |
| square_1000hz | canonical | added_hf_db | 0.1397 | 3 | relative | PASS |
| square_2000hz | canonical | added_hf_db | 0.1497 | 3 | relative | PASS |
| square_5000hz | canonical | added_hf_db | -0.05071 | 3 | relative | PASS |
| square_500hz_a005 | canonical | added_hf_db | 0.1301 | 3 | relative | PASS |
| dc_step_up | canonical | added_hf_db | 0.08288 | 3 | relative | PASS |
| dc_step_down | canonical | added_hf_db | 0.07739 | 3 | relative | PASS |
| impulse | canonical | added_hf_db | -76.74 | 3 | absolute | PASS |
| impulse_train_10ms | canonical | added_hf_db | -76.67 | 3 | absolute | PASS |
| tone_burst_1000hz | canonical | added_hf_db | 55.88 | 3 | absolute | PASS |
| tone_burst_10000hz | canonical | added_hf_db | -42.38 | 3 | absolute | PASS |
| tone_burst_19000hz | canonical | added_hf_db | -73.07 | 3 | absolute | PASS |
| sweep_log_20_20k | canonical | added_hf_db | -24.48 | 3 | absolute | PASS |
| pink_noise_s1234 | canonical | added_hf_db | -71.72 | 3 | absolute | PASS |
| multitone_60_s20260704 | canonical | added_hf_db | -68.11 | 3 | absolute | PASS |
| imd_60hz_7000hz | canonical | added_hf_db | -12.1 | 3 | absolute | PASS |
| square_73hz_held | held_out | added_hf_db | 0.08506 | 3 | relative | PASS |
| square_331hz_held | held_out | added_hf_db | 0.124 | 3 | relative | PASS |
| square_1730hz_held | held_out | added_hf_db | 0.1508 | 3 | relative | PASS |
| square_4400hz_held | held_out | added_hf_db | -0.4278 | 3 | relative | PASS |
| tone_burst_3700hz_held | held_out | added_hf_db | 6.745 | 3 | absolute | PASS |
| tone_burst_14300hz_held | held_out | added_hf_db | -61.9 | 3 | absolute | PASS |
| sweep_log_30_19k_held | held_out | added_hf_db | -22.86 | 3 | absolute | PASS |
| pink_noise_s5678_held | held_out | added_hf_db | -72.02 | 3 | absolute | PASS |
| imd_83hz_6311hz_held | held_out | added_hf_db | -8.695 | 3 | absolute | PASS |

## G8_lb_preservation: PASS (worst: pink_noise_s1234)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| pink_noise_s1234 | canonical | lb_phase_error_deg | 0.01509 | 15 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_group_delay_error_samples | 6.262 | 600 | absolute | PASS |
| pink_noise_s1234 | canonical | lb_waveform_error_db | -66.32 | -20 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_phase_error_deg | 0.009917 | 15 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_group_delay_error_samples | 0.187 | 600 | absolute | PASS |
| multitone_60_s20260704 | canonical | lb_waveform_error_db | -65.21 | -20 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_phase_error_deg | 0.002234 | 15 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_group_delay_error_samples | 0.935 | 600 | absolute | PASS |
| pink_noise_s5678_held | held_out | lb_waveform_error_db | -67 | -20 | absolute | PASS |

## G9_no_modulation_sidebands: FAIL (worst: imd_60hz_7000hz)

| probe | tier | metric | value | threshold | binding | pass |
|---|---|---|---|---|---|---|
| imd_60hz_7000hz | canonical | modulation_sideband_db | -69.7 | -110 | absolute | FAIL |
| imd_83hz_6311hz_held | held_out | modulation_sideband_db | -119.5 | -110 | absolute | PASS |
