# Stage2 Quality Benchmark in C++ (Issue #44)

## Purpose

Provide a reproducible C++ quality benchmark for the linear-phase Stage2 transport FIR to quantify:

- impulse response
- step response
- pre-echo (`< 0.1 ms` target)
- transient step overshoot (`< 5%` target)
- per-stage band characteristics (passband ripple / stopband attenuation)

## Configuration-Driven Inputs

### FIR and stage conditions

- File: `cpp/configs/hirate_fir_linear_phase.ini`
- Injected from config:
- stage taps path (`taps_path`)
- design mode (`design_kind=hirate_linear`)
- passband/stopband measurement bands (`passband_hz`, `stopband_hz`)
- per-stage Nyquist cutoff (`cutoff_hz`)
- fixed tap count (`num_taps`: 255, 63, 39)

### Benchmark conditions and thresholds

- File: `cpp/configs/stage2_quality_benchmark.ini`
- Injected from config:
- probe lengths (`impulse_length`, `step_length`)
- transition position (`transition_index`)
- settle/quantile params (`settle_fraction`, `reference_quantile`)
- quality thresholds (`max_pre_echo_ms`, `max_step_overshoot_ratio`)

## Build and Run

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
./cpp/build/tadm_stage2_quality_bench \
  cpp/configs/hirate_fir_linear_phase.ini \
  cpp/configs/stage2_quality_benchmark.ini
```

Return code:

- `0`: both thresholds passed
- `2`: quality measurement succeeded but threshold check failed
- `1`: config/load/runtime failure

## Notes

- This benchmark intentionally separates measurement logic from pass/fail policy via config.
- The FIR contract matches Totton Audio NN: Kaiser beta 16, linear phase, and per-stage input-Nyquist cutoff.
- The raw-step overshoot is about 13.9%, improved from the retired minimum-phase cascade's roughly 33%, but it does not meet the aspirational `<5%` target. The threshold remains unchanged and the CLI therefore returns `2` for this diagnostic.
