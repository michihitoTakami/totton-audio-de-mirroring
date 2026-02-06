# Stage2 Quality Benchmark in C++ (Issue #44)

## Purpose

Provide a reproducible C++ quality benchmark for Stage2 (HIE) to quantify:

- impulse response
- step response
- pre-echo (`< 0.1 ms` target)
- transient step overshoot (`< 5%` target)
- per-stage band characteristics (passband ripple / stopband attenuation)

## Configuration-Driven Inputs

### FIR and stage conditions

- File: `cpp/configs/hie_fir_min_phase.ini`
- Injected from config:
- stage taps path (`taps_path`)
- passband/stopband cutoff (`passband_hz`, `stopband_hz`)
- tap constraints (`num_taps`, `min_taps`, `max_taps`)

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
  cpp/configs/hie_fir_min_phase.ini \
  cpp/configs/stage2_quality_benchmark.ini
```

Return code:

- `0`: both thresholds passed
- `2`: quality measurement succeeded but threshold check failed
- `1`: config/load/runtime failure

## Notes

- This benchmark intentionally separates measurement logic from pass/fail policy via config.
- Issue #49 baseline indicates current taps may exceed the `<5%` overshoot target; this benchmark provides the C++ side reproducible measurement path for redesign iterations.
