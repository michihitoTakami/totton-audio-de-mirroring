#ifndef TOTTON_AUDIO_DE_MIRRORING_DSP_STAGE2_QUALITY_BENCHMARK_H
#define TOTTON_AUDIO_DE_MIRRORING_DSP_STAGE2_QUALITY_BENCHMARK_H

#include "totton_audio_de_mirroring/dsp/fir_config.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace totton_audio_de_mirroring::dsp {

struct Stage2BenchmarkConfig {
    double source_sample_rate_hz = 88200.0;
    std::size_t impulse_length = 4096;
    std::size_t step_length = 4096;
    std::size_t transition_index = 512;
    double settle_fraction = 0.75;
    double reference_quantile = 0.95;
    double pre_echo_threshold_ratio = 1e-3;
    double max_pre_echo_ms = 0.1;
    double max_step_overshoot_ratio = 0.05;
};

struct StageBandMetrics {
    std::string name;
    std::size_t taps_count = 0;
    double passband_hz = 0.0;
    double stopband_hz = 0.0;
    double passband_ripple_db = 0.0;
    double stopband_atten_db = 0.0;
};

struct Stage2QualityMetrics {
    double output_sample_rate_hz = 0.0;
    double pre_echo_ms = 0.0;
    double step_overshoot_ratio = 0.0;
    double step_peak = 0.0;
    double step_reference = 0.0;
    std::vector<double> impulse_response;
    std::vector<double> step_response;
    std::vector<StageBandMetrics> per_stage_band_metrics;
};

struct Stage2QualityAssessment {
    Stage2QualityMetrics metrics;
    bool pre_echo_within_limit = false;
    bool overshoot_within_limit = false;
};

Stage2BenchmarkConfig load_stage2_benchmark_config(const std::filesystem::path& path);

Stage2QualityAssessment evaluate_stage2_quality(const FirConfig& fir_config,
                                                const Stage2BenchmarkConfig& benchmark_config);

}  // namespace totton_audio_de_mirroring::dsp

#endif  // TOTTON_AUDIO_DE_MIRRORING_DSP_STAGE2_QUALITY_BENCHMARK_H
