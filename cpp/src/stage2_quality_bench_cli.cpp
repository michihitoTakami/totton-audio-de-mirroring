#include "totton_audio_de_mirroring/dsp/fir_config.h"
#include "totton_audio_de_mirroring/dsp/stage2_quality_benchmark.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

void print_usage() {
    std::cout << "Usage: tadm_stage2_quality_bench <fir_config.ini> <benchmark_config.ini>\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    try {
        const std::filesystem::path fir_config_path(argv[1]);
        const std::filesystem::path benchmark_config_path(argv[2]);

        const auto fir_config = totton_audio_de_mirroring::dsp::load_fir_config(fir_config_path);
        const auto benchmark_config =
            totton_audio_de_mirroring::dsp::load_stage2_benchmark_config(benchmark_config_path);
        const auto result =
            totton_audio_de_mirroring::dsp::evaluate_stage2_quality(fir_config, benchmark_config);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "output_sample_rate_hz=" << result.metrics.output_sample_rate_hz << "\n";
        std::cout << "pre_echo_ms=" << result.metrics.pre_echo_ms << "\n";
        std::cout << "step_overshoot_ratio=" << result.metrics.step_overshoot_ratio << "\n";
        std::cout << "step_peak=" << result.metrics.step_peak << "\n";
        std::cout << "step_reference=" << result.metrics.step_reference << "\n";
        std::cout << "pre_echo_within_limit=" << (result.pre_echo_within_limit ? "true" : "false")
                  << "\n";
        std::cout << "overshoot_within_limit=" << (result.overshoot_within_limit ? "true" : "false")
                  << "\n";

        for (const auto& stage : result.metrics.per_stage_band_metrics) {
            std::cout << "[" << stage.name << "]\n";
            std::cout << "  taps_count=" << stage.taps_count << "\n";
            std::cout << "  passband_hz=" << stage.passband_hz << "\n";
            std::cout << "  stopband_hz=" << stage.stopband_hz << "\n";
            std::cout << "  passband_ripple_db=" << stage.passband_ripple_db << "\n";
            std::cout << "  stopband_atten_db=" << stage.stopband_atten_db << "\n";
        }

        if (!result.pre_echo_within_limit || !result.overshoot_within_limit) {
            return 2;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Stage2 quality benchmark failed: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
