#include "totton_audio_de_mirroring/dsp/fir_config.h"
#include "totton_audio_de_mirroring/dsp/stage2_quality_benchmark.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::size_t check(bool condition, const char* message) {
    if (condition) {
        return 0;
    }
    std::fprintf(stderr, "TEST FAILED: %s\n", message);
    return 1;
}

std::filesystem::path make_temp_dir(const std::string& name) {
    const auto root = std::filesystem::temp_directory_path() / "tadm_stage2_quality_tests" / name;
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    return root;
}

void write_text_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("failed to write file: " + path.string());
    }
    output << content;
}

std::size_t test_benchmark_config_loads() {
    const auto dir = make_temp_dir("benchmark_config_loads");
    const auto config_path = dir / "benchmark.ini";
    write_text_file(config_path,
                    "[benchmark]\n"
                    "source_sample_rate_hz=96000\n"
                    "impulse_length=2048\n"
                    "step_length=4096\n"
                    "transition_index=128\n"
                    "settle_fraction=0.8\n"
                    "reference_quantile=0.9\n"
                    "pre_echo_threshold_ratio=0.0005\n"
                    "max_pre_echo_ms=0.1\n"
                    "max_step_overshoot_ratio=0.05\n");

    const auto loaded = totton_audio_de_mirroring::dsp::load_stage2_benchmark_config(config_path);

    std::size_t failures = 0;
    failures += check(std::fabs(loaded.source_sample_rate_hz - 96000.0) < 1e-9,
                      "source_sample_rate_hz mismatch");
    failures += check(loaded.impulse_length == 2048, "impulse_length mismatch");
    failures += check(loaded.step_length == 4096, "step_length mismatch");
    failures += check(loaded.transition_index == 128, "transition_index mismatch");
    failures += check(std::fabs(loaded.settle_fraction - 0.8) < 1e-9, "settle_fraction mismatch");
    return failures;
}

std::filesystem::path write_fir_config(const std::filesystem::path& dir,
                                       const std::vector<double>& taps, const std::string& name) {
    const auto taps_path = dir / (name + "_taps.txt");
    totton_audio_de_mirroring::dsp::write_taps_file(taps_path, taps);

    const auto config_path = dir / (name + "_fir.ini");
    write_text_file(config_path,
                    "[stage1]\n"
                    "sample_rate_hz=176400\n"
                    "passband_hz=20000\n"
                    "stopband_hz=25000\n"
                    "attenuation_db=20\n"
                    "passband_ripple_db=3\n"
                    "overshoot_max=0.5\n"
                    "pre_echo_ms_max=0.1\n"
                    "post_ringing_ms_max=10\n"
                    "num_taps=0\n"
                    "min_taps=3\n"
                    "max_taps=31\n"
                    "taps_path=" +
                        taps_path.string() + "\n");
    return config_path;
}

std::size_t test_identity_taps_meet_limits() {
    const auto dir = make_temp_dir("identity_taps_meet_limits");
    const auto fir_config_path = write_fir_config(dir, {1.0}, "identity");
    const auto fir_config = totton_audio_de_mirroring::dsp::load_fir_config(fir_config_path);

    totton_audio_de_mirroring::dsp::Stage2BenchmarkConfig benchmark;
    benchmark.source_sample_rate_hz = 88200.0;
    benchmark.impulse_length = 512;
    benchmark.step_length = 512;
    benchmark.transition_index = 64;
    benchmark.settle_fraction = 0.75;
    benchmark.reference_quantile = 0.95;
    benchmark.pre_echo_threshold_ratio = 1e-3;
    benchmark.max_pre_echo_ms = 0.1;
    benchmark.max_step_overshoot_ratio = 0.05;

    const auto result =
        totton_audio_de_mirroring::dsp::evaluate_stage2_quality(fir_config, benchmark);

    std::size_t failures = 0;
    failures += check(result.pre_echo_within_limit, "identity taps should pass pre-echo limit");
    failures += check(result.overshoot_within_limit, "identity taps should pass overshoot limit");
    failures +=
        check(result.metrics.per_stage_band_metrics.size() == 1, "expected one stage band metrics");
    failures += check(result.metrics.impulse_response.size() == benchmark.impulse_length * 2,
                      "impulse response length mismatch");
    failures += check(result.metrics.step_response.size() == benchmark.step_length * 2,
                      "step response length mismatch");
    return failures;
}

std::size_t test_overshoot_violation_is_detected() {
    const auto dir = make_temp_dir("overshoot_violation_detected");
    const auto fir_config_path = write_fir_config(dir, {1.0, 0.0, -0.95}, "ringing");
    const auto fir_config = totton_audio_de_mirroring::dsp::load_fir_config(fir_config_path);

    totton_audio_de_mirroring::dsp::Stage2BenchmarkConfig benchmark;
    benchmark.source_sample_rate_hz = 88200.0;
    benchmark.impulse_length = 512;
    benchmark.step_length = 512;
    benchmark.transition_index = 64;
    benchmark.settle_fraction = 0.75;
    benchmark.reference_quantile = 0.95;
    benchmark.pre_echo_threshold_ratio = 1e-3;
    benchmark.max_pre_echo_ms = 0.1;
    benchmark.max_step_overshoot_ratio = 0.05;

    const auto result =
        totton_audio_de_mirroring::dsp::evaluate_stage2_quality(fir_config, benchmark);

    std::size_t failures = 0;
    failures +=
        check(!result.overshoot_within_limit, "ringing taps should violate overshoot limit");
    failures += check(result.metrics.step_overshoot_ratio > 0.05,
                      "overshoot ratio should exceed threshold");
    return failures;
}

}  // namespace

int main() {
    std::size_t failures = 0;
    failures += test_benchmark_config_loads();
    failures += test_identity_taps_meet_limits();
    failures += test_overshoot_violation_is_detected();
    return failures == 0 ? 0 : 1;
}
