#include "totton_audio_de_mirroring/dsp/fir_config.h"
#include "totton_audio_de_mirroring/dsp/fir_design.h"
#include "totton_audio_de_mirroring/dsp/hirate_fir_design.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <vector>

namespace {

using totton_audio_de_mirroring::dsp::hirate_fir::StageDesign;

constexpr double kPassbandHz = 20000.0;
constexpr double kPassbandRippleMaxDb = 0.001;
constexpr double kImageSuppressionMinDb = 140.0;
constexpr double kCascadeFlatnessMaxDb = 0.01;

std::size_t check(bool condition, const char* message) {
    if (condition) {
        return 0;
    }
    std::fprintf(stderr, "TEST FAILED: %s\n", message);
    return 1;
}

std::filesystem::path repo_root() {
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
}

double magnitude_db(const std::vector<float>& kernel, double frequency_hz, double sample_rate_hz) {
    std::complex<double> response(0.0, 0.0);
    const double omega =
        2.0 * totton_audio_de_mirroring::dsp::hirate_fir::kPi * frequency_hz / sample_rate_hz;
    for (std::size_t index = 0; index < kernel.size(); ++index) {
        const double phase = omega * static_cast<double>(index);
        response += static_cast<double>(kernel[index]) *
                    std::complex<double>(std::cos(phase), -std::sin(phase));
    }
    const double dc_gain = std::accumulate(kernel.begin(), kernel.end(), 0.0);
    return 20.0 * std::log10(std::max(std::abs(response) / dc_gain, 1e-30));
}

std::vector<float> make_effective_kernel(const StageDesign& stage) {
    return totton_audio_de_mirroring::dsp::hirate_fir::design_interpolation_kernel(
        stage.taps, stage.cutoff_hz, stage.output_rate_hz);
}

std::size_t test_config_matches_canonical_plan() {
    const auto path = repo_root() / "cpp/configs/hirate_fir_linear_phase.ini";
    const auto config = totton_audio_de_mirroring::dsp::load_fir_config(path);
    const auto plan = totton_audio_de_mirroring::dsp::hirate_fir::design_8x_plan(88200.0);
    std::size_t failures = check(config.stages.size() == plan.size(), "expected three stages");
    for (std::size_t index = 0; index < plan.size(); ++index) {
        const auto& spec = config.stages[index].design;
        failures += check(spec.design_kind == "hirate_linear", "design kind mismatch");
        failures += check(spec.num_taps == plan[index].taps, "tap ladder mismatch");
        failures += check(spec.cutoff_hz == plan[index].cutoff_hz, "cutoff mismatch");
        failures +=
            check(spec.sample_rate_hz == plan[index].output_rate_hz, "output rate mismatch");
    }
    return failures;
}

std::size_t test_checked_in_taps_match_designer() {
    const auto plan = totton_audio_de_mirroring::dsp::hirate_fir::design_8x_plan(88200.0);
    std::size_t failures = 0;
    for (std::size_t index = 0; index < plan.size(); ++index) {
        const auto expected =
            totton_audio_de_mirroring::dsp::hirate_fir::design_runtime_storage_taps(
                plan[index].taps, plan[index].cutoff_hz, plan[index].output_rate_hz);
        const auto actual = totton_audio_de_mirroring::dsp::read_taps_file(
            repo_root() / "cpp/configs" / ("stage" + std::to_string(index + 1) + "_taps.txt"));
        failures += check(actual == expected, "checked-in taps differ from designer");
    }
    return failures;
}

std::size_t test_stage_contract(double input_rate_hz) {
    const auto plan = totton_audio_de_mirroring::dsp::hirate_fir::design_8x_plan(input_rate_hz);
    std::size_t failures = 0;
    for (std::size_t stage_index = 0; stage_index < plan.size(); ++stage_index) {
        const auto kernel = make_effective_kernel(plan[stage_index]);
        failures += check(!kernel.empty(), "kernel must not be empty");
        failures += check(kernel.size() % 2 == 1, "kernel must have odd length");
        failures += check(kernel == std::vector<float>(kernel.rbegin(), kernel.rend()),
                          "kernel must be exactly symmetric");
        const double dc_gain = std::accumulate(kernel.begin(), kernel.end(), 0.0);
        failures += check(std::abs(dc_gain - 2.0) < 1e-5, "effective DC gain mismatch");

        for (double frequency = 100.0; frequency <= kPassbandHz; frequency += 100.0) {
            failures +=
                check(std::abs(magnitude_db(kernel, frequency, plan[stage_index].output_rate_hz)) <
                          kPassbandRippleMaxDb,
                      "audible passband deviation exceeds contract");
        }
        const double stage_input_rate = input_rate_hz * static_cast<double>(1U << stage_index);
        for (double frequency = stage_input_rate - kPassbandHz; frequency <= stage_input_rate;
             frequency += 20.0) {
            failures += check(magnitude_db(kernel, frequency, plan[stage_index].output_rate_hz) <
                                  -kImageSuppressionMinDb,
                              "image suppression does not meet contract");
        }
    }
    return failures;
}

std::size_t test_cascade_contract(double input_rate_hz) {
    const auto plan = totton_audio_de_mirroring::dsp::hirate_fir::design_8x_plan(input_rate_hz);
    std::size_t failures = 0;
    for (double frequency : {1000.0, 10000.0, 20000.0}) {
        double total_db = 0.0;
        for (const auto& stage : plan) {
            total_db += magnitude_db(make_effective_kernel(stage), frequency, stage.output_rate_hz);
        }
        failures += check(std::abs(total_db) < kCascadeFlatnessMaxDb,
                          "cascade passband deviation exceeds contract");
    }
    const std::size_t final_delay =
        ((plan[0].taps - 1) / 2) * 4 + ((plan[1].taps - 1) / 2) * 2 + (plan[2].taps - 1) / 2;
    failures += check(final_delay == 589, "final-rate group delay mismatch");
    return failures;
}

std::size_t test_invalid_design_requests() {
    using totton_audio_de_mirroring::dsp::hirate_fir::design_interpolation_kernel;
    std::size_t failures = 0;
    failures += check(design_interpolation_kernel(64, 44100.0, 176400.0).empty(),
                      "even tap count must be rejected");
    failures += check(design_interpolation_kernel(1, 44100.0, 176400.0).empty(),
                      "short kernel must be rejected");
    failures += check(design_interpolation_kernel(63, 0.0, 176400.0).empty(),
                      "zero cutoff must be rejected");
    failures += check(design_interpolation_kernel(63, 44100.0, 0.0).empty(),
                      "zero output rate must be rejected");
    return failures;
}

std::size_t test_legacy_minimum_phase_designer_remains_available() {
    totton_audio_de_mirroring::dsp::FirDesignSpec spec;
    spec.sample_rate_hz = 176400.0;
    spec.passband_hz = 20000.0;
    spec.stopband_hz = 68200.0;
    spec.attenuation_db = 20.0;
    spec.passband_ripple_db = 1.0;
    spec.num_taps = 31;
    spec.min_taps = 31;
    spec.max_taps = 31;

    totton_audio_de_mirroring::dsp::FirDesignMetrics metrics;
    const auto taps = totton_audio_de_mirroring::dsp::design_minimum_phase_lowpass(spec, &metrics);
    std::size_t failures =
        check(taps.size() == spec.num_taps, "legacy minimum-phase designer must remain available");
    failures += check(std::isfinite(metrics.passband_ripple_db),
                      "legacy designer must return finite metrics");
    return failures;
}

}  // namespace

int main() {
    std::size_t failures = 0;
    failures += test_config_matches_canonical_plan();
    failures += test_checked_in_taps_match_designer();
    failures += test_stage_contract(88200.0);
    failures += test_stage_contract(96000.0);
    failures += test_cascade_contract(88200.0);
    failures += test_cascade_contract(96000.0);
    failures += test_invalid_design_requests();
    failures += test_legacy_minimum_phase_designer_remains_available();
    return failures == 0 ? 0 : 1;
}
