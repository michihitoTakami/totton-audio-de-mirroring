#include "totton_audio_de_mirroring/dsp/fir_config.h"
#include "totton_audio_de_mirroring/dsp/fir_design.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace {

std::size_t check(bool condition, const char* message) {
    if (condition) {
        return 0;
    }
    std::fprintf(stderr, "TEST FAILED: %s\n", message);
    return 1;
}

std::filesystem::path config_path() {
    const auto test_path = std::filesystem::path(__FILE__);
    return test_path.parent_path().parent_path() / "configs" / "hie_fir_min_phase.ini";
}

std::size_t test_config_loads() {
    const auto config = totton_audio_de_mirroring::dsp::load_fir_config(config_path());
    std::size_t failures = 0;
    failures += check(config.stages.size() == 3, "expected 3 stages");
    failures += check(config.stages[0].name == "stage1", "stage1 name mismatch");
    failures += check(std::fabs(config.stages[0].design.sample_rate_hz - 176400.0) < 1e-6,
                      "stage1 sample rate mismatch");
    failures += check(!config.stages[0].taps_path.empty(), "stage1 taps path empty");
    return failures;
}

std::size_t test_minimum_phase_design_metrics() {
    const auto config = totton_audio_de_mirroring::dsp::load_fir_config(config_path());
    const auto& stage = config.stages[0];

    totton_audio_de_mirroring::dsp::FirDesignMetrics metrics;
    const auto taps =
        totton_audio_de_mirroring::dsp::design_minimum_phase_lowpass(stage.design, &metrics);

    std::size_t failures = 0;
    failures += check(!taps.empty(), "taps must not be empty");
    failures += check(taps.size() % 2 == 1, "taps must be odd length");
    failures += check(metrics.stopband_atten_db >= stage.design.attenuation_db - 1e-6,
                      "stopband attenuation too low");
    failures += check(metrics.passband_ripple_db <= stage.design.passband_ripple_db + 1e-6,
                      "passband ripple too high");
    failures +=
        check(metrics.overshoot_ratio <= stage.design.overshoot_max + 1e-6, "overshoot too high");
    failures +=
        check(metrics.pre_echo_ms <= stage.design.pre_echo_ms_max + 1e-6, "pre-echo too high");
    failures += check(metrics.post_ringing_ms <= stage.design.post_ringing_ms_max + 1e-6,
                      "post-ringing too high");
    return failures;
}

}  // namespace

int main() {
    std::size_t failures = 0;
    failures += test_config_loads();
    failures += test_minimum_phase_design_metrics();
    return failures == 0 ? 0 : 1;
}
