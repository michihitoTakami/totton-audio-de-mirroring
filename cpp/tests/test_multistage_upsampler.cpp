#include "totton_audio_de_mirroring/dsp/multistage_upsampler.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

template <typename T>
bool nearly_equal(T a, T b, T tol) {
    return std::fabs(a - b) <= tol;
}

std::size_t check(bool condition, const char* message) {
    if (condition) {
        return 0;
    }
    std::fprintf(stderr, "TEST FAILED: %s\n", message);
    return 1;
}

std::size_t test_upsampler_doubles_length() {
    const std::vector<double> taps = {1.0};
    totton_audio_de_mirroring::dsp::FirUpsampler2x upsampler(taps);

    const std::vector<double> input = {0.1, -0.2, 0.3, -0.4};
    const auto output = upsampler.process_block(input);

    std::size_t failures = 0;
    failures += check(output.size() == input.size() * 2, "2x length mismatch");
    failures += check(output.size() == upsampler.output_length_for_input(input.size()),
                      "output_length_for_input mismatch");
    return failures;
}

std::size_t test_upsampler_zero_stuffing() {
    const std::vector<double> taps = {1.0};
    totton_audio_de_mirroring::dsp::FirUpsampler2x upsampler(taps);

    const std::vector<double> input = {1.0, 2.0, 3.0};
    const auto output = upsampler.process_block(input);

    const std::vector<double> expected = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0};
    return check(output == expected, "zero-stuffing output mismatch");
}

std::size_t test_input_is_not_modified() {
    const std::vector<double> taps = {0.5, 0.5};
    totton_audio_de_mirroring::dsp::FirUpsampler2x upsampler(taps);

    std::vector<double> input = {0.25, -0.5, 0.75, 1.0};
    const std::vector<double> original = input;
    const auto output = upsampler.process_block(input);
    (void)output;

    return check(input == original, "process_block modified input buffer");
}

std::size_t test_streaming_consistency() {
    const std::vector<double> taps = {0.25, 0.5, 0.25};
    totton_audio_de_mirroring::dsp::FirUpsampler2x upsampler(taps);

    std::vector<double> input;
    input.reserve(16);
    for (int i = 0; i < 16; ++i) {
        input.push_back(static_cast<double>(i) / 10.0);
    }

    const auto output_full = upsampler.process_block(input);

    upsampler.reset();
    const std::vector<double> first_half(input.begin(), input.begin() + 8);
    const std::vector<double> second_half(input.begin() + 8, input.end());

    const auto out_first = upsampler.process_block(first_half);
    const auto out_second = upsampler.process_block(second_half);

    std::vector<double> output_split;
    output_split.reserve(out_first.size() + out_second.size());
    output_split.insert(output_split.end(), out_first.begin(), out_first.end());
    output_split.insert(output_split.end(), out_second.begin(), out_second.end());

    std::size_t failures = 0;
    failures +=
        check(output_full.size() == output_split.size(), "streaming output length mismatch");
    for (std::size_t i = 0; i < output_full.size(); ++i) {
        failures += check(nearly_equal(output_full[i], output_split[i], 1e-12),
                          "streaming output sample mismatch");
    }
    return failures;
}

std::size_t test_dc_gain() {
    const std::vector<double> taps = {1.0, 1.0};
    totton_audio_de_mirroring::dsp::FirUpsampler2x upsampler(taps);
    const std::vector<double> input(32, 1.0);
    const auto output = upsampler.process_block(input);

    std::size_t failures = 0;
    failures += check(nearly_equal(upsampler.steady_state_dc_gain_even(), 1.0, 1e-12),
                      "even DC gain mismatch");
    failures += check(nearly_equal(upsampler.steady_state_dc_gain_odd(), 1.0, 1e-12),
                      "odd DC gain mismatch");
    for (std::size_t i = 16; i < output.size(); ++i) {
        failures += check(nearly_equal(output[i], 1.0, 1e-12), "steady-state DC output mismatch");
    }
    return failures;
}

std::size_t test_impulse_response() {
    const std::vector<double> taps = {0.2, 0.4, 0.6, 0.8, 1.0};
    totton_audio_de_mirroring::dsp::FirUpsampler2x upsampler(taps);

    std::vector<double> input(taps.size(), 0.0);
    input[0] = 1.0;
    const auto output = upsampler.process_block(input);

    std::size_t failures = 0;
    for (std::size_t i = 0; i < taps.size(); ++i) {
        failures +=
            check(nearly_equal(output[i], taps[i], 1e-12), "impulse response coefficient mismatch");
    }
    for (std::size_t i = taps.size(); i < output.size(); ++i) {
        failures += check(nearly_equal(output[i], 0.0, 1e-12), "impulse response tail mismatch");
    }
    return failures;
}

std::size_t test_reproducibility_after_reset() {
    const std::vector<double> taps = {0.125, 0.25, 0.5, 0.25, 0.125};
    totton_audio_de_mirroring::dsp::FirUpsampler2x upsampler(taps);
    const std::vector<double> input = {0.2, -0.1, 0.3, -0.2, 0.4, -0.3};

    const auto first = upsampler.process_block(input);
    upsampler.reset();
    const auto second = upsampler.process_block(input);

    std::size_t failures = 0;
    failures += check(first.size() == second.size(), "reproducibility length mismatch");
    for (std::size_t i = 0; i < first.size(); ++i) {
        failures +=
            check(nearly_equal(first[i], second[i], 1e-12), "reproducibility sample mismatch");
    }
    return failures;
}

std::size_t test_config_file_injection() {
    const auto unique =
        std::to_string(std::filesystem::file_time_type::clock::now().time_since_epoch().count());
    const auto taps_path =
        std::filesystem::temp_directory_path() / ("tadm_issue43_taps_" + unique + ".txt");
    {
        std::ofstream out(taps_path);
        if (!out.is_open()) {
            return check(false, "failed to create temp taps file");
        }
        out << "1.0\n";
    }

    const std::vector<double> input = {1.0, 2.0, 3.0};
    auto upsampler = totton_audio_de_mirroring::dsp::FirUpsampler2x::from_taps_file(taps_path);
    const auto output = upsampler.process_block(input);
    std::filesystem::remove(taps_path);

    const std::vector<double> expected = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0};
    return check(output == expected, "config-file injected taps mismatch");
}

std::size_t test_multistage_length() {
    const std::vector<std::vector<double>> stage_taps = {
        {1.0},
        {1.0},
    };
    totton_audio_de_mirroring::dsp::MultiStageUpsampler upsampler(stage_taps);

    const std::vector<double> input = {0.5, -0.5, 0.25};
    const auto output = upsampler.process_block(input);

    return check(output.size() == input.size() * 4, "multistage length mismatch");
}

std::size_t test_multistage_positions() {
    const std::vector<std::vector<double>> stage_taps = {
        {1.0},
        {1.0},
    };
    totton_audio_de_mirroring::dsp::MultiStageUpsampler upsampler(stage_taps);

    const std::vector<double> input = {1.0, 2.0, 3.0};
    const auto output = upsampler.process_block(input);

    std::size_t failures = 0;
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (i % 4 == 0) {
            const std::size_t idx = i / 4;
            failures +=
                check(nearly_equal(output[i], input[idx], 1e-12), "multistage alignment mismatch");
        } else {
            failures +=
                check(nearly_equal(output[i], 0.0, 1e-12), "multistage zero-stuffing mismatch");
        }
    }
    return failures;
}

}  // namespace

int main() {
    std::size_t failures = 0;
    failures += test_upsampler_doubles_length();
    failures += test_upsampler_zero_stuffing();
    failures += test_input_is_not_modified();
    failures += test_streaming_consistency();
    failures += test_dc_gain();
    failures += test_impulse_response();
    failures += test_reproducibility_after_reset();
    failures += test_config_file_injection();
    failures += test_multistage_length();
    failures += test_multistage_positions();
    return failures == 0 ? 0 : 1;
}
