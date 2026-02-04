#include "totton_audio_de_mirroring/dsp/multistage_upsampler.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

template <typename T>
bool nearly_equal(T a, T b, T tol) {
    return std::fabs(a - b) <= tol;
}

void test_upsampler_doubles_length() {
    const std::vector<double> taps = {1.0};
    totton_audio_de_mirroring::dsp::FirUpsampler2x upsampler(taps);

    const std::vector<double> input = {0.1, -0.2, 0.3, -0.4};
    const auto output = upsampler.process_block(input);

    assert(output.size() == input.size() * 2);
}

void test_upsampler_zero_stuffing() {
    const std::vector<double> taps = {1.0};
    totton_audio_de_mirroring::dsp::FirUpsampler2x upsampler(taps);

    const std::vector<double> input = {1.0, 2.0, 3.0};
    const auto output = upsampler.process_block(input);

    const std::vector<double> expected = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0};
    assert(output == expected);
}

void test_streaming_consistency() {
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

    assert(output_full.size() == output_split.size());
    for (std::size_t i = 0; i < output_full.size(); ++i) {
        assert(nearly_equal(output_full[i], output_split[i], 1e-12));
    }
}

void test_multistage_length() {
    const std::vector<std::vector<double>> stage_taps = {
        {1.0},
        {1.0},
    };
    totton_audio_de_mirroring::dsp::MultiStageUpsampler upsampler(stage_taps);

    const std::vector<double> input = {0.5, -0.5, 0.25};
    const auto output = upsampler.process_block(input);

    assert(output.size() == input.size() * 4);
}

void test_multistage_positions() {
    const std::vector<std::vector<double>> stage_taps = {
        {1.0},
        {1.0},
    };
    totton_audio_de_mirroring::dsp::MultiStageUpsampler upsampler(stage_taps);

    const std::vector<double> input = {1.0, 2.0, 3.0};
    const auto output = upsampler.process_block(input);

    for (std::size_t i = 0; i < output.size(); ++i) {
        if (i % 4 == 0) {
            const std::size_t idx = i / 4;
            assert(nearly_equal(output[i], input[idx], 1e-12));
        } else {
            assert(nearly_equal(output[i], 0.0, 1e-12));
        }
    }
}

}  // namespace

int main() {
    test_upsampler_doubles_length();
    test_upsampler_zero_stuffing();
    test_streaming_consistency();
    test_multistage_length();
    test_multistage_positions();
    return 0;
}
