#include "totton_audio_de_mirroring/dsp/multistage_upsampler_c_api.h"

#include "totton_audio_de_mirroring/dsp/fir_config.h"
#include "totton_audio_de_mirroring/dsp/multistage_upsampler.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct UpsamplerHandle {
    explicit UpsamplerHandle(std::vector<std::vector<double>> stage_taps)
        : stage_count(stage_taps.size()), upsampler(std::move(stage_taps)) {}

    std::size_t stage_count;
    totton_audio_de_mirroring::dsp::MultiStageUpsampler upsampler;
};

void write_error(const std::string& message, char* buffer, std::size_t buffer_length) {
    if (buffer == nullptr || buffer_length == 0) {
        return;
    }
    const std::size_t copy_length = std::min(buffer_length - 1, message.size());
    std::memcpy(buffer, message.data(), copy_length);
    buffer[copy_length] = '\0';
}

std::size_t checked_output_length(std::size_t input_length, std::size_t num_stages) {
    if (num_stages == 0) {
        throw std::invalid_argument("num_stages must be positive");
    }

    std::size_t output_length = input_length;
    for (std::size_t i = 0; i < num_stages; ++i) {
        if (output_length > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::overflow_error("output length overflow");
        }
        output_length *= 2;
    }
    return output_length;
}

std::vector<std::vector<double>> load_stage_taps(const std::filesystem::path& config_dir,
                                                 std::size_t num_stages) {
    if (num_stages == 0) {
        throw std::invalid_argument("num_stages must be positive");
    }
    if (!std::filesystem::exists(config_dir)) {
        throw std::runtime_error("config directory not found: " + config_dir.string());
    }

    std::vector<std::vector<double>> stage_taps;
    stage_taps.reserve(num_stages);
    for (std::size_t stage_index = 1; stage_index <= num_stages; ++stage_index) {
        const auto taps_path = config_dir / ("stage" + std::to_string(stage_index) + "_taps.txt");
        stage_taps.push_back(totton_audio_de_mirroring::dsp::read_taps_file(taps_path));
    }
    return stage_taps;
}

}  // namespace

extern "C" {

void* tadm_create_multistage_upsampler_from_dir(const char* config_dir, size_t num_stages,
                                                char* error_buffer, size_t error_buffer_length) {
    try {
        if (config_dir == nullptr) {
            throw std::invalid_argument("config_dir must not be null");
        }
        const auto stage_taps = load_stage_taps(std::filesystem::path(config_dir), num_stages);
        auto handle = std::make_unique<UpsamplerHandle>(stage_taps);
        return handle.release();
    } catch (const std::exception& ex) {
        write_error(ex.what(), error_buffer, error_buffer_length);
        return nullptr;
    }
}

void tadm_destroy_multistage_upsampler(void* handle) {
    auto* typed = static_cast<UpsamplerHandle*>(handle);
    delete typed;
}

size_t tadm_multistage_output_length(size_t input_length, size_t num_stages) {
    try {
        return checked_output_length(input_length, num_stages);
    } catch (...) {
        return 0;
    }
}

size_t tadm_multistage_process_block(void* handle, const double* input, size_t input_length,
                                     double* output, size_t output_length, char* error_buffer,
                                     size_t error_buffer_length) {
    try {
        if (handle == nullptr) {
            throw std::invalid_argument("handle must not be null");
        }
        if (input_length == 0) {
            return 0;
        }
        if (input == nullptr) {
            throw std::invalid_argument("input must not be null");
        }
        if (output == nullptr) {
            throw std::invalid_argument("output must not be null");
        }

        auto* typed = static_cast<UpsamplerHandle*>(handle);
        const auto expected_length = checked_output_length(input_length, typed->stage_count);
        if (output_length < expected_length) {
            throw std::invalid_argument("output buffer is too small");
        }

        const auto result = typed->upsampler.process_block(input, input_length);
        if (result.size() != expected_length) {
            throw std::runtime_error("unexpected output length from upsampler");
        }
        std::copy(result.begin(), result.end(), output);
        return result.size();
    } catch (const std::exception& ex) {
        write_error(ex.what(), error_buffer, error_buffer_length);
        return 0;
    }
}

}  // extern "C"
