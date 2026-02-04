#include "totton_audio_de_mirroring/dsp/multistage_upsampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace totton_audio_de_mirroring::dsp {

FirUpsampler2x::FirUpsampler2x(std::vector<double> taps)
    : history_pos_(0) {
    validate_taps(taps);

    even_taps_.reserve((taps.size() + 1) / 2);
    odd_taps_.reserve(taps.size() / 2);

    for (std::size_t i = 0; i < taps.size(); ++i) {
        if (i % 2 == 0) {
            even_taps_.push_back(taps[i]);
        } else {
            odd_taps_.push_back(taps[i]);
        }
    }

    const std::size_t history_len = std::max(even_taps_.size(), odd_taps_.size());
    history_.assign(history_len, 0.0);
    history_pos_ = history_len == 0 ? 0 : history_len - 1;
}

std::vector<double> FirUpsampler2x::process_block(const std::vector<double>& input) {
    return process_block(input.data(), input.size());
}

std::vector<double> FirUpsampler2x::process_block(
    const double* input,
    std::size_t length
) {
    if (length == 0) {
        return {};
    }
    if (input == nullptr) {
        throw std::invalid_argument("input pointer must not be null");
    }

    std::vector<double> output;
    output.reserve(length * 2);

    for (std::size_t n = 0; n < length; ++n) {
        if (!history_.empty()) {
            history_pos_ = (history_pos_ + 1) % history_.size();
            history_[history_pos_] = input[n];
        }

        const double even = dot(even_taps_);
        const double odd = dot(odd_taps_);
        output.push_back(even);
        output.push_back(odd);
    }

    return output;
}

void FirUpsampler2x::reset() {
    std::fill(history_.begin(), history_.end(), 0.0);
    history_pos_ = history_.empty() ? 0 : history_.size() - 1;
}

std::size_t FirUpsampler2x::taps_size() const {
    return even_taps_.size() + odd_taps_.size();
}

std::size_t FirUpsampler2x::history_size() const {
    return history_.size();
}

double FirUpsampler2x::dot(const std::vector<double>& taps) const {
    if (taps.empty() || history_.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    std::size_t idx = history_pos_;
    for (std::size_t i = 0; i < taps.size(); ++i) {
        sum += taps[i] * history_[idx];
        idx = idx == 0 ? history_.size() - 1 : idx - 1;
    }
    return sum;
}

void FirUpsampler2x::validate_taps(const std::vector<double>& taps) const {
    if (taps.empty()) {
        throw std::invalid_argument("taps must not be empty");
    }
    for (double value : taps) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("taps must be finite values");
        }
    }
}

MultiStageUpsampler::MultiStageUpsampler(
    std::vector<std::vector<double>> stage_taps
) {
    validate_stage_taps(stage_taps);
    stages_.reserve(stage_taps.size());
    for (auto& taps : stage_taps) {
        stages_.emplace_back(std::move(taps));
    }
}

std::vector<double> MultiStageUpsampler::process_block(
    const std::vector<double>& input
) {
    return process_block(input.data(), input.size());
}

std::vector<double> MultiStageUpsampler::process_block(
    const double* input,
    std::size_t length
) {
    if (length == 0) {
        return {};
    }
    if (input == nullptr) {
        throw std::invalid_argument("input pointer must not be null");
    }

    std::vector<double> current(input, input + length);
    for (auto& stage : stages_) {
        current = stage.process_block(current);
    }
    return current;
}

void MultiStageUpsampler::reset() {
    for (auto& stage : stages_) {
        stage.reset();
    }
}

std::size_t MultiStageUpsampler::stage_count() const {
    return stages_.size();
}

void MultiStageUpsampler::validate_stage_taps(
    const std::vector<std::vector<double>>& stage_taps
) const {
    if (stage_taps.empty()) {
        throw std::invalid_argument("stage_taps must not be empty");
    }
    for (const auto& taps : stage_taps) {
        if (taps.empty()) {
            throw std::invalid_argument("stage taps must not be empty");
        }
        for (double value : taps) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("stage taps must be finite values");
            }
        }
    }
}

}  // namespace totton_audio_de_mirroring::dsp
