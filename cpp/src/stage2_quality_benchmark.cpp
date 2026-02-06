#include "totton_audio_de_mirroring/dsp/stage2_quality_benchmark.h"

#include "totton_audio_de_mirroring/dsp/fir_design.h"
#include "totton_audio_de_mirroring/dsp/multistage_upsampler.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace totton_audio_de_mirroring::dsp {

namespace {

std::string trim(const std::string& value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

void apply_benchmark_kv(Stage2BenchmarkConfig& config, const std::string& key,
                        const std::string& value) {
    if (key == "source_sample_rate_hz") {
        config.source_sample_rate_hz = std::stod(value);
    } else if (key == "impulse_length") {
        config.impulse_length = static_cast<std::size_t>(std::stoul(value));
    } else if (key == "step_length") {
        config.step_length = static_cast<std::size_t>(std::stoul(value));
    } else if (key == "transition_index") {
        config.transition_index = static_cast<std::size_t>(std::stoul(value));
    } else if (key == "settle_fraction") {
        config.settle_fraction = std::stod(value);
    } else if (key == "reference_quantile") {
        config.reference_quantile = std::stod(value);
    } else if (key == "pre_echo_threshold_ratio") {
        config.pre_echo_threshold_ratio = std::stod(value);
    } else if (key == "max_pre_echo_ms") {
        config.max_pre_echo_ms = std::stod(value);
    } else if (key == "max_step_overshoot_ratio") {
        config.max_step_overshoot_ratio = std::stod(value);
    } else {
        throw std::invalid_argument("unknown benchmark key: " + key);
    }
}

void validate_benchmark_config(const Stage2BenchmarkConfig& config) {
    if (config.source_sample_rate_hz <= 0.0) {
        throw std::invalid_argument("source_sample_rate_hz must be positive");
    }
    if (config.impulse_length < 8) {
        throw std::invalid_argument("impulse_length must be >= 8");
    }
    if (config.step_length < 8) {
        throw std::invalid_argument("step_length must be >= 8");
    }
    if (config.transition_index == 0) {
        throw std::invalid_argument("transition_index must be > 0");
    }
    if (config.transition_index >= config.impulse_length) {
        throw std::invalid_argument("transition_index must be less than impulse_length");
    }
    if (config.transition_index >= config.step_length) {
        throw std::invalid_argument("transition_index must be less than step_length");
    }
    if (config.settle_fraction <= 0.0 || config.settle_fraction >= 1.0) {
        throw std::invalid_argument("settle_fraction must be in (0, 1)");
    }
    if (config.reference_quantile < 0.5 || config.reference_quantile >= 1.0) {
        throw std::invalid_argument("reference_quantile must be in [0.5, 1)");
    }
    if (config.pre_echo_threshold_ratio <= 0.0) {
        throw std::invalid_argument("pre_echo_threshold_ratio must be positive");
    }
    if (config.max_pre_echo_ms <= 0.0) {
        throw std::invalid_argument("max_pre_echo_ms must be positive");
    }
    if (config.max_step_overshoot_ratio < 0.0) {
        throw std::invalid_argument("max_step_overshoot_ratio must be non-negative");
    }
}

std::size_t upsampling_factor(std::size_t num_stages) {
    std::size_t factor = 1;
    for (std::size_t i = 0; i < num_stages; ++i) {
        factor *= 2;
    }
    return factor;
}

std::vector<std::vector<double>> load_stage_taps(const FirConfig& fir_config) {
    std::vector<std::vector<double>> taps;
    taps.reserve(fir_config.stages.size());
    for (const auto& stage : fir_config.stages) {
        taps.push_back(read_taps_file(stage.taps_path));
    }
    return taps;
}

std::vector<double> make_impulse_input(const Stage2BenchmarkConfig& config) {
    std::vector<double> input(config.impulse_length, 0.0);
    input[config.transition_index] = 1.0;
    return input;
}

std::vector<double> make_step_input(const Stage2BenchmarkConfig& config) {
    std::vector<double> input(config.step_length, 0.0);
    for (std::size_t i = config.transition_index; i < input.size(); ++i) {
        input[i] = 1.0;
    }
    return input;
}

double abs_max(const std::vector<double>& values) {
    double max_value = 0.0;
    for (double value : values) {
        max_value = std::max(max_value, std::fabs(value));
    }
    return max_value;
}

double quantile(std::vector<double> values, double q) {
    if (values.empty()) {
        throw std::invalid_argument("quantile input must not be empty");
    }
    std::sort(values.begin(), values.end());
    const double index = q * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(index));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
    if (lower == upper) {
        return values[lower];
    }
    const double alpha = index - static_cast<double>(lower);
    return values[lower] * (1.0 - alpha) + values[upper] * alpha;
}

double compute_step_reference(const std::vector<double>& step_response,
                              const Stage2BenchmarkConfig& config) {
    const std::size_t settle_start =
        static_cast<std::size_t>(std::floor(config.settle_fraction * step_response.size()));
    if (settle_start >= step_response.size()) {
        throw std::runtime_error("settle_fraction produced empty settled region");
    }
    std::vector<double> settled(step_response.begin() + static_cast<std::ptrdiff_t>(settle_start),
                                step_response.end());
    return quantile(std::move(settled), config.reference_quantile);
}

double compute_step_overshoot_ratio(const std::vector<double>& step_response,
                                    const Stage2BenchmarkConfig& config, double* peak_out,
                                    double* reference_out) {
    const double reference = compute_step_reference(step_response, config);
    const double peak = *std::max_element(step_response.begin(), step_response.end());
    if (peak_out != nullptr) {
        *peak_out = peak;
    }
    if (reference_out != nullptr) {
        *reference_out = reference;
    }
    if (std::fabs(reference) < 1e-12) {
        throw std::runtime_error("step reference is too close to zero");
    }
    return std::max(0.0, (peak - reference) / std::fabs(reference));
}

double compute_pre_echo_ms(const std::vector<double>& impulse_response,
                           const Stage2BenchmarkConfig& config,
                           std::size_t output_upsampling_factor, double output_sample_rate_hz) {
    if (impulse_response.empty()) {
        throw std::invalid_argument("impulse_response must not be empty");
    }
    if (output_sample_rate_hz <= 0.0) {
        throw std::invalid_argument("output_sample_rate_hz must be positive");
    }

    const std::size_t edge_index = config.transition_index * output_upsampling_factor;
    if (edge_index >= impulse_response.size()) {
        throw std::runtime_error("transition index is outside impulse response");
    }

    const double peak = abs_max(impulse_response);
    const double threshold = peak * config.pre_echo_threshold_ratio;
    if (threshold <= 0.0) {
        return 0.0;
    }

    std::size_t last_nonzero_before_edge = 0;
    bool found = false;
    for (std::size_t i = 0; i < edge_index; ++i) {
        if (std::fabs(impulse_response[i]) >= threshold) {
            last_nonzero_before_edge = i;
            found = true;
        }
    }

    if (!found) {
        return 0.0;
    }

    const std::size_t delta_samples = edge_index - last_nonzero_before_edge;
    return static_cast<double>(delta_samples) / output_sample_rate_hz * 1000.0;
}

std::vector<StageBandMetrics> compute_per_stage_band_metrics(
    const FirConfig& fir_config, const std::vector<std::vector<double>>& stage_taps) {
    if (fir_config.stages.size() != stage_taps.size()) {
        throw std::invalid_argument("fir_config and stage_taps size mismatch");
    }

    std::vector<StageBandMetrics> metrics;
    metrics.reserve(fir_config.stages.size());
    for (std::size_t i = 0; i < fir_config.stages.size(); ++i) {
        const auto& stage = fir_config.stages[i];
        const auto analyzed = analyze_fir(stage_taps[i], stage.design.sample_rate_hz,
                                          stage.design.passband_hz, stage.design.stopband_hz);
        StageBandMetrics band;
        band.name = stage.name;
        band.taps_count = stage_taps[i].size();
        band.passband_hz = stage.design.passband_hz;
        band.stopband_hz = stage.design.stopband_hz;
        band.passband_ripple_db = analyzed.passband_ripple_db;
        band.stopband_atten_db = analyzed.stopband_atten_db;
        metrics.push_back(band);
    }
    return metrics;
}

}  // namespace

Stage2BenchmarkConfig load_stage2_benchmark_config(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("benchmark config file not found: " + path.string());
    }

    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open benchmark config: " + path.string());
    }

    Stage2BenchmarkConfig config;
    bool in_benchmark_section = false;

    std::string line;
    std::size_t line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        const auto comment_pos = line.find_first_of("#;");
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        if (starts_with(line, "[") && line.back() == ']') {
            const std::string section_name = trim(line.substr(1, line.size() - 2));
            in_benchmark_section = (section_name == "benchmark");
            continue;
        }

        if (!in_benchmark_section) {
            continue;
        }

        const auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            throw std::runtime_error("invalid benchmark config line " + std::to_string(line_no));
        }

        const std::string key = trim(line.substr(0, eq_pos));
        const std::string value = trim(line.substr(eq_pos + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error("invalid benchmark key/value line " + std::to_string(line_no));
        }
        apply_benchmark_kv(config, key, value);
    }

    validate_benchmark_config(config);
    return config;
}

Stage2QualityAssessment evaluate_stage2_quality(const FirConfig& fir_config,
                                                const Stage2BenchmarkConfig& benchmark_config) {
    validate_benchmark_config(benchmark_config);
    if (fir_config.stages.empty()) {
        throw std::invalid_argument("fir_config must contain at least one stage");
    }

    const auto stage_taps = load_stage_taps(fir_config);
    MultiStageUpsampler upsampler(stage_taps);

    const auto impulse_input = make_impulse_input(benchmark_config);
    const auto step_input = make_step_input(benchmark_config);

    const auto impulse_response = upsampler.process_block(impulse_input);
    upsampler.reset();
    const auto step_response = upsampler.process_block(step_input);

    const std::size_t factor = upsampling_factor(fir_config.stages.size());
    const double output_sample_rate_hz =
        benchmark_config.source_sample_rate_hz * static_cast<double>(factor);

    Stage2QualityMetrics metrics;
    metrics.output_sample_rate_hz = output_sample_rate_hz;
    metrics.impulse_response = impulse_response;
    metrics.step_response = step_response;
    metrics.pre_echo_ms =
        compute_pre_echo_ms(impulse_response, benchmark_config, factor, output_sample_rate_hz);
    metrics.step_overshoot_ratio = compute_step_overshoot_ratio(
        step_response, benchmark_config, &metrics.step_peak, &metrics.step_reference);
    metrics.per_stage_band_metrics = compute_per_stage_band_metrics(fir_config, stage_taps);

    Stage2QualityAssessment assessment;
    assessment.pre_echo_within_limit = metrics.pre_echo_ms <= benchmark_config.max_pre_echo_ms;
    assessment.overshoot_within_limit =
        metrics.step_overshoot_ratio <= benchmark_config.max_step_overshoot_ratio;
    assessment.metrics = std::move(metrics);
    return assessment;
}

}  // namespace totton_audio_de_mirroring::dsp
