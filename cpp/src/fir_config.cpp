#include "totton_audio_de_mirroring/dsp/fir_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

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

void apply_kv(FirStageConfig& stage, const std::string& key, const std::string& value) {
    if (key == "design_kind") {
        stage.design.design_kind = value;
    } else if (key == "sample_rate_hz") {
        stage.design.sample_rate_hz = std::stod(value);
    } else if (key == "passband_hz") {
        stage.design.passband_hz = std::stod(value);
    } else if (key == "stopband_hz") {
        stage.design.stopband_hz = std::stod(value);
    } else if (key == "cutoff_hz") {
        stage.design.cutoff_hz = std::stod(value);
    } else if (key == "attenuation_db") {
        stage.design.attenuation_db = std::stod(value);
    } else if (key == "passband_ripple_db") {
        stage.design.passband_ripple_db = std::stod(value);
    } else if (key == "overshoot_max") {
        stage.design.overshoot_max = std::stod(value);
    } else if (key == "pre_echo_ms_max") {
        stage.design.pre_echo_ms_max = std::stod(value);
    } else if (key == "post_ringing_ms_max") {
        stage.design.post_ringing_ms_max = std::stod(value);
    } else if (key == "num_taps") {
        stage.design.num_taps = static_cast<std::size_t>(std::stoul(value));
    } else if (key == "min_taps") {
        stage.design.min_taps = static_cast<std::size_t>(std::stoul(value));
    } else if (key == "max_taps") {
        stage.design.max_taps = static_cast<std::size_t>(std::stoul(value));
    } else if (key == "taps_path") {
        stage.taps_path = value;
    } else {
        throw std::invalid_argument("unknown key: " + key);
    }
}

void validate_stage(const FirStageConfig& stage) {
    if (stage.name.empty()) {
        throw std::invalid_argument("stage name is required");
    }
    if (stage.design.sample_rate_hz <= 0.0) {
        throw std::invalid_argument("sample_rate_hz must be positive");
    }
    if (stage.design.passband_hz <= 0.0) {
        throw std::invalid_argument("passband_hz must be positive");
    }
    if (stage.design.stopband_hz <= stage.design.passband_hz) {
        throw std::invalid_argument("stopband_hz must exceed passband_hz");
    }
    if (stage.design.design_kind != "minimum_phase" &&
        stage.design.design_kind != "hirate_linear") {
        throw std::invalid_argument("design_kind must be minimum_phase or hirate_linear");
    }
    if (stage.design.design_kind == "hirate_linear") {
        const double nyquist = 0.5 * stage.design.sample_rate_hz;
        if (stage.design.cutoff_hz <= 0.0 || stage.design.cutoff_hz >= nyquist) {
            throw std::invalid_argument("cutoff_hz must be in (0, Nyquist)");
        }
        if (stage.design.num_taps < 3 || stage.design.num_taps % 2 == 0) {
            throw std::invalid_argument("hirate_linear num_taps must be an odd value >= 3");
        }
    }
    if (stage.taps_path.empty()) {
        throw std::invalid_argument("taps_path is required");
    }
}

}  // namespace

FirConfig load_fir_config(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("config file not found: " + path.string());
    }

    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open config file: " + path.string());
    }

    FirConfig config;
    FirStageConfig current;
    bool has_section = false;

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
            if (has_section) {
                validate_stage(current);
                config.stages.push_back(current);
            }
            current = FirStageConfig{};
            current.name = trim(line.substr(1, line.size() - 2));
            current.design.label = current.name;
            has_section = true;
            continue;
        }

        const auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            throw std::runtime_error("invalid config line " + std::to_string(line_no));
        }
        if (!has_section) {
            throw std::runtime_error("key-value outside of section at line " +
                                     std::to_string(line_no));
        }
        const std::string key = trim(line.substr(0, eq_pos));
        const std::string value = trim(line.substr(eq_pos + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error("invalid key/value at line " + std::to_string(line_no));
        }
        apply_kv(current, key, value);
    }

    if (has_section) {
        validate_stage(current);
        config.stages.push_back(current);
    }

    if (config.stages.empty()) {
        throw std::runtime_error("no stages defined in config");
    }

    return config;
}

void write_taps_file(const std::filesystem::path& path, const std::vector<double>& taps) {
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("failed to write taps file: " + path.string());
    }
    for (double tap : taps) {
        output << std::setprecision(std::numeric_limits<double>::max_digits10) << std::scientific
               << tap << "\n";
    }
}

std::vector<double> read_taps_file(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("taps file not found: " + path.string());
    }
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open taps file: " + path.string());
    }

    std::vector<double> taps;
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || starts_with(line, "#")) {
            continue;
        }
        taps.push_back(std::stod(line));
    }
    if (taps.empty()) {
        throw std::runtime_error("taps file is empty: " + path.string());
    }
    return taps;
}

}  // namespace totton_audio_de_mirroring::dsp
