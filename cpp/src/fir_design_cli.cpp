#include "totton_audio_de_mirroring/dsp/fir_config.h"
#include "totton_audio_de_mirroring/dsp/fir_design.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

void print_metrics(const totton_audio_de_mirroring::dsp::FirDesignMetrics& metrics) {
    std::cout << "passband_ripple_db=" << metrics.passband_ripple_db << "\n";
    std::cout << "stopband_atten_db=" << metrics.stopband_atten_db << "\n";
    std::cout << "overshoot_ratio=" << metrics.overshoot_ratio << "\n";
    std::cout << "pre_echo_ms=" << metrics.pre_echo_ms << "\n";
    std::cout << "post_ringing_ms=" << metrics.post_ringing_ms << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: tadm_fir_design <config.ini>\n";
        return 1;
    }

    try {
        const std::filesystem::path config_path(argv[1]);
        const auto config = totton_audio_de_mirroring::dsp::load_fir_config(config_path);

        for (const auto& stage : config.stages) {
            std::cout << "[" << stage.name << "]\n";
            totton_audio_de_mirroring::dsp::FirDesignMetrics metrics;
            auto taps = totton_audio_de_mirroring::dsp::design_minimum_phase_lowpass(stage.design,
                                                                                     &metrics);

            const auto tap_dir = stage.taps_path.parent_path();
            if (!tap_dir.empty()) {
                std::filesystem::create_directories(tap_dir);
            }

            totton_audio_de_mirroring::dsp::write_taps_file(stage.taps_path, taps);
            print_metrics(metrics);
            std::cout << "taps_written=" << stage.taps_path.string() << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "FIR design failed: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
