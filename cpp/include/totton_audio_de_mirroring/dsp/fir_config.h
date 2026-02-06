#ifndef TOTTON_AUDIO_DE_MIRRORING_DSP_FIR_CONFIG_H
#define TOTTON_AUDIO_DE_MIRRORING_DSP_FIR_CONFIG_H

#include "totton_audio_de_mirroring/dsp/fir_design.h"

#include <filesystem>
#include <string>
#include <vector>

namespace totton_audio_de_mirroring::dsp {

struct FirStageConfig {
    std::string name;
    FirDesignSpec design;
    std::filesystem::path taps_path;
};

struct FirConfig {
    std::vector<FirStageConfig> stages;
};

FirConfig load_fir_config(const std::filesystem::path& path);

void write_taps_file(const std::filesystem::path& path, const std::vector<double>& taps);

std::vector<double> read_taps_file(const std::filesystem::path& path);

}  // namespace totton_audio_de_mirroring::dsp

#endif  // TOTTON_AUDIO_DE_MIRRORING_DSP_FIR_CONFIG_H
