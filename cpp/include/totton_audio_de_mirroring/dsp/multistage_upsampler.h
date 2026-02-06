#ifndef TOTTON_AUDIO_DE_MIRRORING_DSP_MULTISTAGE_UPSAMPLER_H
#define TOTTON_AUDIO_DE_MIRRORING_DSP_MULTISTAGE_UPSAMPLER_H

#include <cstddef>
#include <filesystem>
#include <vector>

namespace totton_audio_de_mirroring::dsp {

struct FirUpsampler2xConfig {
    std::vector<double> taps;
};

class FirUpsampler2x {
   public:
    explicit FirUpsampler2x(FirUpsampler2xConfig config);
    explicit FirUpsampler2x(std::vector<double> taps);

    std::vector<double> process_block(const std::vector<double>& input);
    std::vector<double> process_block(const double* input, std::size_t length);

    static FirUpsampler2x from_taps_file(const std::filesystem::path& taps_path);

    void reset();
    std::size_t taps_size() const;
    std::size_t history_size() const;
    std::size_t output_length_for_input(std::size_t input_length) const;
    double steady_state_dc_gain_even() const;
    double steady_state_dc_gain_odd() const;

   private:
    std::vector<double> even_taps_;
    std::vector<double> odd_taps_;
    std::vector<double> history_;
    std::size_t history_pos_;

    double dot(const std::vector<double>& taps) const;
    void validate_taps(const std::vector<double>& taps) const;
};

class MultiStageUpsampler {
   public:
    explicit MultiStageUpsampler(std::vector<std::vector<double>> stage_taps);

    std::vector<double> process_block(const std::vector<double>& input);
    std::vector<double> process_block(const double* input, std::size_t length);

    void reset();
    std::size_t stage_count() const;

   private:
    std::vector<FirUpsampler2x> stages_;

    void validate_stage_taps(const std::vector<std::vector<double>>& stage_taps) const;
};

}  // namespace totton_audio_de_mirroring::dsp

#endif  // TOTTON_AUDIO_DE_MIRRORING_DSP_MULTISTAGE_UPSAMPLER_H
