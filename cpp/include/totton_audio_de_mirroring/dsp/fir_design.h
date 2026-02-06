#ifndef TOTTON_AUDIO_DE_MIRRORING_DSP_FIR_DESIGN_H
#define TOTTON_AUDIO_DE_MIRRORING_DSP_FIR_DESIGN_H

#include <cstddef>
#include <string>
#include <vector>

namespace totton_audio_de_mirroring::dsp {

struct FirDesignSpec {
    double sample_rate_hz = 0.0;
    double passband_hz = 0.0;
    double stopband_hz = 0.0;
    double attenuation_db = 60.0;
    double passband_ripple_db = 0.1;
    double overshoot_max = 0.05;
    double pre_echo_ms_max = 0.1;
    double post_ringing_ms_max = 2.0;
    std::size_t num_taps = 0;
    std::size_t min_taps = 17;
    std::size_t max_taps = 63;
    std::string label;
};

struct FirDesignMetrics {
    double passband_ripple_db = 0.0;
    double stopband_atten_db = 0.0;
    double step_overshoot_ratio = 0.0;
    double square_overshoot_ratio = 0.0;
    double pre_echo_ms = 0.0;
    double post_ringing_ms = 0.0;
};

std::vector<double> design_minimum_phase_lowpass(const FirDesignSpec& spec,
                                                 FirDesignMetrics* metrics = nullptr);

FirDesignMetrics analyze_fir(const std::vector<double>& taps, double sample_rate_hz,
                             double passband_hz, double stopband_hz);

std::vector<double> design_linear_phase_lowpass(const FirDesignSpec& spec, std::size_t num_taps);

}  // namespace totton_audio_de_mirroring::dsp

#endif  // TOTTON_AUDIO_DE_MIRRORING_DSP_FIR_DESIGN_H
