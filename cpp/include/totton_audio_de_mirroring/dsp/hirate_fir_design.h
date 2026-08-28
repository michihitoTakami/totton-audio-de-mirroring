#ifndef TOTTON_AUDIO_DE_MIRRORING_DSP_HIRATE_FIR_DESIGN_H
#define TOTTON_AUDIO_DE_MIRRORING_DSP_HIRATE_FIR_DESIGN_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace totton_audio_de_mirroring::dsp::hirate_fir {

// Stage 2 is a transport-only 8x cascade. Each 2x stage cuts at its input
// Nyquist so the already-decided CAPB audible band stays flat while new
// zero-stuffing images are removed without synthesizing high-frequency data.
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kKaiserBeta = 16.0;
inline constexpr std::array<std::size_t, 3> kStageTaps8x = {255, 63, 39};

struct StageDesign {
    std::size_t taps = 0;
    double cutoff_hz = 0.0;
    double output_rate_hz = 0.0;
};

/** Evaluate the normalized sinc function.
 *
 * Physical Basis:
 *     The ideal low-pass impulse response is a scaled sinc function.
 *
 * Args:
 *     value: Dimensionless sample offset.
 *
 * Returns:
 *     sin(pi * value) / (pi * value), including its limit at zero.
 */
inline double sinc(double value) {
    if (std::abs(value) < 1e-12) {
        return 1.0;
    }
    return std::sin(kPi * value) / (kPi * value);
}

/** Evaluate the modified Bessel function I0 for Kaiser window design.
 *
 * Physical Basis:
 *     Kaiser windows use I0 to control the stopband-versus-transition-width
 *     tradeoff without changing the symmetric linear-phase structure.
 *
 * Args:
 *     value: Real-valued Bessel function argument.
 *
 * Returns:
 *     The I0 series approximation used by the canonical HiRate designer.
 */
inline double bessel_i0(double value) {
    const double half_value = 0.5 * value;
    double term = 1.0;
    double sum = 1.0;
    for (int order = 1; order < 64; ++order) {
        const double factor = half_value / static_cast<double>(order);
        term *= factor * factor;
        sum += term;
        if (term < sum * 1e-18) {
            break;
        }
    }
    return sum;
}

/** Design one canonical gain-2 interpolation kernel.
 *
 * Physical Basis:
 *     A symmetric Kaiser-windowed sinc removes the image introduced by 2x
 *     zero stuffing. The cutoff is limited to 0.249 of the output rate to
 *     match Totton Audio NN's transport FIR exactly.
 *
 * Args:
 *     taps: Odd kernel length.
 *     cutoff_hz: Low-pass cutoff in Hz.
 *     output_rate_hz: Sample rate after 2x interpolation.
 *
 * Returns:
 *     Float32-compatible coefficients normalized to DC gain 2, or an empty
 *     vector when the request is invalid.
 */
inline std::vector<float> design_interpolation_kernel(std::size_t taps, double cutoff_hz,
                                                      double output_rate_hz) {
    if (taps < 3 || taps % 2 == 0 || cutoff_hz <= 0.0 || output_rate_hz <= 0.0) {
        return {};
    }

    const double normalized_cutoff = std::clamp(cutoff_hz / output_rate_hz, 0.001, 0.249);
    const double center = static_cast<double>(taps - 1) * 0.5;
    const double window_norm = bessel_i0(kKaiserBeta);
    std::vector<double> design(taps, 0.0);
    double sum = 0.0;
    for (std::size_t index = 0; index < taps; ++index) {
        const double offset = static_cast<double>(index) - center;
        const double ideal = 2.0 * normalized_cutoff * sinc(2.0 * normalized_cutoff * offset);
        const double relative = offset / center;
        const double window =
            bessel_i0(kKaiserBeta * std::sqrt(std::max(0.0, 1.0 - relative * relative))) /
            window_norm;
        design[index] = ideal * window;
        sum += design[index];
    }

    if (std::abs(sum) <= 1e-12) {
        return {};
    }
    const double scale = 2.0 / sum;
    std::vector<float> kernel(taps, 0.0F);
    for (std::size_t index = 0; index < taps; ++index) {
        kernel[index] = static_cast<float>(design[index] * scale);
    }
    return kernel;
}

/** Convert the canonical kernel to this runtime's stored-tap convention.
 *
 * Physical Basis:
 *     FirUpsampler2x supplies interpolation gain 2 outside the convolution,
 *     so stored coefficients must have half the canonical gain.
 *
 * Args:
 *     taps: Odd kernel length.
 *     cutoff_hz: Low-pass cutoff in Hz.
 *     output_rate_hz: Sample rate after 2x interpolation.
 *
 * Returns:
 *     Double-precision storage values derived from canonical float32 taps.
 */
inline std::vector<double> design_runtime_storage_taps(std::size_t taps, double cutoff_hz,
                                                       double output_rate_hz) {
    const auto effective_kernel = design_interpolation_kernel(taps, cutoff_hz, output_rate_hz);
    std::vector<double> storage_taps;
    storage_taps.reserve(effective_kernel.size());
    // FirUpsampler2x applies the interpolation gain externally. Store half
    // of the canonical gain-2 kernel so its effective transfer is unchanged.
    for (float tap : effective_kernel) {
        storage_taps.push_back(0.5 * static_cast<double>(tap));
    }
    return storage_taps;
}

/** Build the fixed three-stage 8x transport plan.
 *
 * Physical Basis:
 *     Each stage cuts at its input Nyquist, retaining only information already
 *     represented by CAPB while rejecting the corresponding interpolation
 *     image. The 255/63/39 ladder matches Totton Audio NN.
 *
 * Args:
 *     input_rate_hz: CAPB output rate entering Stage 2.
 *
 * Returns:
 *     Three 2x stage specifications, or zero-initialized stages for an invalid
 *     input rate.
 */
inline std::array<StageDesign, 3> design_8x_plan(double input_rate_hz) {
    if (input_rate_hz <= 0.0) {
        return {};
    }
    return {
        StageDesign{kStageTaps8x[0], 0.5 * input_rate_hz, 2.0 * input_rate_hz},
        StageDesign{kStageTaps8x[1], input_rate_hz, 4.0 * input_rate_hz},
        StageDesign{kStageTaps8x[2], 2.0 * input_rate_hz, 8.0 * input_rate_hz},
    };
}

}  // namespace totton_audio_de_mirroring::dsp::hirate_fir

#endif  // TOTTON_AUDIO_DE_MIRRORING_DSP_HIRATE_FIR_DESIGN_H
