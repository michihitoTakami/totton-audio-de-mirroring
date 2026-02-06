#include "totton_audio_de_mirroring/dsp/fir_design.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace totton_audio_de_mirroring::dsp {

namespace {

constexpr double kPi = 3.14159265358979323846;

void validate_spec(const FirDesignSpec& spec) {
    if (spec.sample_rate_hz <= 0.0) {
        throw std::invalid_argument("sample_rate_hz must be positive");
    }
    if (spec.passband_hz <= 0.0) {
        throw std::invalid_argument("passband_hz must be positive");
    }
    if (spec.stopband_hz <= spec.passband_hz) {
        throw std::invalid_argument("stopband_hz must be greater than passband_hz");
    }
    const double nyquist = 0.5 * spec.sample_rate_hz;
    if (spec.stopband_hz >= nyquist) {
        throw std::invalid_argument("stopband_hz must be less than Nyquist");
    }
    if (spec.attenuation_db <= 0.0) {
        throw std::invalid_argument("attenuation_db must be positive");
    }
    if (spec.passband_ripple_db <= 0.0) {
        throw std::invalid_argument("passband_ripple_db must be positive");
    }
    if (spec.min_taps < 3 || spec.max_taps < 3) {
        throw std::invalid_argument("tap counts must be >= 3");
    }
    if (spec.min_taps > spec.max_taps) {
        throw std::invalid_argument("min_taps must be <= max_taps");
    }
    if (spec.num_taps != 0 && (spec.num_taps % 2 == 0)) {
        throw std::invalid_argument("num_taps must be odd for FIR design");
    }
}

std::size_t ensure_odd(std::size_t value) {
    return (value % 2 == 0) ? value + 1 : value;
}

double sinc(double x) {
    if (std::fabs(x) < 1e-12) {
        return 1.0;
    }
    return std::sin(kPi * x) / (kPi * x);
}

double bessel_i0(double x) {
    double sum = 1.0;
    double y = (x * x) / 4.0;
    double t = y;
    for (int k = 1; k < 25; ++k) {
        sum += t;
        t *= y / (static_cast<double>(k + 1) * static_cast<double>(k + 1));
    }
    return sum;
}

double kaiser_beta(double attenuation_db) {
    if (attenuation_db > 50.0) {
        return 0.1102 * (attenuation_db - 8.7);
    }
    if (attenuation_db >= 21.0) {
        return 0.5842 * std::pow(attenuation_db - 21.0, 0.4) + 0.07886 * (attenuation_db - 21.0);
    }
    return 0.0;
}

std::vector<double> kaiser_window(std::size_t num_taps, double beta) {
    std::vector<double> window(num_taps, 1.0);
    if (num_taps == 1) {
        return window;
    }
    const double denom = bessel_i0(beta);
    const double n_minus_1 = static_cast<double>(num_taps - 1);
    for (std::size_t n = 0; n < num_taps; ++n) {
        const double ratio = 2.0 * static_cast<double>(n) / n_minus_1 - 1.0;
        const double value = bessel_i0(beta * std::sqrt(1.0 - ratio * ratio)) / denom;
        window[n] = value;
    }
    return window;
}

std::vector<std::complex<double>> dft_complex(const std::vector<std::complex<double>>& input) {
    const std::size_t nfft = input.size();
    std::vector<std::complex<double>> output(nfft);
    for (std::size_t k = 0; k < nfft; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (std::size_t n = 0; n < nfft; ++n) {
            const double angle =
                -2.0 * kPi * static_cast<double>(k * n) / static_cast<double>(nfft);
            sum += input[n] * std::exp(std::complex<double>(0.0, angle));
        }
        output[k] = sum;
    }
    return output;
}

std::vector<std::complex<double>> idft_complex(const std::vector<std::complex<double>>& input) {
    const std::size_t nfft = input.size();
    std::vector<std::complex<double>> output(nfft);
    for (std::size_t n = 0; n < nfft; ++n) {
        std::complex<double> sum(0.0, 0.0);
        for (std::size_t k = 0; k < nfft; ++k) {
            const double angle = 2.0 * kPi * static_cast<double>(k * n) / static_cast<double>(nfft);
            sum += input[k] * std::exp(std::complex<double>(0.0, angle));
        }
        output[n] = sum / static_cast<double>(nfft);
    }
    return output;
}

std::vector<double> minimum_phase_from_linear(const std::vector<double>& linear) {
    const std::size_t nfft = linear.size();
    std::vector<std::complex<double>> linear_complex(nfft);
    for (std::size_t i = 0; i < nfft; ++i) {
        linear_complex[i] = std::complex<double>(linear[i], 0.0);
    }

    auto spectrum = dft_complex(linear_complex);
    std::vector<std::complex<double>> log_mag(nfft);
    for (std::size_t k = 0; k < nfft; ++k) {
        const double mag = std::abs(spectrum[k]);
        const double safe_mag = std::max(mag, 1e-12);
        log_mag[k] = std::complex<double>(std::log(safe_mag), 0.0);
    }

    auto cepstrum = idft_complex(log_mag);
    std::vector<std::complex<double>> min_cepstrum(nfft, std::complex<double>(0.0, 0.0));
    min_cepstrum[0] = cepstrum[0];
    const std::size_t half = nfft / 2;
    for (std::size_t n = 1; n < half; ++n) {
        min_cepstrum[n] = 2.0 * cepstrum[n];
    }
    if (nfft % 2 == 0) {
        min_cepstrum[half] = cepstrum[half];
    }

    auto min_spec = dft_complex(min_cepstrum);
    for (auto& value : min_spec) {
        value = std::exp(value);
    }

    auto min_impulse = idft_complex(min_spec);
    std::vector<double> output(nfft);
    for (std::size_t i = 0; i < nfft; ++i) {
        output[i] = min_impulse[i].real();
    }
    return output;
}

double compute_passband_ripple_db(const std::vector<double>& taps, double sample_rate_hz,
                                  double passband_hz) {
    const std::size_t grid = 4096;
    double max_mag = 0.0;
    double min_mag = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i <= grid; ++i) {
        const double freq = passband_hz * static_cast<double>(i) / grid;
        const double omega = 2.0 * kPi * freq / sample_rate_hz;
        std::complex<double> sum(0.0, 0.0);
        for (std::size_t n = 0; n < taps.size(); ++n) {
            sum += taps[n] * std::exp(std::complex<double>(0.0, -omega * n));
        }
        const double mag = std::abs(sum);
        max_mag = std::max(max_mag, mag);
        min_mag = std::min(min_mag, mag);
    }
    if (min_mag <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return 20.0 * std::log10(max_mag / min_mag);
}

double compute_stopband_atten_db(const std::vector<double>& taps, double sample_rate_hz,
                                 double stopband_hz) {
    const std::size_t grid = 4096;
    const double nyquist = 0.5 * sample_rate_hz;
    double max_mag = 0.0;
    for (std::size_t i = 0; i <= grid; ++i) {
        const double freq = stopband_hz + (nyquist - stopband_hz) * static_cast<double>(i) / grid;
        const double omega = 2.0 * kPi * freq / sample_rate_hz;
        std::complex<double> sum(0.0, 0.0);
        for (std::size_t n = 0; n < taps.size(); ++n) {
            sum += taps[n] * std::exp(std::complex<double>(0.0, -omega * n));
        }
        max_mag = std::max(max_mag, std::abs(sum));
    }
    if (max_mag <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return -20.0 * std::log10(max_mag);
}

FirDesignMetrics analyze_fir_internal(const std::vector<double>& taps, double sample_rate_hz,
                                      double passband_hz, double stopband_hz) {
    FirDesignMetrics metrics;
    metrics.passband_ripple_db = compute_passband_ripple_db(taps, sample_rate_hz, passband_hz);
    metrics.stopband_atten_db = compute_stopband_atten_db(taps, sample_rate_hz, stopband_hz);

    double acc = 0.0;
    double max_step = -std::numeric_limits<double>::infinity();
    for (double tap : taps) {
        acc += tap;
        max_step = std::max(max_step, acc);
    }
    metrics.step_overshoot_ratio = std::max(0.0, max_step - 1.0);

    std::size_t peak_index = 0;
    double peak_mag = 0.0;
    for (std::size_t i = 0; i < taps.size(); ++i) {
        const double mag = std::fabs(taps[i]);
        if (mag > peak_mag) {
            peak_mag = mag;
            peak_index = i;
        }
    }
    std::size_t first_index = 0;
    const double threshold = peak_mag * 1e-3;
    while (first_index < taps.size() && std::fabs(taps[first_index]) < threshold) {
        ++first_index;
    }
    if (first_index >= taps.size()) {
        first_index = 0;
    }
    metrics.pre_echo_ms = static_cast<double>(first_index) / sample_rate_hz * 1000.0;

    std::size_t last_index = peak_index;
    for (std::size_t i = peak_index; i < taps.size(); ++i) {
        if (std::fabs(taps[i]) >= threshold) {
            last_index = i;
        }
    }
    metrics.post_ringing_ms =
        static_cast<double>(last_index - peak_index) / sample_rate_hz * 1000.0;

    const std::size_t samples = 8192;
    const double square_hz = 1000.0;
    std::vector<double> square(samples);
    for (std::size_t n = 0; n < samples; ++n) {
        const double phase = 2.0 * kPi * square_hz * static_cast<double>(n) / sample_rate_hz;
        square[n] = (std::sin(phase) >= 0.0) ? 1.0 : -1.0;
    }
    std::vector<double> filtered(samples, 0.0);
    for (std::size_t n = 0; n < samples; ++n) {
        double acc_sq = 0.0;
        const std::size_t tap_count = std::min(n + 1, taps.size());
        for (std::size_t k = 0; k < tap_count; ++k) {
            acc_sq += taps[k] * square[n - k];
        }
        filtered[n] = acc_sq;
    }
    double max_sq = -std::numeric_limits<double>::infinity();
    double min_sq = std::numeric_limits<double>::infinity();
    for (std::size_t n = 2048; n < samples; ++n) {
        max_sq = std::max(max_sq, filtered[n]);
        min_sq = std::min(min_sq, filtered[n]);
    }
    metrics.square_overshoot_ratio = std::max({0.0, max_sq - 1.0, -1.0 - min_sq});

    return metrics;
}

std::vector<double> normalize_dc(std::vector<double> taps) {
    double sum = 0.0;
    for (double tap : taps) {
        sum += tap;
    }
    if (std::fabs(sum) < 1e-12) {
        throw std::runtime_error("FIR taps sum to zero");
    }
    for (double& tap : taps) {
        tap /= sum;
    }
    return taps;
}

std::vector<double> design_minimum_phase_fixed(const FirDesignSpec& spec, std::size_t num_taps) {
    auto linear = design_linear_phase_lowpass(spec, num_taps);
    auto min_phase = minimum_phase_from_linear(linear);
    min_phase = normalize_dc(std::move(min_phase));
    return min_phase;
}

bool meets_targets(const FirDesignSpec& spec, const FirDesignMetrics& metrics) {
    if (metrics.stopband_atten_db + 1e-6 < spec.attenuation_db) {
        return false;
    }
    if (metrics.passband_ripple_db > spec.passband_ripple_db) {
        return false;
    }
    if (metrics.step_overshoot_ratio > spec.overshoot_max) {
        return false;
    }
    if (metrics.pre_echo_ms > spec.pre_echo_ms_max) {
        return false;
    }
    if (metrics.post_ringing_ms > spec.post_ringing_ms_max) {
        return false;
    }
    return true;
}

}  // namespace

std::vector<double> design_linear_phase_lowpass(const FirDesignSpec& spec, std::size_t num_taps) {
    if (num_taps < 3) {
        throw std::invalid_argument("num_taps must be >= 3");
    }
    if (num_taps % 2 == 0) {
        throw std::invalid_argument("num_taps must be odd for linear-phase design");
    }

    const double cutoff_hz = 0.5 * (spec.passband_hz + spec.stopband_hz);
    const double normalized_cutoff = 2.0 * cutoff_hz / spec.sample_rate_hz;

    std::vector<double> taps(num_taps);
    const double m = static_cast<double>(num_taps - 1) / 2.0;
    for (std::size_t n = 0; n < num_taps; ++n) {
        const double offset = static_cast<double>(n) - m;
        taps[n] = normalized_cutoff * sinc(normalized_cutoff * offset);
    }

    const double beta = kaiser_beta(spec.attenuation_db);
    const auto window = kaiser_window(num_taps, beta);
    for (std::size_t n = 0; n < num_taps; ++n) {
        taps[n] *= window[n];
    }

    return normalize_dc(std::move(taps));
}

FirDesignMetrics analyze_fir(const std::vector<double>& taps, double sample_rate_hz,
                             double passband_hz, double stopband_hz) {
    if (taps.empty()) {
        throw std::invalid_argument("taps must not be empty");
    }
    if (sample_rate_hz <= 0.0) {
        throw std::invalid_argument("sample_rate_hz must be positive");
    }
    if (passband_hz <= 0.0 || stopband_hz <= passband_hz) {
        throw std::invalid_argument("invalid passband/stopband");
    }
    return analyze_fir_internal(taps, sample_rate_hz, passband_hz, stopband_hz);
}

std::vector<double> design_minimum_phase_lowpass(const FirDesignSpec& spec,
                                                 FirDesignMetrics* metrics) {
    validate_spec(spec);

    if (spec.num_taps != 0) {
        auto taps = design_minimum_phase_fixed(spec, spec.num_taps);
        if (metrics != nullptr) {
            *metrics =
                analyze_fir_internal(taps, spec.sample_rate_hz, spec.passband_hz, spec.stopband_hz);
        }
        return taps;
    }

    std::size_t min_taps = ensure_odd(spec.min_taps);
    std::size_t max_taps = ensure_odd(spec.max_taps);
    if (min_taps > max_taps) {
        throw std::invalid_argument("min_taps must be <= max_taps after odd adjustment");
    }

    FirDesignMetrics best_metrics{};
    std::size_t best_taps_count = 0;
    bool has_best = false;
    for (std::size_t taps = min_taps; taps <= max_taps; taps += 2) {
        auto candidate = design_minimum_phase_fixed(spec, taps);
        auto candidate_metrics = analyze_fir_internal(candidate, spec.sample_rate_hz,
                                                      spec.passband_hz, spec.stopband_hz);
        if (meets_targets(spec, candidate_metrics)) {
            if (metrics != nullptr) {
                *metrics = candidate_metrics;
            }
            return candidate;
        }
        if (!has_best || candidate_metrics.stopband_atten_db > best_metrics.stopband_atten_db) {
            best_metrics = candidate_metrics;
            best_taps_count = taps;
            has_best = true;
        }
    }

    if (metrics != nullptr) {
        *metrics = best_metrics;
    }
    if (has_best) {
        std::ostringstream oss;
        oss << "failed to satisfy design constraints up to " << max_taps << " taps; "
            << "best candidate taps=" << best_taps_count
            << ", stopband_atten_db=" << best_metrics.stopband_atten_db
            << ", passband_ripple_db=" << best_metrics.passband_ripple_db
            << ", step_overshoot_ratio=" << best_metrics.step_overshoot_ratio
            << ", square_overshoot_ratio=" << best_metrics.square_overshoot_ratio
            << ", pre_echo_ms=" << best_metrics.pre_echo_ms
            << ", post_ringing_ms=" << best_metrics.post_ringing_ms;
        throw std::runtime_error(oss.str());
    }

    throw std::runtime_error("failed to design minimum-phase FIR");
}

}  // namespace totton_audio_de_mirroring::dsp
