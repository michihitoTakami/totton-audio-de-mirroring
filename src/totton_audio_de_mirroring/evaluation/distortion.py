"""Coherent-line distortion measurements for CAPB diagnostics."""

from __future__ import annotations

import numpy as np

_EPSILON = 1.0e-300


def tone_amplitude(
    signal: np.ndarray,
    sample_rate: int,
    frequency_hz: float,
) -> float:
    """Measure the peak amplitude of one coherent sinusoidal line.

    Args:
        signal: One-dimensional finite waveform containing an integer number
            of cycles at ``frequency_hz``.
        sample_rate: Sample rate in Hz.
        frequency_hz: Analysis frequency in Hz.

    Returns:
        Peak sinusoidal amplitude from a complex projection.

    Raises:
        ValueError: If the waveform or frequency is invalid.

    Physical Basis:
        Projecting directly onto the requested sinusoid avoids FFT-bin
        interpolation and window sidelobes. The report uses one-second,
        integer-Hz analysis regions so every measured component is coherent.
    """
    values = _validate_signal(signal, sample_rate)
    if not 0.0 < frequency_hz < sample_rate / 2.0:
        raise ValueError("frequency_hz must lie between zero and Nyquist.")
    sample_indices = np.arange(values.size, dtype=np.float64)
    basis = np.exp(-2.0j * np.pi * frequency_hz * sample_indices / sample_rate)
    return float(abs(2.0 * np.sum(values * basis) / values.size))


def thd_db(
    signal: np.ndarray,
    sample_rate: int,
    fundamental_hz: float = 1_000.0,
    max_frequency_hz: float = 20_000.0,
) -> float:
    """Measure audio-band total harmonic distortion in dBc.

    Args:
        signal: Coherent single-tone analysis waveform.
        sample_rate: Sample rate in Hz.
        fundamental_hz: Fundamental tone frequency in Hz.
        max_frequency_hz: Highest harmonic included in the RSS sum.

    Returns:
        Root-sum-square harmonic amplitude relative to the fundamental, dB.

    Raises:
        ValueError: If no valid harmonics fit in the requested band.

    Physical Basis:
        An LTI interpolation FIR cannot create harmonics. Restricting the
        sum to 20 kHz prevents the interpolation image from being mislabeled
        as a high-order harmonic when it happens to share an integer bin.
    """
    values = _validate_signal(signal, sample_rate)
    upper_hz = min(max_frequency_hz, np.nextafter(sample_rate / 2.0, 0.0))
    max_order = int(np.floor(upper_hz / fundamental_hz))
    if fundamental_hz <= 0.0 or max_order < 2:
        raise ValueError("At least one harmonic must fit in the analysis band.")
    fundamental = tone_amplitude(values, sample_rate, fundamental_hz)
    harmonics = tuple(
        tone_amplitude(values, sample_rate, order * fundamental_hz)
        for order in range(2, max_order + 1)
    )
    return _ratio_db(float(np.linalg.norm(harmonics)), fundamental)


def smpte_imd_db(
    signal: np.ndarray,
    sample_rate: int,
    low_tone_hz: float = 60.0,
    high_tone_hz: float = 7_000.0,
    max_sideband_order: int = 5,
) -> float:
    """Measure SMPTE-style IMD sidebands around the high tone in dBc.

    Args:
        signal: Coherent two-tone analysis waveform.
        sample_rate: Sample rate in Hz.
        low_tone_hz: Low-frequency tone in Hz.
        high_tone_hz: High-frequency tone in Hz.
        max_sideband_order: Number of sideband pairs included.

    Returns:
        RSS of ``high ± n*low`` lines relative to the high tone, dB.

    Raises:
        ValueError: If the requested sidebands are invalid.

    Physical Basis:
        A linear fixed filter preserves the two input lines without creating
        the symmetric sideband family. CAPB blend-weight modulation can
        multiply the prototype spread by the low tone and create those lines.
    """
    values = _validate_signal(signal, sample_rate)
    frequencies = _smpte_sideband_frequencies(
        sample_rate, low_tone_hz, high_tone_hz, max_sideband_order
    )
    carrier = tone_amplitude(values, sample_rate, high_tone_hz)
    products = tuple(tone_amplitude(values, sample_rate, freq) for freq in frequencies)
    return _ratio_db(float(np.linalg.norm(products)), carrier)


def ccif_imd_db(
    signal: np.ndarray,
    sample_rate: int,
    lower_tone_hz: float = 19_000.0,
    upper_tone_hz: float = 20_000.0,
) -> float:
    """Measure difference- and third-order CCIF products in dBc.

    Args:
        signal: Coherent equal-amplitude two-tone waveform.
        sample_rate: Sample rate in Hz.
        lower_tone_hz: Lower primary frequency in Hz.
        upper_tone_hz: Upper primary frequency in Hz.

    Returns:
        RSS of ``f2-f1``, ``2*f1-f2``, and ``2*f2-f1`` relative to the RSS
        amplitude of the two primaries, dB.

    Raises:
        ValueError: If a primary or product lies outside the measurable band.

    Physical Basis:
        The 1 kHz difference product and adjacent third-order products expose
        nonlinear interaction near the upper audible-band edge without
        treating legitimate interpolation images as distortion.
    """
    values = _validate_signal(signal, sample_rate)
    products_hz = (
        upper_tone_hz - lower_tone_hz,
        2.0 * lower_tone_hz - upper_tone_hz,
        2.0 * upper_tone_hz - lower_tone_hz,
    )
    all_frequencies = (lower_tone_hz, upper_tone_hz, *products_hz)
    if any(freq <= 0.0 or freq >= sample_rate / 2.0 for freq in all_frequencies):
        raise ValueError("CCIF primaries and products must lie below Nyquist.")
    primaries = (
        tone_amplitude(values, sample_rate, lower_tone_hz),
        tone_amplitude(values, sample_rate, upper_tone_hz),
    )
    products = tuple(
        tone_amplitude(values, sample_rate, frequency_hz)
        for frequency_hz in products_hz
    )
    return _ratio_db(float(np.linalg.norm(products)), float(np.linalg.norm(primaries)))


def added_am_sideband_db(
    signal: np.ndarray,
    sample_rate: int,
    carrier_hz: float = 10_000.0,
    modulation_hz: float = 37.0,
    first_added_order: int = 2,
    max_order: int = 6,
) -> float:
    """Measure the strongest sideband absent from a sinusoidal AM input.

    Args:
        signal: Coherent amplitude-modulated waveform.
        sample_rate: Sample rate in Hz.
        carrier_hz: AM carrier in Hz.
        modulation_hz: AM modulation frequency in Hz.
        first_added_order: First sideband order absent from the input.
        max_order: Last sideband order inspected.

    Returns:
        Strongest added sideband relative to the carrier, dB.

    Raises:
        ValueError: If the order range or frequencies are invalid.

    Physical Basis:
        Sinusoidal AM contains only the carrier and first sideband pair.
        Higher-order pairs therefore isolate modulation added by the
        time-varying interpolation path.
    """
    values = _validate_signal(signal, sample_rate)
    if first_added_order < 2 or max_order < first_added_order:
        raise ValueError("AM sideband orders must satisfy 2 <= first <= max.")
    frequencies = tuple(
        carrier_hz + sign * order * modulation_hz
        for order in range(first_added_order, max_order + 1)
        for sign in (-1.0, 1.0)
    )
    if any(freq <= 0.0 or freq >= sample_rate / 2.0 for freq in frequencies):
        raise ValueError("AM sidebands must lie between zero and Nyquist.")
    carrier = tone_amplitude(values, sample_rate, carrier_hz)
    strongest = max(tone_amplitude(values, sample_rate, freq) for freq in frequencies)
    return _ratio_db(strongest, carrier)


def relative_line_levels_db(
    signal: np.ndarray,
    sample_rate: int,
    frequencies_hz: tuple[float, ...],
    reference_hz: float,
) -> tuple[float, ...]:
    """Return coherent spectral-line levels relative to one reference line.

    Args:
        signal: Coherent analysis waveform.
        sample_rate: Sample rate in Hz.
        frequencies_hz: Frequencies to measure.
        reference_hz: Reference frequency for dBc normalization.

    Returns:
        Tuple of line levels in dBc, index-aligned with ``frequencies_hz``.

    Physical Basis:
        Explicit line measurements show individual sideband or harmonic
        structure without conflating it with broadband noise or FFT leakage.
    """
    values = _validate_signal(signal, sample_rate)
    reference = tone_amplitude(values, sample_rate, reference_hz)
    return tuple(
        _ratio_db(tone_amplitude(values, sample_rate, freq), reference)
        for freq in frequencies_hz
    )


def _smpte_sideband_frequencies(
    sample_rate: int,
    low_tone_hz: float,
    high_tone_hz: float,
    max_sideband_order: int,
) -> tuple[float, ...]:
    """Build and validate the SMPTE sideband frequency tuple."""
    if max_sideband_order <= 0:
        raise ValueError("max_sideband_order must be positive.")
    frequencies = tuple(
        high_tone_hz + sign * order * low_tone_hz
        for order in range(1, max_sideband_order + 1)
        for sign in (-1.0, 1.0)
    )
    all_frequencies = (low_tone_hz, high_tone_hz, *frequencies)
    if any(freq <= 0.0 or freq >= sample_rate / 2.0 for freq in all_frequencies):
        raise ValueError("SMPTE primaries and sidebands must lie below Nyquist.")
    return frequencies


def _ratio_db(numerator: float, denominator: float) -> float:
    """Convert an amplitude ratio to decibels with a numerical floor."""
    if denominator <= 0.0:
        raise ValueError("Reference amplitude must be positive.")
    return float(20.0 * np.log10(max(numerator / denominator, _EPSILON)))


def _validate_signal(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Validate and copy a one-dimensional analysis waveform."""
    values = np.array(signal, dtype=np.float64, copy=True)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty 1D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    return values
