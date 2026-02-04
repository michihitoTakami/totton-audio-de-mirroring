import numpy as np
import pytest

from totton_audio_de_mirroring.models.band_split import (
    BandSplitConfig,
    BandSplitProcessor,
    compensate_delay,
)

SAMPLE_RATE = 88_200
CUTOFF_HZ = 20_000.0
NUM_TAPS = 1025


def _make_processor() -> BandSplitProcessor:
    config = BandSplitConfig(
        cutoff_hz=CUTOFF_HZ,
        sample_rate=SAMPLE_RATE,
        num_taps=NUM_TAPS,
    )
    return BandSplitProcessor(config)


def test_band_split_delay_samples_matches_config() -> None:
    config = BandSplitConfig(
        cutoff_hz=CUTOFF_HZ,
        sample_rate=SAMPLE_RATE,
        num_taps=NUM_TAPS,
    )
    processor = BandSplitProcessor(config)

    assert processor.delay_samples == config.delay_samples()
    assert processor.delay_samples == (NUM_TAPS - 1) // 2


def test_band_split_recombines_after_delay_compensation() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(0.0, 1.0, 4096)

    processor = _make_processor()
    result = processor.process(signal)

    aligned_recombined = processor.compensate_delay(result.recombined)
    aligned_original = signal[: -processor.delay_samples]

    error = float(np.mean(np.abs(aligned_recombined - aligned_original)))
    assert error < 1e-2


def test_band_split_supports_2d_signal() -> None:
    rng = np.random.default_rng(1)
    signal = rng.normal(0.0, 1.0, (2, 4096))

    processor = _make_processor()
    result = processor.process(signal)

    assert result.low_band.shape == signal.shape
    assert result.high_band.shape == signal.shape
    assert result.recombined.shape == signal.shape


def test_high_band_processor_shape_mismatch_raises() -> None:
    rng = np.random.default_rng(2)
    signal = rng.normal(0.0, 1.0, 2048)

    processor = _make_processor()

    def bad_processor(high_band: np.ndarray) -> np.ndarray:
        return high_band[:-1]

    with pytest.raises(ValueError, match="same shape"):
        processor.process(signal, high_band_processor=bad_processor)


def test_compensate_delay_rejects_invalid_length() -> None:
    signal = np.zeros(16)

    with pytest.raises(ValueError, match="signal length"):
        compensate_delay(signal, delay_samples=16)
