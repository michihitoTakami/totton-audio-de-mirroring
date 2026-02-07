"""Mirror/aliasing reduction metrics and visualization utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from scipy import stats as sp_stats

DEFAULT_MIRROR_BAND_HZ = (20_000.0, 22_050.0)
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_WINDOW = "hann"
EPSILON = 1.0e-12


@dataclass(frozen=True)
class MirrorReductionMetrics:
    """Quantitative mirror/aliasing reduction metrics.

    Args:
        symmetry_score_before: STFT symmetry score around mirror center before.
        symmetry_score_after: STFT symmetry score around mirror center after.
        symmetry_reduction_ratio: Relative reduction in symmetry score.
        mirror_band_energy_before: Mean STFT energy in mirror band before.
        mirror_band_energy_after: Mean STFT energy in mirror band after.
        mirror_band_energy_reduction_ratio: Relative reduction in mirror-band energy.
        stripe_score_before: Temporal modulation score in mirror band before.
        stripe_score_after: Temporal modulation score in mirror band after.
        stripe_reduction_ratio: Relative reduction in stripe score.

    Physical Basis:
        Mirror artifacts appear as symmetric high-band structures around
        Nyquist and often create stripe-like temporal modulation in STFT.
        Reduction is assessed by symmetry, energy, and stripe proxies.
    """

    symmetry_score_before: float
    symmetry_score_after: float
    symmetry_reduction_ratio: float
    mirror_band_energy_before: float
    mirror_band_energy_after: float
    mirror_band_energy_reduction_ratio: float
    stripe_score_before: float
    stripe_score_after: float
    stripe_reduction_ratio: float


@dataclass(frozen=True)
class MirrorVisualizationArtifacts:
    """Saved artifact paths for mirror reduction inspection plots.

    Args:
        plot_path: Exported PNG path for before/after comparison.

    Physical Basis:
        Visual STFT comparison helps verify that mirror-like structures are
        attenuated without relying solely on scalar metrics.
    """

    plot_path: Path


@dataclass(frozen=True)
class MirrorSampleEvaluationResult:
    """Per-sample mirror reduction evaluation payload.

    Args:
        sample_id: Sample identifier, usually file stem.
        metrics: Mirror reduction metrics for the sample.

    Physical Basis:
        Mirror suppression quality can vary per sample, so outliers are
        tracked at sample granularity.
    """

    sample_id: str
    metrics: MirrorReductionMetrics


@dataclass(frozen=True)
class MirrorDatasetEvaluationResult:
    """Dataset-level mirror reduction aggregate.

    Args:
        samples: Per-sample mirror reduction results.
        mean_metrics: Arithmetic mean of mirror metrics across samples.
        target_reduction_ratio: Target symmetry reduction threshold.
        symmetry_target_pass_rate: Fraction of samples meeting target reduction.

    Physical Basis:
        Acceptance requires stable mirror reduction across a dataset,
        not only isolated examples.
    """

    samples: tuple[MirrorSampleEvaluationResult, ...]
    mean_metrics: MirrorReductionMetrics
    target_reduction_ratio: float
    symmetry_target_pass_rate: float


@dataclass(frozen=True)
class ListeningCorrelationMetrics:
    """Correlation between objective mirror metrics and listening scores.

    Args:
        pearson_correlation: Pearson linear correlation coefficient.
        spearman_correlation: Spearman rank correlation coefficient.
        num_pairs: Number of paired samples used.

    Physical Basis:
        Correlation quantifies whether objective mirror metrics align
        with subjective listening quality trends.
    """

    pearson_correlation: float
    spearman_correlation: float
    num_pairs: int


def evaluate_mirror_reduction(
    before_signal: np.ndarray,
    after_signal: np.ndarray,
    sample_rate: int,
    mirror_band_hz: tuple[float, float] = DEFAULT_MIRROR_BAND_HZ,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    mirror_center_hz: float | None = None,
) -> MirrorReductionMetrics:
    """Evaluate mirror/aliasing reduction from before/after signals.

    Args:
        before_signal: Signal before mirror suppression.
        after_signal: Signal after mirror suppression.
        sample_rate: Sample rate in Hz.
        mirror_band_hz: Frequency band used for mirror analysis.
        n_fft: STFT FFT size.
        hop_length: STFT hop size.
        mirror_center_hz: Symmetry center frequency in Hz.

    Returns:
        MirrorReductionMetrics with symmetry, energy, and stripe reductions.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Mirror artifacts are characterized by energy symmetry around the
        source Nyquist and elevated energy in the 20-22kHz band.
    """

    _validate_signal_pair(before_signal, after_signal)
    _validate_sample_rate(sample_rate)
    _validate_stft_params(n_fft, hop_length)
    _validate_mirror_band(mirror_band_hz, sample_rate)

    center_hz = _resolve_mirror_center_hz(sample_rate, mirror_center_hz)
    _validate_mirror_center(center_hz, sample_rate)

    freqs, _, stft_before = _stft(before_signal, sample_rate, n_fft, hop_length)
    _, _, stft_after = _stft(after_signal, sample_rate, n_fft, hop_length)

    mag_before = np.abs(stft_before)
    mag_after = np.abs(stft_after)

    symmetry_before = _symmetry_score(freqs, mag_before, mirror_band_hz, center_hz)
    symmetry_after = _symmetry_score(freqs, mag_after, mirror_band_hz, center_hz)
    mirror_energy_before = _mirror_band_energy(freqs, mag_before, mirror_band_hz)
    mirror_energy_after = _mirror_band_energy(freqs, mag_after, mirror_band_hz)
    stripe_before = _stripe_score(freqs, mag_before, mirror_band_hz)
    stripe_after = _stripe_score(freqs, mag_after, mirror_band_hz)

    return MirrorReductionMetrics(
        symmetry_score_before=symmetry_before,
        symmetry_score_after=symmetry_after,
        symmetry_reduction_ratio=_relative_reduction(symmetry_before, symmetry_after),
        mirror_band_energy_before=mirror_energy_before,
        mirror_band_energy_after=mirror_energy_after,
        mirror_band_energy_reduction_ratio=_relative_reduction(
            mirror_energy_before,
            mirror_energy_after,
        ),
        stripe_score_before=stripe_before,
        stripe_score_after=stripe_after,
        stripe_reduction_ratio=_relative_reduction(stripe_before, stripe_after),
    )


def evaluate_mirror_reduction_dataset(
    samples: list[tuple[str, np.ndarray, np.ndarray]],
    sample_rate: int,
    mirror_band_hz: tuple[float, float] = DEFAULT_MIRROR_BAND_HZ,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    mirror_center_hz: float | None = None,
    target_reduction_ratio: float = 0.70,
) -> MirrorDatasetEvaluationResult:
    """Evaluate mirror/aliasing reduction for a dataset of paired signals.

    Args:
        samples: List of (sample_id, before_signal, after_signal).
        sample_rate: Sample rate in Hz.
        mirror_band_hz: Frequency band used for mirror analysis.
        n_fft: STFT FFT size.
        hop_length: STFT hop size.
        mirror_center_hz: Symmetry center frequency in Hz.
        target_reduction_ratio: Target symmetry reduction threshold.

    Returns:
        MirrorDatasetEvaluationResult with per-sample and aggregate values.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        Dataset-level acceptance checks mirror reduction robustness and
        pass-rate against a fixed target threshold.
    """
    if len(samples) == 0:
        raise ValueError("samples cannot be empty.")
    _validate_ratio(target_reduction_ratio, "target_reduction_ratio")

    sample_results: list[MirrorSampleEvaluationResult] = []
    for sample_id, before_signal, after_signal in samples:
        metrics = evaluate_mirror_reduction(
            before_signal=before_signal,
            after_signal=after_signal,
            sample_rate=sample_rate,
            mirror_band_hz=mirror_band_hz,
            n_fft=n_fft,
            hop_length=hop_length,
            mirror_center_hz=mirror_center_hz,
        )
        sample_results.append(
            MirrorSampleEvaluationResult(sample_id=sample_id, metrics=metrics)
        )

    metric_matrix = np.array(
        [
            [
                sample.metrics.symmetry_score_before,
                sample.metrics.symmetry_score_after,
                sample.metrics.symmetry_reduction_ratio,
                sample.metrics.mirror_band_energy_before,
                sample.metrics.mirror_band_energy_after,
                sample.metrics.mirror_band_energy_reduction_ratio,
                sample.metrics.stripe_score_before,
                sample.metrics.stripe_score_after,
                sample.metrics.stripe_reduction_ratio,
            ]
            for sample in sample_results
        ],
        dtype=np.float64,
    )
    mean_values = np.mean(metric_matrix, axis=0)
    pass_rate = float(
        np.mean(
            [
                sample.metrics.symmetry_reduction_ratio >= target_reduction_ratio
                for sample in sample_results
            ]
        )
    )

    mean_metrics = MirrorReductionMetrics(
        symmetry_score_before=float(mean_values[0]),
        symmetry_score_after=float(mean_values[1]),
        symmetry_reduction_ratio=float(mean_values[2]),
        mirror_band_energy_before=float(mean_values[3]),
        mirror_band_energy_after=float(mean_values[4]),
        mirror_band_energy_reduction_ratio=float(mean_values[5]),
        stripe_score_before=float(mean_values[6]),
        stripe_score_after=float(mean_values[7]),
        stripe_reduction_ratio=float(mean_values[8]),
    )
    return MirrorDatasetEvaluationResult(
        samples=tuple(sample_results),
        mean_metrics=mean_metrics,
        target_reduction_ratio=target_reduction_ratio,
        symmetry_target_pass_rate=pass_rate,
    )


def evaluate_metric_listening_correlation(
    metric_values: np.ndarray,
    listening_scores: np.ndarray,
) -> ListeningCorrelationMetrics:
    """Evaluate objective-to-subjective correlation.

    Args:
        metric_values: Objective metric values (e.g. symmetry reduction ratios).
        listening_scores: Listening-test scores aligned with metric values.

    Returns:
        ListeningCorrelationMetrics with Pearson and Spearman correlations.

    Raises:
        ValueError: If inputs are invalid.

    Physical Basis:
        A useful objective metric should correlate with perceived quality
        improvements in listening tests.
    """
    metric_vector = _validate_vector(metric_values, "metric_values")
    listening_vector = _validate_vector(listening_scores, "listening_scores")
    if metric_vector.shape != listening_vector.shape:
        raise ValueError(
            "metric_values and listening_scores must have the same shape. "
            f"Got {metric_vector.shape} and {listening_vector.shape}."
        )
    if metric_vector.size < 2:
        raise ValueError("At least two paired samples are required.")

    pearson = float(np.corrcoef(metric_vector, listening_vector)[0, 1])
    spearman = float(sp_stats.spearmanr(metric_vector, listening_vector).statistic)
    return ListeningCorrelationMetrics(
        pearson_correlation=_finite_or_zero(pearson),
        spearman_correlation=_finite_or_zero(spearman),
        num_pairs=int(metric_vector.size),
    )


def mirror_dataset_result_to_payload(
    result: MirrorDatasetEvaluationResult,
) -> dict[str, object]:
    """Convert mirror dataset evaluation to JSON-serializable payload.

    Args:
        result: Dataset-level mirror evaluation result.

    Returns:
        Dictionary with aggregate and per-sample mirror metrics.

    Physical Basis:
        Structured payloads support reproducible acceptance checks and
        downstream regression tracking.
    """
    return {
        "summary": asdict(result.mean_metrics)
        | {
            "target_reduction_ratio": result.target_reduction_ratio,
            "symmetry_target_pass_rate": result.symmetry_target_pass_rate,
            "num_samples": len(result.samples),
        },
        "samples": [
            {"sample_id": sample.sample_id, **asdict(sample.metrics)}
            for sample in result.samples
        ],
    }


def export_mirror_reduction_visualization(
    before_signal: np.ndarray,
    after_signal: np.ndarray,
    sample_rate: int,
    output_path: Path,
    mirror_band_hz: tuple[float, float] = DEFAULT_MIRROR_BAND_HZ,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    mirror_center_hz: float | None = None,
    title: str = "Mirror/Aliasing Before vs After",
) -> MirrorVisualizationArtifacts:
    """Export before/after mirror comparison figure.

    Args:
        before_signal: Signal before suppression.
        after_signal: Signal after suppression.
        sample_rate: Sample rate in Hz.
        output_path: Output PNG path.
        mirror_band_hz: Frequency band used for mirror analysis.
        n_fft: STFT FFT size.
        hop_length: STFT hop size.
        mirror_center_hz: Symmetry center frequency in Hz.
        title: Plot title.

    Returns:
        MirrorVisualizationArtifacts with exported path.

    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If plot export fails.

    Physical Basis:
        Inspecting mirror-band spectrograms before/after provides direct
        evidence that symmetric artifacts are removed.
    """

    metrics = evaluate_mirror_reduction(
        before_signal=before_signal,
        after_signal=after_signal,
        sample_rate=sample_rate,
        mirror_band_hz=mirror_band_hz,
        n_fft=n_fft,
        hop_length=hop_length,
        mirror_center_hz=mirror_center_hz,
    )
    center_hz = _resolve_mirror_center_hz(sample_rate, mirror_center_hz)

    freqs, times, stft_before = _stft(before_signal, sample_rate, n_fft, hop_length)
    _, _, stft_after = _stft(after_signal, sample_rate, n_fft, hop_length)
    mag_before = np.abs(stft_before)
    mag_after = np.abs(stft_after)

    band_mask = (freqs >= mirror_band_hz[0]) & (freqs <= mirror_band_hz[1])
    band_freqs = freqs[band_mask]
    band_before_db = _to_db(mag_before[band_mask])
    band_after_db = _to_db(mag_after[band_mask])
    diff_db = band_after_db - band_before_db

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        _draw_spectrogram(
            axes[0, 0],
            times,
            band_freqs,
            band_before_db,
            "Before (dB)",
            mirror_band_hz,
            center_hz,
        )
        _draw_spectrogram(
            axes[0, 1],
            times,
            band_freqs,
            band_after_db,
            "After (dB)",
            mirror_band_hz,
            center_hz,
        )
        _draw_spectrogram(
            axes[1, 0],
            times,
            band_freqs,
            diff_db,
            "After - Before (dB)",
            mirror_band_hz,
            center_hz,
            cmap="coolwarm",
        )

        axes[1, 1].axis("off")
        summary = (
            f"Symmetry reduction: {metrics.symmetry_reduction_ratio * 100.0:.2f}%\n"
            f"Mirror-band energy reduction: "
            f"{metrics.mirror_band_energy_reduction_ratio * 100.0:.2f}%\n"
            f"Stripe reduction: {metrics.stripe_reduction_ratio * 100.0:.2f}%"
        )
        axes[1, 1].text(0.02, 0.85, summary, fontsize=11, va="top")

        figure.suptitle(title)
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
    except Exception as exc:
        raise RuntimeError(f"Failed to export mirror visualization: {exc}") from exc

    return MirrorVisualizationArtifacts(plot_path=output_path)


def _symmetry_score(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    mirror_band_hz: tuple[float, float],
    mirror_center_hz: float,
) -> float:
    band_indices = np.where(
        (freqs >= mirror_band_hz[0]) & (freqs <= mirror_band_hz[1])
    )[0]
    if band_indices.size == 0:
        return 0.0

    pair_min_energies: list[np.ndarray] = []
    for index in band_indices:
        mirrored_hz = 2.0 * mirror_center_hz - freqs[index]
        if mirrored_hz < 0.0 or mirrored_hz > freqs[-1]:
            continue
        mirrored_index = int(np.argmin(np.abs(freqs - mirrored_hz)))
        if mirrored_index == index:
            continue

        pair_min = np.minimum(magnitude[index], magnitude[mirrored_index])
        pair_min_energies.append(pair_min)

    if len(pair_min_energies) == 0:
        return 0.0

    min_energy = np.concatenate(pair_min_energies)
    return float(np.mean(min_energy))


def _mirror_band_energy(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    mirror_band_hz: tuple[float, float],
) -> float:
    band_mask = (freqs >= mirror_band_hz[0]) & (freqs <= mirror_band_hz[1])
    if not np.any(band_mask):
        return 0.0
    return float(np.mean(np.square(magnitude[band_mask])))


def _stripe_score(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    mirror_band_hz: tuple[float, float],
) -> float:
    band_mask = (freqs >= mirror_band_hz[0]) & (freqs <= mirror_band_hz[1])
    if not np.any(band_mask):
        return 0.0

    band_mag = magnitude[band_mask]
    if band_mag.shape[1] <= 1:
        return 0.0

    temporal_diff = np.diff(band_mag, axis=1)
    return float(np.mean(np.abs(temporal_diff)) / (np.mean(band_mag) + EPSILON))


def _draw_spectrogram(
    axis: plt.Axes,
    times: np.ndarray,
    freqs: np.ndarray,
    data_db: np.ndarray,
    subtitle: str,
    mirror_band_hz: tuple[float, float],
    mirror_center_hz: float,
    cmap: str = "magma",
) -> None:
    mesh = axis.pcolormesh(times, freqs, data_db, shading="auto", cmap=cmap)
    axis.set_title(subtitle)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Frequency [Hz]")
    axis.axhline(mirror_band_hz[0], color="white", linestyle="--", linewidth=0.8)
    axis.axhline(mirror_band_hz[1], color="white", linestyle="--", linewidth=0.8)
    axis.axhline(mirror_center_hz, color="cyan", linestyle=":", linewidth=1.0)
    plt.colorbar(mesh, ax=axis, pad=0.01)


def _to_db(magnitude: np.ndarray) -> np.ndarray:
    db_values = 20.0 * np.log10(np.maximum(magnitude, EPSILON))
    return np.asarray(db_values, dtype=np.float64)


def _relative_reduction(before: float, after: float) -> float:
    if before <= EPSILON:
        return 0.0
    return float((before - after) / before)


def _resolve_mirror_center_hz(
    sample_rate: int, mirror_center_hz: float | None
) -> float:
    if mirror_center_hz is not None:
        return float(mirror_center_hz)
    return sample_rate / 4.0


def _stft(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    noverlap = n_fft - hop_length
    freqs, times, stft = sp_signal.stft(
        signal,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=noverlap,
        window=DEFAULT_WINDOW,
        boundary="zeros",
        padded=True,
    )
    return (
        np.asarray(freqs, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        np.asarray(stft, dtype=np.complex128),
    )


def _validate_signal_pair(before_signal: np.ndarray, after_signal: np.ndarray) -> None:
    if before_signal.ndim != 1 or after_signal.ndim != 1:
        raise ValueError("before_signal and after_signal must be 1D arrays.")
    if before_signal.size == 0 or after_signal.size == 0:
        raise ValueError("before_signal and after_signal cannot be empty.")
    if before_signal.shape != after_signal.shape:
        raise ValueError(
            "before_signal and after_signal must have identical shapes. "
            f"Got {before_signal.shape} and {after_signal.shape}."
        )


def _validate_sample_rate(sample_rate: int) -> None:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")


def _validate_stft_params(n_fft: int, hop_length: int) -> None:
    if n_fft <= 0:
        raise ValueError(f"n_fft must be positive, got {n_fft}.")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}.")
    if hop_length >= n_fft:
        raise ValueError("hop_length must be smaller than n_fft.")


def _validate_mirror_band(
    mirror_band_hz: tuple[float, float],
    sample_rate: int,
) -> None:
    lower, upper = mirror_band_hz
    if lower <= 0.0:
        raise ValueError(f"mirror_band lower must be positive, got {lower}.")
    if upper <= lower:
        raise ValueError("mirror_band upper must be greater than lower.")
    nyquist = sample_rate / 2.0
    if upper > nyquist:
        raise ValueError(
            f"mirror_band upper must be <= Nyquist ({nyquist}), got {upper}."
        )


def _validate_mirror_center(mirror_center_hz: float, sample_rate: int) -> None:
    if mirror_center_hz <= 0.0:
        raise ValueError(f"mirror_center_hz must be positive, got {mirror_center_hz}.")
    nyquist = sample_rate / 2.0
    if mirror_center_hz >= nyquist:
        raise ValueError(
            f"mirror_center_hz must be lower than Nyquist ({nyquist}), "
            f"got {mirror_center_hz}."
        )


def _validate_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {vector.ndim}D.")
    if vector.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite.")
    return vector


def _validate_ratio(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")


def _finite_or_zero(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return value
