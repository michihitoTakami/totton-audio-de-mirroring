"""Hi-res audio corpus loader for genuine high-sample-rate teacher data.

Physical Basis:
    The synthetic ``raw`` teacher path fabricates >22.05kHz content from
    analytic signals. Real instruments and voices carry structured energy
    above 22.05kHz that synthetic generators cannot reproduce. Using genuine
    hi-res recordings (>= the Stage 1 target rate) as the teacher lets the
    network learn a physically faithful high-band suppression target instead
    of mimicking a hand-designed DSP rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as sp_signal

DEFAULT_EXTENSIONS: tuple[str, ...] = (".wav", ".flac", ".aiff", ".aif")
MonoMode = str


@dataclass(frozen=True)
class HiResCorpusConfig:
    """Configuration for the hi-res teacher corpus.

    Args:
        root: Directory containing hi-res audio files.
        min_sample_rate: Minimum native sample rate to accept (Hz).
        min_hf_energy_ratio: Minimum fraction of energy above ``split_hz``
            required for a loaded segment to count as genuinely hi-res.
        split_hz: Frequency separating the "lost" high band from the audible
            band when measuring high-frequency energy.
        mono_mode: How to fold multi-channel files to mono ("downmix" or "left").
        extensions: Accepted file extensions (lowercase, dot-prefixed).

    Physical Basis:
        ``min_sample_rate`` guarantees the teacher actually contains content
        above 22.05kHz; ``min_hf_energy_ratio`` rejects files that are nominally
        hi-res but were upsampled from a band-limited master and thus carry no
        real ultrasonic information.
    """

    root: Path
    min_sample_rate: int = 88_200
    min_hf_energy_ratio: float = 1.0e-6
    split_hz: float = 22_050.0
    mono_mode: MonoMode = "downmix"
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    max_load_attempts: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise ValueError("root must be a pathlib.Path.")
        if self.min_sample_rate <= 0:
            raise ValueError("min_sample_rate must be positive.")
        if self.min_hf_energy_ratio < 0.0:
            raise ValueError("min_hf_energy_ratio must be non-negative.")
        if self.split_hz <= 0.0:
            raise ValueError("split_hz must be positive.")
        if self.mono_mode not in {"downmix", "left"}:
            raise ValueError("mono_mode must be 'downmix' or 'left'.")
        if not self.extensions:
            raise ValueError("extensions must be non-empty.")
        if self.max_load_attempts <= 0:
            raise ValueError("max_load_attempts must be positive.")


class HiResCorpus:
    """Indexable corpus of hi-res recordings serving target-rate teacher chunks.

    Args:
        config: Corpus configuration.
        target_sample_rate: Stage 1 target sample rate (e.g., 88200 Hz).
        source_duration_sec: Full teacher segment duration in seconds.

    Raises:
        FileNotFoundError: If the corpus directory has no usable files.

    Physical Basis:
        Each item is one full-length teacher signal at the target rate; the
        dataset derives the 44.1kHz input by downsampling the same chunk,
        keeping input and target timelines aligned.
    """

    def __init__(
        self,
        config: HiResCorpusConfig,
        *,
        target_sample_rate: int,
        source_duration_sec: float,
    ) -> None:
        if target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive.")
        if source_duration_sec <= 0.0:
            raise ValueError("source_duration_sec must be positive.")
        if config.min_sample_rate < target_sample_rate:
            raise ValueError(
                "min_sample_rate must be >= target_sample_rate to preserve "
                "genuine high-band content."
            )

        self._config = config
        self._target_sample_rate = target_sample_rate
        self._source_duration_sec = source_duration_sec
        self._files = discover_hires_files(config.root, config.extensions)
        self._files = [path for path in self._files if self._has_min_sample_rate(path)]
        if not self._files:
            raise FileNotFoundError(
                f"No usable hi-res files (>= {config.min_sample_rate} Hz) under "
                f"{config.root}."
            )

    @property
    def num_files(self) -> int:
        """Return the number of usable hi-res files."""
        return len(self._files)

    @property
    def files(self) -> tuple[Path, ...]:
        """Return the discovered usable hi-res file paths."""
        return tuple(self._files)

    def load_teacher_source(self, index: int, rng: np.random.Generator) -> np.ndarray:
        """Load one teacher segment at the target sample rate.

        Args:
            index: Sample index (mapped to a file deterministically).
            rng: RNG used to choose the segment start position.

        Returns:
            Mono teacher signal at the target rate, length
            ``round(source_duration_sec * target_sample_rate)``.

        Raises:
            ValueError: If the loaded segment lacks genuine high-frequency
                energy above ``split_hz``.

        Physical Basis:
            A random in-file offset increases content diversity while the
            HF-energy check rejects segments (e.g., silence) that would teach
            the network nothing about real ultrasonic structure.
        """
        if index < 0:
            raise ValueError("index must be non-negative.")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator.")

        target_len = int(round(self._source_duration_sec * self._target_sample_rate))
        last_ratio = 0.0
        last_name = ""
        for attempt in range(self._config.max_load_attempts):
            path = self._files[(index + attempt) % len(self._files)]
            mono, file_sr = _read_mono_segment(
                path,
                duration_sec=self._source_duration_sec,
                mono_mode=self._config.mono_mode,
                rng=rng,
            )
            ratio = high_frequency_energy_ratio(
                mono, file_sr, split_hz=self._config.split_hz
            )
            if ratio >= self._config.min_hf_energy_ratio:
                resampled = resample_signal(mono, file_sr, self._target_sample_rate)
                return _fit_length(resampled, target_len)
            last_ratio = ratio
            last_name = path.name

        raise ValueError(
            f"No segment with sufficient high-frequency energy after "
            f"{self._config.max_load_attempts} attempts (last {last_name}: "
            f"{last_ratio:.3e} < {self._config.min_hf_energy_ratio:.3e})."
        )

    def _has_min_sample_rate(self, path: Path) -> bool:
        try:
            info = sf.info(str(path))
        except Exception:
            return False
        return int(info.samplerate) >= self._config.min_sample_rate


def discover_hires_files(root: Path, extensions: Sequence[str]) -> list[Path]:
    """Recursively discover candidate audio files under a directory.

    Args:
        root: Directory to search.
        extensions: Accepted lowercase, dot-prefixed extensions.

    Returns:
        Sorted list of matching file paths.

    Raises:
        FileNotFoundError: If ``root`` does not exist or is not a directory.

    Physical Basis:
        Deterministic ordering keeps index-to-file mapping reproducible
        across runs.
    """
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Hi-res corpus directory not found: {root}")
    allowed = {ext.lower() for ext in extensions}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )


def high_frequency_energy_ratio(
    signal: np.ndarray, sample_rate: int, *, split_hz: float
) -> float:
    """Compute the fraction of energy above ``split_hz``.

    Args:
        signal: Mono signal.
        sample_rate: Sample rate in Hz.
        split_hz: Split frequency in Hz.

    Returns:
        Energy ratio in [0, 1]; 0 if the signal is silent.

    Physical Basis:
        Genuine hi-res masters retain measurable energy above 22.05kHz.
        Sources upsampled from a band-limited master have a near-zero ratio
        and must be rejected as teachers.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1D.")
    if signal.size == 0:
        raise ValueError("signal must be non-empty.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if split_hz <= 0.0 or split_hz >= sample_rate / 2.0:
        raise ValueError("split_hz must be in (0, Nyquist).")

    spectrum = np.fft.rfft(np.asarray(signal, dtype=np.float64))
    freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / sample_rate)
    power = np.abs(spectrum) ** 2
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    high = float(np.sum(power[freqs >= split_hz]))
    return high / total


def resample_signal(signal: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono signal between arbitrary integer sample rates.

    Args:
        signal: Mono input signal.
        source_sr: Source sample rate in Hz.
        target_sr: Target sample rate in Hz.

    Returns:
        Resampled signal as float64.

    Physical Basis:
        Polyphase rational resampling provides high-quality anti-aliased
        conversion (e.g., 96kHz -> 88.2kHz) needed to align hi-res masters
        with the Stage 1 target rate.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1D.")
    if signal.size == 0:
        raise ValueError("signal must be non-empty.")
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive.")

    data = np.asarray(signal, dtype=np.float64)
    if source_sr == target_sr:
        return data
    divisor = gcd(source_sr, target_sr)
    up = target_sr // divisor
    down = source_sr // divisor
    resampled = sp_signal.resample_poly(data, up=up, down=down, window=("kaiser", 8.6))
    return np.asarray(resampled, dtype=np.float64)


def _read_mono_segment(
    path: Path,
    *,
    duration_sec: float,
    mono_mode: MonoMode,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Read a mono segment of ``duration_sec`` from an audio file.

    Args:
        path: Audio file path.
        duration_sec: Desired segment duration in seconds.
        mono_mode: Multi-channel fold mode ("downmix" or "left").
        rng: RNG for choosing the random start frame.

    Returns:
        Tuple of ``(mono_signal, file_sample_rate)`` at the file's native rate.

    Raises:
        RuntimeError: If the file cannot be read.

    Physical Basis:
        Reading at native rate preserves ultrasonic content; resampling to the
        target rate happens afterwards so the HF-energy check sees real data.
    """
    try:
        info = sf.info(str(path))
        file_sr = int(info.samplerate)
        total_frames = int(info.frames)
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio info: {path}: {exc}") from exc

    want_frames = int(round(duration_sec * file_sr))
    if total_frames <= 0:
        raise RuntimeError(f"Audio file is empty: {path}")

    if total_frames > want_frames:
        start = int(rng.integers(0, total_frames - want_frames + 1))
    else:
        start = 0
    try:
        block, _ = sf.read(
            str(path),
            start=start,
            frames=want_frames,
            dtype="float64",
            always_2d=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio data: {path}: {exc}") from exc

    mono = _fold_to_mono(np.asarray(block, dtype=np.float64), mono_mode)
    mono = _fit_length(mono, want_frames)
    return mono, file_sr


def _fold_to_mono(block: np.ndarray, mono_mode: MonoMode) -> np.ndarray:
    """Fold a (frames, channels) block to a 1D mono signal."""
    if block.ndim != 2:
        raise ValueError("block must be 2D (frames, channels).")
    if block.shape[1] == 1:
        return block[:, 0]
    if mono_mode == "left":
        return block[:, 0]
    return np.asarray(np.mean(block, axis=1), dtype=block.dtype)


def _fit_length(signal: np.ndarray, length: int) -> np.ndarray:
    """Trim or tile a signal to an exact length.

    Args:
        signal: 1D input signal.
        length: Desired output length in samples.

    Returns:
        Signal of exactly ``length`` samples (tiled if shorter).

    Physical Basis:
        Tiling short files preserves spectral content (including the high
        band) better than zero-padding, which would inject spectral edges.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1D.")
    if length <= 0:
        raise ValueError("length must be positive.")
    if signal.size == 0:
        raise ValueError("signal must be non-empty.")

    if signal.shape[0] == length:
        return signal
    if signal.shape[0] > length:
        return signal[:length]
    repeats = int(np.ceil(length / signal.shape[0]))
    tiled = np.tile(signal, repeats)
    return tiled[:length]
