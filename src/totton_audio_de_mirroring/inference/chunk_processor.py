"""Chunk processing utilities with Hann-windowed 50% overlap-add."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChunkProcessingConfig:
    """Chunking configuration for long-form inference.

    Args:
        sample_rate: Input sample rate in Hz.
        chunk_duration_sec: Chunk length in seconds.
        overlap_ratio: Overlap ratio between chunks (fixed to 0.5).
        window: Window type used in overlap-add ("hann" only).

    Physical Basis:
        50% overlapped Hann windows suppress boundary discontinuities while
        keeping reconstruction stable under chunked long-audio inference.
    """

    sample_rate: int
    chunk_duration_sec: float = 0.25
    overlap_ratio: float = 0.5
    window: str = "hann"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.chunk_duration_sec <= 0.0:
            raise ValueError("chunk_duration_sec must be positive.")
        if not np.isclose(self.overlap_ratio, 0.5, atol=1.0e-9):
            raise ValueError("overlap_ratio must be exactly 0.5 for Issue #33.")
        window = self.window.strip().lower()
        if window != "hann":
            raise ValueError("window must be 'hann'.")
        object.__setattr__(self, "window", window)

        chunk_samples = int(round(self.chunk_duration_sec * self.sample_rate))
        if chunk_samples < 2:
            raise ValueError("chunk_duration_sec produced too few samples.")
        overlap_samples = int(round(chunk_samples * self.overlap_ratio))
        if overlap_samples <= 0 or overlap_samples >= chunk_samples:
            raise ValueError("overlap_samples must be in range (0, chunk_samples).")

    @property
    def chunk_samples(self) -> int:
        """Chunk length in samples."""
        return int(round(self.chunk_duration_sec * self.sample_rate))

    @property
    def overlap_samples(self) -> int:
        """Overlap length in samples."""
        return int(round(self.chunk_samples * self.overlap_ratio))

    @property
    def hop_samples(self) -> int:
        """Hop length in samples."""
        return self.chunk_samples - self.overlap_samples


@dataclass(frozen=True)
class ChunkFrame:
    """One chunk frame with source-domain span information."""

    start: int
    end: int
    samples: np.ndarray


def iterate_chunk_frames(
    signal: np.ndarray,
    *,
    chunk_samples: int,
    overlap_samples: int,
) -> Iterator[ChunkFrame]:
    """Split one signal into overlapped frames.

    Args:
        signal: Mono input signal.
        chunk_samples: Frame size in samples.
        overlap_samples: Overlap size in samples.

    Yields:
        Ordered chunk frames covering the full signal.

    Physical Basis:
        Deterministic frame partitioning is required so overlap-add combines
        neighboring windows exactly once without gaps.
    """
    _validate_signal(signal)
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive.")
    if overlap_samples <= 0:
        raise ValueError("overlap_samples must be positive.")
    if overlap_samples >= chunk_samples:
        raise ValueError("overlap_samples must be smaller than chunk_samples.")

    if signal.shape[0] <= chunk_samples:
        yield ChunkFrame(
            start=0,
            end=signal.shape[0],
            samples=np.asarray(signal, dtype=np.float64),
        )
        return

    hop_samples = chunk_samples - overlap_samples
    start = 0
    while start < signal.shape[0]:
        end = min(start + chunk_samples, signal.shape[0])
        yield ChunkFrame(
            start=start,
            end=end,
            samples=np.asarray(signal[start:end], dtype=np.float64),
        )
        if end >= signal.shape[0]:
            break
        start += hop_samples


class HannOverlapAddStreamer:
    """Streaming Hann-window overlap-add combiner.

    Args:
        chunk_samples: Expected maximum chunk length.
        overlap_samples: Overlap length in samples.
        window: Window type ("hann" only).

    Physical Basis:
        Weighted overlap-add with Hann taper reduces boundary artifacts, and
        per-sample normalization preserves amplitude over long sequences.
    """

    def __init__(
        self, *, chunk_samples: int, overlap_samples: int, window: str = "hann"
    ) -> None:
        if chunk_samples <= 1:
            raise ValueError("chunk_samples must be greater than 1.")
        if overlap_samples <= 0:
            raise ValueError("overlap_samples must be positive.")
        if overlap_samples >= chunk_samples:
            raise ValueError("overlap_samples must be smaller than chunk_samples.")
        window_name = window.strip().lower()
        if window_name != "hann":
            raise ValueError("window must be 'hann'.")

        self._chunk_samples = int(chunk_samples)
        self._overlap_samples = int(overlap_samples)
        self._window = _hann_window_nonzero_edges(self._chunk_samples)
        self._eps = 1.0e-12

        self._pending_weighted = np.zeros(0, dtype=np.float64)
        self._pending_weight = np.zeros(0, dtype=np.float64)
        self._pending_raw = np.zeros(0, dtype=np.float64)
        self._pending_count = np.zeros(0, dtype=np.float64)

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process one chunk and return finalized output region."""
        signal = np.asarray(chunk, dtype=np.float64)
        _validate_signal(signal)
        if signal.shape[0] > self._chunk_samples:
            raise ValueError("chunk length exceeds configured chunk_samples.")

        chunk_len = signal.shape[0]
        weights = self._window[:chunk_len]
        weighted = signal * weights
        counts = np.ones(chunk_len, dtype=np.float64)

        if self._pending_weighted.size == 0:
            self._pending_weighted = weighted
            self._pending_weight = weights
            self._pending_raw = signal
            self._pending_count = counts
            return np.zeros(0, dtype=np.float64)

        overlap = min(
            self._overlap_samples,
            self._pending_weighted.shape[0],
            weighted.shape[0],
        )
        emit_non_overlap = self._pending_weighted.shape[0] - overlap

        pieces: list[np.ndarray] = []
        if emit_non_overlap > 0:
            pieces.append(
                _safe_normalize(
                    weighted_sum=self._pending_weighted[:emit_non_overlap],
                    weight_sum=self._pending_weight[:emit_non_overlap],
                    raw_sum=self._pending_raw[:emit_non_overlap],
                    raw_count=self._pending_count[:emit_non_overlap],
                    eps=self._eps,
                )
            )

        if overlap > 0:
            left_slice = slice(emit_non_overlap, self._pending_weighted.shape[0])
            overlap_piece = _safe_normalize(
                weighted_sum=self._pending_weighted[left_slice] + weighted[:overlap],
                weight_sum=self._pending_weight[left_slice] + weights[:overlap],
                raw_sum=self._pending_raw[left_slice] + signal[:overlap],
                raw_count=self._pending_count[left_slice] + counts[:overlap],
                eps=self._eps,
            )
            pieces.append(overlap_piece)

        self._pending_weighted = weighted[overlap:]
        self._pending_weight = weights[overlap:]
        self._pending_raw = signal[overlap:]
        self._pending_count = counts[overlap:]

        if not pieces:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(pieces)

    def finalize(self) -> np.ndarray:
        """Flush remaining tail samples after the last chunk."""
        if self._pending_weighted.size == 0:
            return np.zeros(0, dtype=np.float64)

        tail = _safe_normalize(
            weighted_sum=self._pending_weighted,
            weight_sum=self._pending_weight,
            raw_sum=self._pending_raw,
            raw_count=self._pending_count,
            eps=self._eps,
        )
        self._pending_weighted = np.zeros(0, dtype=np.float64)
        self._pending_weight = np.zeros(0, dtype=np.float64)
        self._pending_raw = np.zeros(0, dtype=np.float64)
        self._pending_count = np.zeros(0, dtype=np.float64)
        return tail


def _safe_normalize(
    *,
    weighted_sum: np.ndarray,
    weight_sum: np.ndarray,
    raw_sum: np.ndarray,
    raw_count: np.ndarray,
    eps: float,
) -> np.ndarray:
    normalized = np.divide(
        weighted_sum,
        np.maximum(weight_sum, eps),
        dtype=np.float64,
    )
    fallback = np.divide(
        raw_sum,
        np.maximum(raw_count, 1.0),
        dtype=np.float64,
    )
    return np.where(weight_sum > eps, normalized, fallback)


def _hann_window_nonzero_edges(length: int) -> np.ndarray:
    if length <= 1:
        raise ValueError("length must be greater than 1.")
    return np.asarray(np.hanning(length + 2)[1:-1], dtype=np.float64)


def _validate_signal(signal: np.ndarray) -> None:
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got {signal.ndim}D.")
    if signal.size == 0:
        raise ValueError("signal cannot be empty.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must contain only finite values.")
