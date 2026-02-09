"""Tests for chunked Hann-window overlap-add processing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from totton_audio_de_mirroring.inference.chunk_processor import (
    ChunkProcessingConfig,
    HannOverlapAddStreamer,
    iterate_chunk_frames,
)


def test_iterate_chunk_frames_covers_full_signal_with_overlap() -> None:
    """Chunk frame iterator should cover all samples with deterministic overlap."""
    signal = np.arange(24, dtype=np.float64)
    frames = tuple(iterate_chunk_frames(signal, chunk_samples=10, overlap_samples=5))

    assert len(frames) == 4
    assert frames[0].start == 0
    assert frames[0].end == 10
    assert frames[1].start == 5
    assert frames[1].end == 15
    assert frames[-1].end == signal.shape[0]
    assert np.allclose(frames[0].samples[-5:], frames[1].samples[:5])


def test_hann_ola_streamer_reconstructs_identity_chunk_stream() -> None:
    """Identity chunk stream should reconstruct the original waveform."""
    signal = np.random.default_rng(42).standard_normal(5000).astype(np.float64)
    frames = tuple(iterate_chunk_frames(signal, chunk_samples=512, overlap_samples=256))

    streamer = HannOverlapAddStreamer(
        chunk_samples=512,
        overlap_samples=256,
        window="hann",
    )
    pieces: list[np.ndarray] = []
    for frame in frames:
        chunk_piece = streamer.process_chunk(frame.samples)
        if chunk_piece.size > 0:
            pieces.append(chunk_piece)
    tail = streamer.finalize()
    if tail.size > 0:
        pieces.append(tail)

    reconstructed = np.concatenate(pieces)
    assert reconstructed.shape == signal.shape
    assert np.allclose(reconstructed, signal, atol=1.0e-6)


@pytest.mark.slow
def test_hann_ola_streamer_handles_ten_minute_streaming_identity() -> None:
    """Ten-minute synthetic stream should be processable chunk-by-chunk."""
    sample_rate = 44_100
    total_samples = sample_rate * 60 * 10
    config = ChunkProcessingConfig(
        sample_rate=sample_rate,
        chunk_duration_sec=0.25,
        overlap_ratio=0.5,
        window="hann",
    )

    streamer = HannOverlapAddStreamer(
        chunk_samples=config.chunk_samples,
        overlap_samples=config.overlap_samples,
        window=config.window,
    )

    emitted = 0
    start = 0
    while start < total_samples:
        end = min(start + config.chunk_samples, total_samples)
        chunk_len = end - start
        indices = start + np.arange(chunk_len, dtype=np.float64)
        chunk = 0.2 * np.sin(2.0 * np.pi * 997.0 * indices / sample_rate)
        piece = streamer.process_chunk(chunk)
        emitted += int(piece.shape[0])
        if end >= total_samples:
            break
        start += config.hop_samples

    emitted += int(streamer.finalize().shape[0])
    assert emitted == total_samples
    expected_chunks = (
        math.ceil(max(total_samples - config.chunk_samples, 0) / config.hop_samples) + 1
    )
    assert expected_chunks > 1
