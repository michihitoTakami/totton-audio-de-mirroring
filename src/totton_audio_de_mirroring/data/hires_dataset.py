"""Dataset that uses genuine hi-res recordings as the Stage 1 raw teacher."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from totton_audio_de_mirroring.data.dataset import (
    assemble_stage1_sample,
    build_raw_teacher_chunks_from_source,
)
from totton_audio_de_mirroring.data.degradation import DegradationProfileManager
from totton_audio_de_mirroring.data.hires_corpus import HiResCorpus, HiResCorpusConfig
from totton_audio_de_mirroring.data.pipeline_config import DataPipelineConfig
from totton_audio_de_mirroring.models.band_split import (
    BandSplitConfig,
    BandSplitProcessor,
)

RAW_TEACHER_TYPES = ("raw_88k2", "raw_176k4")
HIRES_SIGNAL_TAG = "hires"


class HiResTeacherDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """Stage 1 dataset backed by genuine hi-res recordings.

    Args:
        config: Data pipeline configuration (must use a ``raw`` teacher type).
        corpus_config: Hi-res corpus configuration.

    Raises:
        ValueError: If the teacher type is not a raw (native) policy.

    Physical Basis:
        The raw teacher policy expects a teacher signal carrying real
        >22.05kHz energy. This dataset supplies that signal from hi-res
        recordings and reuses the exact degrade -> band-split -> project
        pipeline so batches are interchangeable with the synthetic dataset.
    """

    def __init__(
        self,
        config: DataPipelineConfig,
        corpus_config: HiResCorpusConfig,
    ) -> None:
        if not isinstance(config, DataPipelineConfig):
            raise ValueError("config must be a DataPipelineConfig.")
        if config.teacher_type not in RAW_TEACHER_TYPES:
            raise ValueError(
                "HiResTeacherDataset requires a raw teacher type "
                f"{RAW_TEACHER_TYPES}, got {config.teacher_type!r}."
            )

        self._config = config
        self._corpus = HiResCorpus(
            corpus_config,
            target_sample_rate=config.target_sample_rate,
            source_duration_sec=config.source_duration_sec,
        )
        self._degradation = DegradationProfileManager(config.degradation)
        self._band_split = BandSplitProcessor(
            _band_split_at_rate(config.band_split, config.target_sample_rate)
        )
        self._base_seed = (
            config.seed
            if config.seed is not None
            else int(np.random.SeedSequence().generate_state(1)[0])
        )

    def __len__(self) -> int:
        """Return dataset length (number of sampled items)."""
        return self._config.num_samples

    @property
    def corpus(self) -> HiResCorpus:
        """Return the backing hi-res corpus."""
        return self._corpus

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a Stage 1 training sample built from a hi-res teacher.

        Args:
            index: Sample index.

        Returns:
            Dictionary of tensors and metadata matching the Stage 1 contract.

        Raises:
            IndexError: If index is out of range.

        Physical Basis:
            A hi-res teacher chunk is loaded at the target rate, the 44.1kHz
            input is derived by downsampling the same chunk, and the shared
            assembly applies degradation and amplitude-capped HB targeting.
        """
        if index < 0 or index >= self._config.num_samples:
            raise IndexError("index out of range")

        rng = self._rng_for_index(index)
        teacher_source = self._corpus.load_teacher_source(index, rng)
        source_chunk, teacher_full, chunk_start = build_raw_teacher_chunks_from_source(
            teacher_source=teacher_source,
            source_sr=self._config.source_sample_rate,
            target_sr=self._config.target_sample_rate,
            source_duration_sec=self._config.source_duration_sec,
            chunk_duration_sec=self._config.chunk_duration_sec,
            random_chunk=self._config.random_chunk,
            augmentation=self._config.augmentation,
            teacher_downsample_methods=(
                self._config.degradation.teacher_downsample_methods
            ),
            teacher_downsample_phase_modes=(
                self._config.degradation.teacher_downsample_phase_modes
            ),
            rng=rng,
        )
        return assemble_stage1_sample(
            config=self._config,
            degradation=self._degradation,
            band_split=self._band_split,
            source_chunk=source_chunk,
            teacher_full=teacher_full,
            rng=rng,
            signal_type=HIRES_SIGNAL_TAG,
            chunk_start=chunk_start,
        )

    def _rng_for_index(self, index: int) -> np.random.Generator:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed = int(self._base_seed + index + worker_id * 1_000_000)
        return np.random.default_rng(seed)


def _band_split_at_rate(config: BandSplitConfig, sample_rate: int) -> BandSplitConfig:
    """Return a band-split config bound to the target sample rate."""
    if config.sample_rate == sample_rate:
        return config
    return BandSplitConfig(
        cutoff_hz=config.cutoff_hz,
        sample_rate=sample_rate,
        num_taps=config.num_taps,
        window=config.window,
    )
