"""Evaluation utilities for Stage 1 hard metrics."""

from .frequency_response import (
    FrequencyResponseMetrics,
    compute_frequency_response,
    evaluate_frequency_response_pair,
    plot_frequency_response,
)
from .imd_proxy import (
    IMDPathMetrics,
    IMDProxyMetrics,
    apply_soft_clipping,
    evaluate_imd_path,
    evaluate_imd_proxy,
)
from .lb_preservation import (
    LowBandPreservationMetrics,
    evaluate_lowband_preservation,
)
from .metrics import (
    DatasetEvaluationResult,
    SampleEvaluationResult,
    Stage1HardMetrics,
    evaluate_dataset,
    evaluate_stage1_hard_metrics,
    sample_result_to_flat_dict,
)
from .mirror_metrics import (
    ListeningCorrelationMetrics,
    MirrorDatasetEvaluationResult,
    MirrorReductionMetrics,
    MirrorSampleEvaluationResult,
    MirrorVisualizationArtifacts,
    evaluate_metric_listening_correlation,
    evaluate_mirror_reduction,
    evaluate_mirror_reduction_dataset,
    export_mirror_reduction_visualization,
    mirror_dataset_result_to_payload,
)
from .thdn_visualization import (
    THDNSpectrumMetrics,
    compute_thdn_spectrum,
    evaluate_thdn_spectrum_pair,
    plot_thdn_spectrum,
)
from .time_domain_visualization import (
    ImpulseResponseMetrics,
    SquareWaveMetrics,
    WaveformComparisonMetrics,
    compute_impulse_response,
    compute_square_wave_response,
    compute_waveform_comparison,
    plot_impulse_response,
    plot_square_wave_response,
    plot_waveform_comparison,
)

__all__ = [
    "FrequencyResponseMetrics",
    "ImpulseResponseMetrics",
    "LowBandPreservationMetrics",
    "MirrorReductionMetrics",
    "MirrorSampleEvaluationResult",
    "MirrorDatasetEvaluationResult",
    "ListeningCorrelationMetrics",
    "MirrorVisualizationArtifacts",
    "IMDPathMetrics",
    "IMDProxyMetrics",
    "DatasetEvaluationResult",
    "SampleEvaluationResult",
    "SquareWaveMetrics",
    "Stage1HardMetrics",
    "THDNSpectrumMetrics",
    "WaveformComparisonMetrics",
    "apply_soft_clipping",
    "compute_frequency_response",
    "compute_impulse_response",
    "compute_square_wave_response",
    "compute_thdn_spectrum",
    "compute_waveform_comparison",
    "evaluate_frequency_response_pair",
    "evaluate_imd_path",
    "evaluate_imd_proxy",
    "evaluate_lowband_preservation",
    "evaluate_mirror_reduction_dataset",
    "evaluate_mirror_reduction",
    "evaluate_metric_listening_correlation",
    "evaluate_thdn_spectrum_pair",
    "export_mirror_reduction_visualization",
    "mirror_dataset_result_to_payload",
    "plot_frequency_response",
    "plot_impulse_response",
    "plot_square_wave_response",
    "plot_thdn_spectrum",
    "plot_waveform_comparison",
    "evaluate_dataset",
    "evaluate_stage1_hard_metrics",
    "sample_result_to_flat_dict",
]
