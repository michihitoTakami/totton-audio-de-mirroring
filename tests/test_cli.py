"""Tests for end-user batch CLI (`totton-upsample`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from totton_audio_de_mirroring.cli import _build_stage1_processor, run_cli
from totton_audio_de_mirroring.inference import PipelinePerformance, PipelineResult


def _write_test_wav(path: Path, *, sample_rate: int, samples: int = 512) -> None:
    time_axis = np.arange(samples, dtype=np.float64) / float(sample_rate)
    signal = np.sin(2.0 * np.pi * 440.0 * time_axis)
    sf.write(path, signal.astype(np.float32), sample_rate)


def _fake_pipeline_result(output_length: int = 2048) -> PipelineResult:
    return PipelineResult(
        output_signal=np.zeros(output_length, dtype=np.float64),
        stage1_signal=None,
        stage1_reference=None,
        stage1_metrics=None,
        performance=PipelinePerformance(
            latency_sec=0.01,
            input_duration_sec=0.01,
            throughput_x_realtime=1.0,
            num_chunks=1,
            chunk_latency_ms=10.0,
            peak_memory_mb=100.0,
        ),
    )


def _fake_run_stage1_stage2_pipeline(*_: object, **__: object) -> PipelineResult:
    return _fake_pipeline_result()


def test_cli_single_file_writes_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single input should support explicit output wav path."""
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    _write_test_wav(input_path, sample_rate=44_100)

    monkeypatch.setattr(
        "totton_audio_de_mirroring.cli.run_stage1_stage2_pipeline",
        _fake_run_stage1_stage2_pipeline,
    )

    exit_code = run_cli([str(input_path), "-o", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()


def test_cli_batch_outputs_wav_flac_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch mode should write selected output formats for each file."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    _write_test_wav(input_dir / "a.wav", sample_rate=44_100)
    _write_test_wav(input_dir / "b.wav", sample_rate=44_100)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pipeline:",
                "  source_sample_rate: 44100",
                "  stage1_sample_rate: 88200",
                "  output_sample_rate: 352800",
                "  stage2_num_stages: 2",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "totton_audio_de_mirroring.cli.run_stage1_stage2_pipeline",
        _fake_run_stage1_stage2_pipeline,
    )

    exit_code = run_cli(
        [
            str(input_dir / "*.wav"),
            "-o",
            str(output_dir),
            "-c",
            str(config_path),
            "--output-format",
            "wav",
            "--output-format",
            "flac",
            "--output-format",
            "metadata",
        ]
    )

    assert exit_code == 0
    for stem in ("a", "b"):
        assert (output_dir / f"{stem}.wav").exists()
        assert (output_dir / f"{stem}.flac").exists()
        metadata_path = output_dir / f"{stem}.json"
        assert metadata_path.exists()
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert payload["output_sample_rate"] == 352_800


def test_cli_continues_on_error_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch mode should continue processing and return non-zero when failures exist."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    good = input_dir / "good.wav"
    bad = input_dir / "bad.wav"
    _write_test_wav(good, sample_rate=44_100)
    _write_test_wav(bad, sample_rate=48_000)

    monkeypatch.setattr(
        "totton_audio_de_mirroring.cli.run_stage1_stage2_pipeline",
        _fake_run_stage1_stage2_pipeline,
    )

    exit_code = run_cli(
        [
            str(good),
            str(bad),
            "-o",
            str(output_dir),
            "--output-format",
            "wav",
        ]
    )

    assert exit_code == 1
    assert (output_dir / "good.wav").exists()
    assert not (output_dir / "bad.wav").exists()


def test_build_stage1_processor_nmse_uses_device_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Device override should be forwarded to NMSE loader."""
    calls: list[dict[str, object]] = []

    def _fake_loader(
        *, checkpoint_path: Path, data_config_path: Path, device: str
    ) -> object:
        calls.append(
            {
                "checkpoint_path": checkpoint_path,
                "data_config_path": data_config_path,
                "device": device,
            }
        )
        return object()

    monkeypatch.setattr(
        "totton_audio_de_mirroring.cli.load_nmse_stage1_processor",
        _fake_loader,
    )

    _ = _build_stage1_processor(
        {"mode": "nmse", "checkpoint_path": "model.pt"},
        device_override="cuda",
    )

    assert calls[0]["device"] == "cuda"


def test_cli_fail_fast_stops_batch_on_first_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-fast should stop batch processing after first failed item."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    bad = input_dir / "a_bad.wav"
    good = input_dir / "b_good.wav"
    _write_test_wav(bad, sample_rate=48_000)
    _write_test_wav(good, sample_rate=44_100)

    monkeypatch.setattr(
        "totton_audio_de_mirroring.cli.run_stage1_stage2_pipeline",
        _fake_run_stage1_stage2_pipeline,
    )

    exit_code = run_cli(
        [
            str(input_dir / "*.wav"),
            "-o",
            str(output_dir),
            "--output-format",
            "wav",
            "--fail-fast",
        ]
    )

    assert exit_code == 1
    assert not (output_dir / "b_good.wav").exists()


def test_cli_flac_guard_returns_user_friendly_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """FLAC output above format limit should fail with concise user message."""
    input_path = tmp_path / "input.wav"
    output_dir = tmp_path / "out"
    _write_test_wav(input_path, sample_rate=44_100)

    monkeypatch.setattr(
        "totton_audio_de_mirroring.cli.run_stage1_stage2_pipeline",
        _fake_run_stage1_stage2_pipeline,
    )

    exit_code = run_cli(
        [
            str(input_path),
            "-o",
            str(output_dir),
            "--output-format",
            "flac",
            "--fail-fast",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Failed to process" in captured.err
    assert "FLAC output is not supported above 655350 Hz" in captured.err


def test_cli_reports_setup_error_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Config validation failures should return exit code 1 with short message."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("[]\n", encoding="utf-8")

    exit_code = run_cli(
        [
            "missing.wav",
            "-o",
            str(tmp_path / "out"),
            "-c",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error: Config root must be a mapping." in captured.err
