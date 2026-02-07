"""Execute Issue #63 workflow: retrain, evaluate, and select best Stage 1 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from totton_audio_de_mirroring.data.filters import design_band_split_filters
from totton_audio_de_mirroring.data.pipeline_config import load_data_config
from totton_audio_de_mirroring.evaluation.imd_proxy import evaluate_imd_proxy
from totton_audio_de_mirroring.models.nmse import NMSE
from totton_audio_de_mirroring.training.trainer import (
    TrainingConfig,
    load_training_config,
)


@dataclass(frozen=True)
class CandidateEvaluation:
    """Evaluation summary for one checkpoint candidate.

    Args:
        checkpoint_path: Path to evaluated checkpoint.
        output_dir: Directory with generated Stage 1 outputs.
        hard_summary: Aggregated hard-metric summary payload.
        mirror_summary: Aggregated mirror-metric summary payload.
        imd_summary: Aggregated IMD proxy summary payload.
        passes_hard_gate: True when LB preservation and energy-cap gate passes.
        passes_imd_gate: True when IMD vs naive shows improvement.
        composite_score: Ranking score; higher is better.

    Physical Basis:
        Candidate selection prioritizes audible-band safety first
        (LB preservation and cap compliance), then IMD-risk reduction.
    """

    checkpoint_path: Path
    output_dir: Path
    hard_summary: dict[str, Any]
    mirror_summary: dict[str, Any]
    imd_summary: dict[str, Any]
    passes_hard_gate: bool
    passes_imd_gate: bool
    composite_score: float


@dataclass(frozen=True)
class GateConfig:
    """Acceptance thresholds for Issue #63 checkpoint selection.

    Args:
        max_lb_phase_error_deg: Maximum mean LB phase error.
        max_lb_group_delay_error_samples: Maximum mean LB group-delay error.
        max_lb_amplitude_error_db: Maximum LB waveform error in dB.
        require_zero_energy_cap_violations: Require no HB cap violations.
        require_positive_thdn_improvement: Require THD+N improvement > 0 dB.

    Physical Basis:
        Hard gates enforce low-band integrity and high-band safety before
        ranking candidates by mirror and IMD behavior.
    """

    max_lb_phase_error_deg: float
    max_lb_group_delay_error_samples: float
    max_lb_amplitude_error_db: float
    require_zero_energy_cap_violations: bool
    require_positive_thdn_improvement: bool


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Issue #63 workflow.

    Physical Basis:
        Reproducible training/evaluation requires explicit control of seeds,
        config paths, and dataset pairing for hard metrics and IMD checks.
    """
    parser = argparse.ArgumentParser(description="Run Issue #63 Stage 1 workflow")
    parser.add_argument(
        "--data-config", type=Path, default=Path("configs/data_generation.yaml")
    )
    parser.add_argument(
        "--train-config", type=Path, default=Path("configs/training_stage1.yaml")
    )
    parser.add_argument("--eval-input-dir", type=Path, required=True)
    parser.add_argument("--imd-naive-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/issue63"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("data/checkpoints/issue63")
    )
    parser.add_argument("--eval-glob", type=str, default="*.npy")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--energy-cap", type=float, default=None)
    parser.add_argument("--mirror-target-reduction", type=float, default=0.70)
    parser.add_argument(
        "--candidate-checkpoints",
        nargs="*",
        default=["stage1_best.pt", "stage1_last.pt"],
    )
    parser.add_argument("--max-lb-phase-error-deg", type=float, default=15.0)
    parser.add_argument("--max-lb-group-delay-error-samples", type=float, default=600.0)
    parser.add_argument("--max-lb-amplitude-error-db", type=float, default=-20.0)
    parser.add_argument("--allow-energy-cap-violations", action="store_true")
    parser.add_argument("--allow-nonpositive-thdn-improvement", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run full Issue #63 execution workflow.

    Raises:
        FileNotFoundError: If required input files/directories are missing.
        RuntimeError: If training/evaluation or report serialization fails.

    Physical Basis:
        The workflow enforces fixed training conditions, then verifies that
        selected Stage 1 checkpoints preserve LB while reducing mirror/IMD risk.
    """
    args = parse_args()
    _validate_args(args)

    data_config = load_data_config(args.data_config)
    training_config = _load_and_fix_training_config(
        train_config_path=args.train_config,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        energy_cap=args.energy_cap,
    )
    gate_config = GateConfig(
        max_lb_phase_error_deg=args.max_lb_phase_error_deg,
        max_lb_group_delay_error_samples=args.max_lb_group_delay_error_samples,
        max_lb_amplitude_error_db=args.max_lb_amplitude_error_db,
        require_zero_energy_cap_violations=not args.allow_energy_cap_violations,
        require_positive_thdn_improvement=(not args.allow_nonpositive_thdn_improvement),
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(
        report_dir=args.report_dir,
        args=args,
        training_config=training_config,
        gate_config=gate_config,
    )

    if not args.skip_training:
        _run_training_command(args=args, training_config=training_config)

    input_signals = _load_input_signals(args.eval_input_dir, args.eval_glob)
    candidates = _resolve_candidate_paths(
        args.checkpoint_dir, args.candidate_checkpoints
    )

    evaluated: list[CandidateEvaluation] = []
    for checkpoint_path in candidates:
        output_dir = args.report_dir / "candidate_outputs" / checkpoint_path.stem
        _run_stage1_inference_for_inputs(
            checkpoint_path=checkpoint_path,
            data_config_path=args.data_config,
            input_signals=input_signals,
            output_dir=output_dir,
            device=args.device,
        )

        evaluate_payload = _run_stage1_hard_metrics_command(
            args=args,
            output_dir=output_dir,
            sample_rate=data_config.target_sample_rate,
            energy_cap=training_config.energy_cap,
            mirror_target_reduction=args.mirror_target_reduction,
            candidate_name=checkpoint_path.stem,
        )
        imd_summary = _evaluate_imd_dataset(
            naive_dir=args.imd_naive_dir,
            nmse_dir=output_dir,
            sample_rate=data_config.target_sample_rate,
        )

        hard_summary = _load_hard_summary(evaluate_payload)
        mirror_summary = _load_mirror_summary(evaluate_payload)

        passes_hard = _passes_hard_gate(
            hard_summary=hard_summary, gate_config=gate_config
        )
        passes_imd = _passes_imd_gate(imd_summary=imd_summary, gate_config=gate_config)
        score = _compute_composite_score(
            hard_summary=hard_summary,
            mirror_summary=mirror_summary,
            imd_summary=imd_summary,
        )
        evaluated.append(
            CandidateEvaluation(
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                hard_summary=hard_summary,
                mirror_summary=mirror_summary,
                imd_summary=imd_summary,
                passes_hard_gate=passes_hard,
                passes_imd_gate=passes_imd,
                composite_score=score,
            )
        )

    selected = _select_best_candidate(evaluated)
    selected_dir = args.report_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_checkpoint_path = selected_dir / "stage1_best_selected.pt"
    shutil.copy2(selected.checkpoint_path, selected_checkpoint_path)

    report_payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "selected_checkpoint": str(selected.checkpoint_path),
        "selected_checkpoint_copy": str(selected_checkpoint_path),
        "selection_reason": {
            "passes_hard_gate": selected.passes_hard_gate,
            "passes_imd_gate": selected.passes_imd_gate,
            "composite_score": selected.composite_score,
        },
        "candidates": [_candidate_to_payload(candidate) for candidate in evaluated],
    }
    (selected_dir / "selection_report.json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"selected_checkpoint={selected.checkpoint_path}")
    print(f"selected_checkpoint_copy={selected_checkpoint_path}")
    print(f"selection_report={selected_dir / 'selection_report.json'}")


def _load_and_fix_training_config(
    *,
    train_config_path: Path,
    seed: int,
    epochs: int | None,
    device: str,
    energy_cap: float | None,
) -> TrainingConfig:
    """Load training config and apply deterministic overrides.

    Physical Basis:
        Fixing seed/device/cap conditions is required to compare checkpoints
        under identical optimization and safety constraints.
    """
    base = load_training_config(train_config_path)
    payload = asdict(base)
    payload["seed"] = seed
    payload["device"] = device
    payload["require_cuda"] = not device.startswith("cpu")
    if epochs is not None:
        payload["epochs"] = epochs
    if energy_cap is not None:
        payload["energy_cap"] = energy_cap
    return TrainingConfig.from_dict(payload)


def _run_training_command(
    args: argparse.Namespace, training_config: TrainingConfig
) -> None:
    """Run scripts/train_stage1.py with frozen settings.

    Raises:
        RuntimeError: If the training process exits with non-zero status.

    Physical Basis:
        Reusing the project training CLI keeps loss composition and
        checkpoint semantics consistent with existing Stage 1 flow.
    """
    command = [
        sys.executable,
        "scripts/train_stage1.py",
        "--data-config",
        str(args.data_config),
        "--train-config",
        str(args.train_config),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--validation-split",
        str(args.validation_split),
        "--device",
        str(training_config.device),
    ]
    if training_config.require_cuda:
        command.append("--require-cuda")
    else:
        command.append("--allow-cpu")
    if training_config.seed is not None:
        command.extend(["--seed", str(training_config.seed)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.energy_cap is not None:
        command.extend(["--energy-cap", str(args.energy_cap)])

    training_log = args.report_dir / "training_stdout_stderr.log"
    return_code = _run_command_with_live_log(
        command,
        log_path=training_log,
        section_label="train_stage1",
    )
    if return_code != 0:
        raise RuntimeError(f"Stage 1 training failed. See log: {training_log}")


def _load_input_signals(input_dir: Path, pattern: str) -> dict[str, np.ndarray]:
    """Load evaluation input signals from directory.

    Physical Basis:
        Deterministic file-based pairing guarantees reproducible candidate
        comparisons across checkpoints.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"eval_input_dir not found: {input_dir}")

    loaded: dict[str, np.ndarray] = {}
    for path in sorted(input_dir.glob(pattern)):
        loaded[path.stem] = np.asarray(np.load(path), dtype=np.float64)

    if len(loaded) == 0:
        raise FileNotFoundError(
            f"No input files matched pattern '{pattern}' in {input_dir}."
        )
    return loaded


def _resolve_candidate_paths(checkpoint_dir: Path, names: list[str]) -> list[Path]:
    """Resolve candidate checkpoint files from names.

    Physical Basis:
        Evaluating both best/last checkpoints guards against selection drift.
    """
    resolved: list[Path] = []
    for name in names:
        path = checkpoint_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Candidate checkpoint not found: {path}")
        resolved.append(path)
    if len(resolved) == 0:
        raise ValueError("candidate_checkpoints cannot be empty")
    return resolved


def _build_stage1_model_from_checkpoint(
    checkpoint_path: Path,
    data_config_path: Path,
    device: str,
) -> NMSE:
    """Restore NMSE model for direct Stage 1-domain inference.

    Physical Basis:
        Matching training-time band-split filters and energy cap keeps
        checkpoint inference aligned with NMSE safety constraints.
    """
    data_config = load_data_config(data_config_path)
    lowpass_taps, highpass_taps = design_band_split_filters(
        cutoff_hz=data_config.band_split.cutoff_hz,
        sample_rate=data_config.band_split.sample_rate,
        num_taps=data_config.band_split.num_taps,
        window=data_config.band_split.window,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_config_raw = checkpoint.get("training_config", {})
    energy_cap = float(
        training_config_raw.get("energy_cap", data_config.hb_target.energy_cap)
    )

    model = NMSE(
        sample_rate=data_config.target_sample_rate,
        cutoff_hz=data_config.band_split.cutoff_hz,
        energy_cap=energy_cap,
        envelope_floor=data_config.hb_target.envelope_min,
        lowpass_taps=lowpass_taps,
        highpass_taps=highpass_taps,
    )
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise RuntimeError(f"Invalid checkpoint model_state: {checkpoint_path}")
    model.load_state_dict(model_state)
    model.eval()
    return model.to(torch.device(device))


def _run_stage1_inference_for_inputs(
    *,
    checkpoint_path: Path,
    data_config_path: Path,
    input_signals: dict[str, np.ndarray],
    output_dir: Path,
    device: str,
) -> None:
    """Generate Stage 1 outputs for one checkpoint on fixed inputs.

    Physical Basis:
        Candidate outputs must be generated from identical inputs to compare
        mirror reduction and IMD proxy behavior fairly.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _build_stage1_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        data_config_path=data_config_path,
        device=device,
    )
    torch_device = torch.device(device)

    with torch.no_grad():
        for sample_id, signal in input_signals.items():
            tensor = (
                torch.from_numpy(np.asarray(signal, dtype=np.float32))
                .unsqueeze(0)
                .to(torch_device)
            )
            output = model(tensor)
            output_np = np.asarray(
                output.squeeze(0).detach().cpu().numpy(), dtype=np.float64
            )
            np.save(output_dir / f"{sample_id}.npy", output_np)


def _run_stage1_hard_metrics_command(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    sample_rate: int,
    energy_cap: float,
    mirror_target_reduction: float,
    candidate_name: str,
) -> dict[str, Any]:
    """Run scripts/evaluate_stage1.py and return parsed JSON payload.

    Raises:
        RuntimeError: If evaluation command fails or JSON parsing fails.

    Physical Basis:
        Reusing evaluate_stage1 CLI keeps README 7.4 metric semantics
        consistent between manual and automated checkpoint selection runs.
    """
    metrics_dir = args.report_dir / "candidate_metrics" / candidate_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "stage1_metrics.json"
    csv_path = metrics_dir / "stage1_metrics.csv"
    log_path = metrics_dir / "evaluate_stage1_stdout_stderr.log"
    command = [
        sys.executable,
        "scripts/evaluate_stage1.py",
        "--input-dir",
        str(args.eval_input_dir),
        "--output-dir",
        str(output_dir),
        "--glob",
        args.eval_glob,
        "--sample-rate",
        str(sample_rate),
        "--energy-cap",
        str(energy_cap),
        "--mirror-target-reduction",
        str(mirror_target_reduction),
        "--json",
        str(json_path),
        "--csv",
        str(csv_path),
    ]
    return_code = _run_command_with_live_log(
        command,
        log_path=log_path,
        section_label=f"evaluate_stage1:{candidate_name}",
    )
    if return_code != 0:
        raise RuntimeError(f"Stage 1 evaluation failed. See log: {log_path}")
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse stage1 metrics JSON ({json_path}): {exc}"
        ) from exc


def _load_hard_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract hard metric summary from evaluate_stage1 JSON payload.

    Raises:
        RuntimeError: If summary payload is missing or malformed.

    Physical Basis:
        Hard summary drives LB-preservation and energy-cap safety gates.
    """
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("evaluate_stage1 payload missing 'summary' object.")
    return dict(summary)


def _load_mirror_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract mirror summary from evaluate_stage1 JSON payload.

    Raises:
        RuntimeError: If mirror summary payload is missing or malformed.

    Physical Basis:
        Mirror summary quantifies alias/mirror suppression used in ranking.
    """
    mirror = payload.get("mirror_metrics")
    if not isinstance(mirror, dict):
        raise RuntimeError("evaluate_stage1 payload missing 'mirror_metrics' object.")
    summary = mirror.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("evaluate_stage1 payload missing mirror summary.")
    return dict(summary)


def _evaluate_imd_dataset(
    *,
    naive_dir: Path,
    nmse_dir: Path,
    sample_rate: int,
) -> dict[str, Any]:
    """Evaluate IMD proxy against naive references over shared sample IDs.

    Physical Basis:
        IMD proxy compares audible-band distortion after nonlinear transfer,
        indicating whether HB suppression improves playback robustness.
    """
    if not naive_dir.exists() or not naive_dir.is_dir():
        raise FileNotFoundError(f"imd_naive_dir not found: {naive_dir}")
    if not nmse_dir.exists() or not nmse_dir.is_dir():
        raise FileNotFoundError(f"nmse_dir not found: {nmse_dir}")

    sample_ids = sorted(path.stem for path in naive_dir.glob("*.npy"))
    if len(sample_ids) == 0:
        raise FileNotFoundError(f"No IMD naive files found in {naive_dir}")

    per_sample: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        naive_path = naive_dir / f"{sample_id}.npy"
        nmse_path = nmse_dir / f"{sample_id}.npy"
        if not nmse_path.exists():
            raise FileNotFoundError(f"Missing NMSE IMD pair: {nmse_path}")

        naive_signal = np.asarray(np.load(naive_path), dtype=np.float64)
        nmse_signal = np.asarray(np.load(nmse_path), dtype=np.float64)
        metrics = evaluate_imd_proxy(
            naive_signal=naive_signal,
            nmse_signal=nmse_signal,
            sample_rate=sample_rate,
            num_taps=1025,
        )
        per_sample.append(
            {
                "sample_id": sample_id,
                "audible_distortion_reduction_db": metrics.audible_distortion_reduction_db,
                "thdn_improvement_db": metrics.thdn_improvement_db,
                "nmse_has_lower_imd": metrics.nmse_has_lower_imd,
                "thdn_improvement_over_10db": metrics.thdn_improvement_over_10db,
            }
        )

    return {
        "num_samples": len(per_sample),
        "mean_audible_distortion_reduction_db": float(
            np.mean(
                [float(item["audible_distortion_reduction_db"]) for item in per_sample]
            )
        ),
        "mean_thdn_improvement_db": float(
            np.mean([float(item["thdn_improvement_db"]) for item in per_sample])
        ),
        "all_nmse_has_lower_imd": bool(
            all(bool(item["nmse_has_lower_imd"]) for item in per_sample)
        ),
        "all_thdn_improvement_over_10db": bool(
            all(bool(item["thdn_improvement_over_10db"]) for item in per_sample)
        ),
        "samples": per_sample,
    }


def _passes_hard_gate(*, hard_summary: dict[str, Any], gate_config: GateConfig) -> bool:
    """Check hard metric gate for LB preservation and energy cap.

    Physical Basis:
        Stage 1 acceptance requires low-band integrity and no unsafe
        high-band cap violations.
    """
    cap_violation_rate = float(hard_summary["hb_energy_cap_violation_rate"])
    lb_phase_error = float(hard_summary["lb_phase_error_deg"])
    lb_group_delay_error = float(hard_summary["lb_group_delay_error_samples"])
    lb_amplitude_error = float(hard_summary["lb_amplitude_error_db"])

    if gate_config.require_zero_energy_cap_violations and cap_violation_rate > 0.0:
        return False
    if lb_phase_error > gate_config.max_lb_phase_error_deg:
        return False
    if lb_group_delay_error > gate_config.max_lb_group_delay_error_samples:
        return False
    return lb_amplitude_error <= gate_config.max_lb_amplitude_error_db


def _passes_imd_gate(*, imd_summary: dict[str, Any], gate_config: GateConfig) -> bool:
    """Check IMD gate against naive baseline improvement.

    Physical Basis:
        Positive THD+N improvement and lower IMD indicate reduced audible
        nonlinearity artifacts in downstream analog stages.
    """
    if not bool(imd_summary["all_nmse_has_lower_imd"]):
        return False
    return not (
        gate_config.require_positive_thdn_improvement
        and float(imd_summary["mean_thdn_improvement_db"]) <= 0.0
    )


def _compute_composite_score(
    *,
    hard_summary: dict[str, Any],
    mirror_summary: dict[str, Any],
    imd_summary: dict[str, Any],
) -> float:
    """Compute ranking score for tie-breaking between passing candidates.

    Physical Basis:
        Ranking prioritizes THD+N and mirror-symmetry reduction, while
        penalizing touch and low-band phase error increases.
    """
    return float(
        2.0 * float(imd_summary["mean_thdn_improvement_db"])
        + 1.0 * float(mirror_summary["symmetry_reduction_ratio"])
        - 0.2 * float(hard_summary["touch_metric"])
        - 0.05 * float(hard_summary["lb_phase_error_deg"])
    )


def _select_best_candidate(
    candidates: list[CandidateEvaluation],
) -> CandidateEvaluation:
    """Select best candidate using gates then composite score.

    Raises:
        RuntimeError: If no candidate satisfies required hard gate.

    Physical Basis:
        Hard safety gates are mandatory; ranking is only valid among safe
        candidates.
    """
    if len(candidates) == 0:
        raise ValueError("candidates cannot be empty")

    passing = [
        candidate
        for candidate in candidates
        if candidate.passes_hard_gate and candidate.passes_imd_gate
    ]
    if len(passing) == 0:
        raise RuntimeError("No checkpoint passed hard+IMD gates.")

    return sorted(
        passing,
        key=lambda item: (
            item.composite_score,
            float(item.imd_summary["mean_thdn_improvement_db"]),
            float(item.mirror_summary["symmetry_reduction_ratio"]),
            -float(item.hard_summary["touch_metric"]),
        ),
        reverse=True,
    )[0]


def _candidate_to_payload(candidate: CandidateEvaluation) -> dict[str, Any]:
    return {
        "checkpoint_path": str(candidate.checkpoint_path),
        "output_dir": str(candidate.output_dir),
        "passes_hard_gate": candidate.passes_hard_gate,
        "passes_imd_gate": candidate.passes_imd_gate,
        "composite_score": candidate.composite_score,
        "hard_metrics": candidate.hard_summary,
        "mirror_metrics": candidate.mirror_summary,
        "imd_proxy": candidate.imd_summary,
    }


def _run_command_with_live_log(
    command: list[str],
    *,
    log_path: Path,
    section_label: str,
) -> int:
    """Run command and stream stdout/stderr to log file in real time.

    Args:
        command: Command and args to execute.
        log_path: Log file path.
        section_label: Human-readable section label.

    Returns:
        Process return code.

    Raises:
        RuntimeError: If subprocess start/streaming fails.

    Physical Basis:
        Real-time logs are required to monitor long-running Stage 1 training
        and pinpoint failure epoch/step without waiting for process exit.
    """
    if len(command) == 0:
        raise ValueError("command cannot be empty.")

    ensure_parent = log_path.parent
    ensure_parent.mkdir(parents=True, exist_ok=True)
    header = f"[{_timestamp_utc()}] [{section_label}] $ {shlex.join(command)}\n"

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(header)
            log_file.flush()
            print(header.rstrip(), flush=True)

            process = subprocess.Popen(  # noqa: S603
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            if process.stdout is None:
                raise RuntimeError("Failed to capture subprocess output stream.")

            for line in process.stdout:
                stamped = f"[{_timestamp_utc()}] [{section_label}] {line}"
                log_file.write(stamped)
                log_file.flush()
                print(stamped.rstrip(), flush=True)

            return_code = process.wait()
            footer = f"[{_timestamp_utc()}] [{section_label}] exit_code={return_code}\n"
            log_file.write(footer)
            log_file.flush()
            print(footer.rstrip(), flush=True)
            return int(return_code)
    except Exception as exc:
        raise RuntimeError(f"Failed while running command: {command}: {exc}") from exc


def _timestamp_utc() -> str:
    """Return compact UTC timestamp for log lines."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _write_run_manifest(
    *,
    report_dir: Path,
    args: argparse.Namespace,
    training_config: TrainingConfig,
    gate_config: GateConfig,
) -> None:
    """Persist immutable run manifest for reproducibility.

    Physical Basis:
        Fixing and hashing configs prevents accidental drift across
        retraining attempts and supports checkpoint provenance tracking.
    """
    manifest = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "argv": sys.argv,
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "training_config": asdict(training_config),
        "gate_config": asdict(gate_config),
        "data_config_sha256": _sha256_file(args.data_config),
        "train_config_sha256": _sha256_file(args.train_config),
        "git_commit": _git_commit_or_unknown(),
    }
    (report_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            block = file_obj.read(64 * 1024)
            if len(block) == 0:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_commit_or_unknown() -> str:
    completed = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip()


def _validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.validation_split < 0.0 or args.validation_split >= 1.0:
        raise ValueError("validation_split must be in [0.0, 1.0)")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if len(args.device.strip()) == 0:
        raise ValueError("device must be non-empty")


if __name__ == "__main__":
    main()
