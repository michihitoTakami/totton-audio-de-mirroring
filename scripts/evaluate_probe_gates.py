"""Evaluate a Stage 1 candidate against the worst-case probe gates.

Backends:
- ``prototype:<name>``: a fixed kernel from the CAPB prototype bank.
- ``bessel``: the Bessel reference SRC itself (sanity baseline).
- ``ideal``: an ideal polyphase resampler (upper fidelity baseline).
- ``checkpoint``: an NMSE checkpoint fed with the Bessel reference signal.

Usage examples:
    uv run python scripts/evaluate_probe_gates.py --backend prototype:gentle
    uv run python scripts/evaluate_probe_gates.py --backend checkpoint \
        --checkpoint data/checkpoints/stage1_best.pt \
        --data-config configs/data_generation.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.degradation import upsample_bessel_reference
from totton_audio_de_mirroring.evaluation.gates import (
    ProbeEvaluation,
    evaluate_gates,
    evaluate_probe,
    render_markdown_report,
    report_to_dict,
)
from totton_audio_de_mirroring.evaluation.probe_suite import (
    build_default_probe_suite,
    generate_probe,
    manifest_hash,
    suite_manifest,
)
from totton_audio_de_mirroring.models.proto_bank import (
    build_prototype_bank,
    upsample_with_kernel,
)

SOURCE_SR = 44_100
TARGET_SR = 88_200
BESSEL_CUTOFF_HZ = 20_000.0
BESSEL_ORDER = 6

ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def main() -> None:
    """Run the probe gate evaluation for one backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--tier", choices=["canonical", "held_out", "both"], default="both"
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports/probe_gates"))
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    label = args.label or args.backend.replace(":", "_")
    report_dir = args.report_dir / label
    report_dir.mkdir(parents=True, exist_ok=True)

    model_fn = _build_backend(args)
    suite = build_default_probe_suite()
    if args.tier != "both":
        suite = tuple(spec for spec in suite if spec.tier == args.tier)

    evaluations: list[ProbeEvaluation] = []
    for spec in suite:
        source = generate_probe(spec, SOURCE_SR)
        bessel_ref = upsample_bessel_reference(
            signal=source,
            source_sr=SOURCE_SR,
            target_sr=TARGET_SR,
            cutoff_hz=BESSEL_CUTOFF_HZ,
            order=BESSEL_ORDER,
        )
        ideal_ref = np.asarray(sp_signal.resample_poly(source, 2, 1), dtype=np.float64)
        output = model_fn(source, bessel_ref)
        evaluations.append(
            evaluate_probe(
                spec=spec,
                source=source,
                source_sample_rate=SOURCE_SR,
                bessel_reference=bessel_ref,
                ideal_reference=ideal_ref,
                output=output,
                target_sample_rate=TARGET_SR,
            )
        )

    report = evaluate_gates(
        evaluations,
        manifest_hash=manifest_hash(suite_manifest(build_default_probe_suite())),
    )

    (report_dir / "gate_report.json").write_text(
        json.dumps(report_to_dict(report), indent=2)
    )
    markdown = render_markdown_report(report)
    (report_dir / "gate_report.md").write_text(markdown)
    print(markdown)
    print(f"Reports written to {report_dir}")

    if not report.all_passed and not args.no_strict:
        sys.exit(1)


def _build_backend(args: argparse.Namespace) -> ModelFn:
    """Build the candidate model callable: (source, bessel_ref) -> output."""
    backend = args.backend
    if backend.startswith("prototype:"):
        name = backend.split(":", 1)[1]
        bank = build_prototype_bank()
        if name not in bank.names:
            raise ValueError(f"Unknown prototype '{name}', have {bank.names}.")
        kernel = bank.kernels[bank.names.index(name)]

        def prototype_fn(source: np.ndarray, _bessel: np.ndarray) -> np.ndarray:
            return upsample_with_kernel(source, kernel, bank.upsample_ratio)

        return prototype_fn

    if backend == "bessel":
        return lambda _source, bessel_ref: bessel_ref

    if backend == "ideal":
        return lambda source, _bessel: np.asarray(
            sp_signal.resample_poly(source, 2, 1), dtype=np.float64
        )

    if backend == "checkpoint":
        if args.checkpoint is None or args.data_config is None:
            raise ValueError(
                "checkpoint backend requires --checkpoint and --data-config."
            )
        return _build_checkpoint_backend(args.checkpoint, args.data_config, args.device)

    raise ValueError(f"Unknown backend: {backend}")


def _build_checkpoint_backend(
    checkpoint_path: Path, data_config_path: Path, device: str
) -> ModelFn:
    """Restore an NMSE checkpoint for Stage 1-domain inference.

    Physical Basis:
        Matching training-time band-split filters and energy cap keeps
        checkpoint inference aligned with NMSE safety constraints; the
        model consumes the Bessel reference signal exactly as in training.
    """
    import torch

    from totton_audio_de_mirroring.data.filters import design_band_split_filters
    from totton_audio_de_mirroring.data.pipeline_config import load_data_config
    from totton_audio_de_mirroring.models.nmse import NMSE

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
    incompatible = model.load_state_dict(model_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {incompatible.unexpected_keys}"
        )
    if incompatible.missing_keys:
        print(
            "Warning: checkpoint predates buffers "
            f"{incompatible.missing_keys}; using model defaults."
        )
    model.eval()
    torch_device = torch.device(device)
    model = model.to(torch_device)

    def checkpoint_fn(_source: np.ndarray, bessel_ref: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = (
                torch.from_numpy(np.asarray(bessel_ref, dtype=np.float32))
                .unsqueeze(0)
                .to(torch_device)
            )
            output = model(tensor)
        return np.asarray(output.squeeze(0).detach().cpu().numpy(), dtype=np.float64)

    return checkpoint_fn


if __name__ == "__main__":
    main()
