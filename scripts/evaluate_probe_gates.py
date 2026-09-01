"""Evaluate a Stage 1 candidate against the worst-case probe gates.

Backends:
- ``prototype:<name>``: a fixed kernel from the CAPB prototype bank.
- ``bessel``: the Bessel reference SRC itself (sanity baseline).
- ``ideal``: an ideal polyphase resampler (upper fidelity baseline).
- ``capb``: a CAPB checkpoint, or the untrained structural baseline.

Usage examples:
    uv run python scripts/evaluate_probe_gates.py --backend prototype:gentle
    uv run python scripts/evaluate_probe_gates.py --backend capb \
        --checkpoint data/checkpoints/capb/capb_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

from totton_audio_de_mirroring.data.reference import upsample_bessel_reference
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
    RELEASE_PROTOTYPE_PROFILE,
    build_prototype_bank_for_profile,
    supported_prototype_profiles,
    upsample_with_kernel,
)
from totton_audio_de_mirroring.torch_precision import configure_torch_precision

BESSEL_CUTOFF_HZ = 20_000.0
BESSEL_ORDER = 6
# Rate families: source/target sample rates. The Bessel reference cutoff and
# the gate windows/floors are absolute (audible-band based) and shared.
RATE_FAMILIES: dict[str, tuple[int, int]] = {
    "44k1": (44_100, 88_200),
    "48k": (48_000, 96_000),
}

ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def main() -> None:
    """Run the probe gate evaluation for one backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument(
        "--prototype-profile",
        choices=supported_prototype_profiles(),
        default=None,
        help="Override the checkpoint bank for a controller-only experiment.",
    )
    parser.add_argument(
        "--fir-compute-dtype",
        choices=("float32", "float64"),
        default=None,
        help="Override fixed-FIR arithmetic for a controller-only experiment.",
    )
    parser.add_argument("--rate-family", choices=sorted(RATE_FAMILIES), default="44k1")
    parser.add_argument(
        "--tier", choices=["canonical", "held_out", "both"], default="both"
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports/probe_gates"))
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    precision = configure_torch_precision(args.device, allow_tf32=args.allow_tf32)
    source_sr, target_sr = RATE_FAMILIES[args.rate_family]
    default_label = args.backend.replace(":", "_")
    if args.prototype_profile is not None:
        default_label += f"_{args.prototype_profile}"
    if args.fir_compute_dtype is not None:
        default_label += f"_{args.fir_compute_dtype}"
    if args.rate_family != "44k1":
        default_label += f"_{args.rate_family}"
    label = args.label or default_label
    report_dir = args.report_dir / label
    report_dir.mkdir(parents=True, exist_ok=True)

    model_fn, candidate_identity = _build_backend(args, target_sr)
    suite = build_default_probe_suite()
    if args.tier != "both":
        suite = tuple(spec for spec in suite if spec.tier == args.tier)

    evaluations: list[ProbeEvaluation] = []
    for spec in suite:
        source = generate_probe(spec, source_sr)
        bessel_ref = upsample_bessel_reference(
            signal=source,
            source_sr=source_sr,
            target_sr=target_sr,
            cutoff_hz=BESSEL_CUTOFF_HZ,
            order=BESSEL_ORDER,
        )
        ideal_ref = np.asarray(sp_signal.resample_poly(source, 2, 1), dtype=np.float64)
        output = model_fn(source, bessel_ref)
        evaluations.append(
            evaluate_probe(
                spec=spec,
                source=source,
                source_sample_rate=source_sr,
                bessel_reference=bessel_ref,
                ideal_reference=ideal_ref,
                output=output,
                target_sample_rate=target_sr,
            )
        )

    report = evaluate_gates(
        evaluations,
        manifest_hash=manifest_hash(
            suite_manifest(build_default_probe_suite(), source_sr)
        ),
    )

    payload = report_to_dict(report)
    payload["execution"] = precision.to_dict()
    payload["candidate_identity"] = candidate_identity
    (report_dir / "gate_report.json").write_text(json.dumps(payload, indent=2))
    markdown = render_markdown_report(report)
    (report_dir / "gate_report.md").write_text(markdown)
    print(markdown)
    print(f"Reports written to {report_dir}")

    if not report.all_passed and not args.no_strict:
        sys.exit(1)


def _build_backend(
    args: argparse.Namespace, target_sr: int
) -> tuple[ModelFn, dict[str, str]]:
    """Build the candidate model callable: (source, bessel_ref) -> output."""
    backend = args.backend
    if backend.startswith("prototype:"):
        name = backend.split(":", 1)[1]
        profile = args.prototype_profile or RELEASE_PROTOTYPE_PROFILE
        bank = build_prototype_bank_for_profile(
            target_sr,
            profile,
        )
        if name not in bank.names:
            raise ValueError(f"Unknown prototype '{name}', have {bank.names}.")
        kernel = bank.kernels[bank.names.index(name)]

        def prototype_fn(source: np.ndarray, _bessel: np.ndarray) -> np.ndarray:
            return upsample_with_kernel(source, kernel, bank.upsample_ratio)

        return prototype_fn, {
            "prototype_profile": bank.profile_name,
            "prototype_hash": bank.coefficient_hash,
            "fir_compute_dtype": "float64",
        }

    if backend == "bessel":
        return (lambda _source, bessel_ref: bessel_ref), {}

    if backend == "ideal":
        return (
            lambda source, _bessel: np.asarray(
                sp_signal.resample_poly(source, 2, 1), dtype=np.float64
            ),
            {},
        )

    if backend == "capb":
        return _build_capb_backend(
            args.checkpoint,
            args.device,
            target_sr,
            prototype_profile=args.prototype_profile,
            fir_compute_dtype=args.fir_compute_dtype,
        )

    raise ValueError(f"Unknown backend: {backend}")


def _build_capb_backend(
    checkpoint_path: Path | None,
    device: str,
    target_sr: int,
    *,
    prototype_profile: str | None = None,
    fir_compute_dtype: str | None = None,
) -> tuple[ModelFn, dict[str, str]]:
    """Build a CAPB backend (untrained init blend if no checkpoint given).

    Physical Basis:
        The untrained CAPB equals the fixed init blend of prototypes, so
        gating it validates the structural baseline before any training.
    """
    import torch

    from totton_audio_de_mirroring.models.capb import (
        CAPB,
        capb_candidate_from_checkpoint,
        capb_from_checkpoint,
    )

    profile = prototype_profile or RELEASE_PROTOTYPE_PROFILE
    dtype = fir_compute_dtype or "float32"
    bank = build_prototype_bank_for_profile(target_sr, profile)
    model = CAPB(bank=bank, fir_compute_dtype=dtype)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_input_rate = checkpoint.get("expected_input_rate")
        if (
            expected_input_rate is not None
            and int(expected_input_rate) * bank.upsample_ratio != target_sr
        ):
            raise ValueError(
                f"Checkpoint expects input rate {expected_input_rate} Hz, "
                f"incompatible with --rate-family target {target_sr} Hz."
            )
        if prototype_profile is not None or fir_compute_dtype is not None:
            model = capb_candidate_from_checkpoint(
                checkpoint,
                prototype_profile=profile,
                fir_compute_dtype=dtype,
            )
        else:
            model = capb_from_checkpoint(checkpoint)
    model.eval()
    torch_device = torch.device(device)
    model = model.to(torch_device)

    def capb_fn(source: np.ndarray, _bessel: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = (
                torch.from_numpy(np.asarray(source, dtype=np.float32))
                .unsqueeze(0)
                .to(torch_device)
            )
            output = model(tensor)
        return np.asarray(output.squeeze(0).cpu().numpy(), dtype=np.float64)

    return capb_fn, {
        "prototype_profile": model.prototype_profile,
        "prototype_hash": model.prototype_hash,
        "fir_compute_dtype": model.fir_compute_dtype,
    }


if __name__ == "__main__":
    main()
