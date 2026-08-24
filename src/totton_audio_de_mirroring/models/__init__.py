"""CAPB model and fixed interpolation prototype utilities."""

from totton_audio_de_mirroring.models.capb import CAPB, CAPBController
from totton_audio_de_mirroring.models.proto_bank import (
    PrototypeBank,
    build_prototype_bank,
    prototype_specs_for_target_rate,
)

__all__ = [
    "CAPB",
    "CAPBController",
    "PrototypeBank",
    "build_prototype_bank",
    "prototype_specs_for_target_rate",
]
