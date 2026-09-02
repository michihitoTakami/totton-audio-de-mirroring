"""CAPB model and fixed interpolation prototype utilities."""

from totton_audio_de_mirroring.models.capb import (
    CAPB,
    CAPBController,
    capb_candidate_from_checkpoint,
)
from totton_audio_de_mirroring.models.proto_bank import (
    RELEASE_PROTOTYPE_PROFILE,
    TWO_PROTOTYPE_PROFILE,
    PrototypeBank,
    build_prototype_bank,
    build_prototype_bank_for_profile,
    prototype_specs_for_profile,
    prototype_specs_for_target_rate,
    supported_prototype_profiles,
)

__all__ = [
    "CAPB",
    "CAPBController",
    "RELEASE_PROTOTYPE_PROFILE",
    "TWO_PROTOTYPE_PROFILE",
    "PrototypeBank",
    "build_prototype_bank",
    "build_prototype_bank_for_profile",
    "capb_candidate_from_checkpoint",
    "prototype_specs_for_profile",
    "prototype_specs_for_target_rate",
    "supported_prototype_profiles",
]
