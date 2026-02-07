"""ctypes bridge to the C++ Stage 2 multi-stage upsampler core API."""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_ERROR_BUFFER_BYTES = 4096


@dataclass(frozen=True)
class CppStage2RuntimeConfig:
    """Runtime configuration for C++ Stage 2 core API access.

    Args:
        config_dir: Directory containing ``stage{i}_taps.txt`` files.
        num_stages: Number of cascaded 2x interpolation stages.
        cpp_project_dir: CMake project root for the C++ Stage 2 core.
        cpp_build_dir: CMake build output directory.

    Physical Basis:
        Stage 2 runs as a cascade of deterministic 2x FIR stages, and this
        config fixes the exact stage count and tap set used in inference.
    """

    config_dir: Path
    num_stages: int
    cpp_project_dir: Path = Path("cpp")
    cpp_build_dir: Path = Path("cpp/build")

    def __post_init__(self) -> None:
        if self.num_stages <= 0:
            raise ValueError("num_stages must be positive.")


class CppStage2Upsampler:
    """Stateful Stage 2 upsampler backed by C++ core API.

    Physical Basis:
        Keeping one persistent core instance preserves FIR history across
        chunk boundaries, matching streaming behavior expected by Stage 2.
    """

    def __init__(self, config: CppStage2RuntimeConfig) -> None:
        self._config = config
        self._handle: ctypes.c_void_p | None = None
        self._lib: ctypes.CDLL | None = None
        self._lib = _load_capi_library(config.cpp_project_dir, config.cpp_build_dir)
        self._handle = self._create_handle()

    def close(self) -> None:
        """Release native resources if allocated."""
        handle = self._handle
        library = self._lib
        if handle is not None and library is not None:
            library.tadm_destroy_multistage_upsampler(handle)
            self._handle = None

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Upsample one signal block using the C++ Stage 2 core.

        Args:
            signal: Input Stage 1 block at 88.2kHz.

        Returns:
            Output block after ``num_stages`` cascade.

        Physical Basis:
            Core processing uses 2x zero-stuff + FIR interpolation per stage,
            identical to the C++ benchmark path.
        """
        _validate_signal(signal)
        if self._handle is None:
            raise RuntimeError("C++ Stage 2 handle is closed.")
        library = self._lib
        if library is None:
            raise RuntimeError("C++ Stage 2 library is not loaded.")

        input_signal = np.asarray(signal, dtype=np.float64)
        expected_length = int(
            library.tadm_multistage_output_length(
                ctypes.c_size_t(input_signal.shape[0]),
                ctypes.c_size_t(self._config.num_stages),
            )
        )
        if expected_length <= 0:
            raise RuntimeError(
                "Failed to compute Stage 2 output length in C++ core API."
            )

        output = np.empty(expected_length, dtype=np.float64)
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
        written = int(
            library.tadm_multistage_process_block(
                self._handle,
                input_signal.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_size_t(input_signal.shape[0]),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_size_t(output.shape[0]),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
        )
        if written != output.shape[0]:
            message = error_buffer.value.decode("utf-8", errors="replace")
            if message == "":
                message = (
                    "C++ Stage 2 process_block returned unexpected output length: "
                    f"written={written}, expected={output.shape[0]}"
                )
            raise RuntimeError(message)
        return np.asarray(output, dtype=np.float64)

    def _create_handle(self) -> ctypes.c_void_p:
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
        library = self._lib
        if library is None:
            raise RuntimeError("C++ Stage 2 library is not loaded.")

        raw_handle = library.tadm_create_multistage_upsampler_from_dir(
            str(self._config.config_dir).encode("utf-8"),
            ctypes.c_size_t(self._config.num_stages),
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
        if raw_handle is None:
            message = error_buffer.value.decode("utf-8", errors="replace")
            if message == "":
                message = "Failed to create C++ Stage 2 upsampler handle."
            raise RuntimeError(message)
        return ctypes.c_void_p(raw_handle)

    def __enter__(self) -> CppStage2Upsampler:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


@lru_cache(maxsize=4)
def _load_capi_library(cpp_project_dir: Path, cpp_build_dir: Path) -> ctypes.CDLL:
    _build_capi_library(cpp_project_dir=cpp_project_dir, cpp_build_dir=cpp_build_dir)
    shared_library_path = _find_shared_library(cpp_build_dir)

    library = ctypes.CDLL(str(shared_library_path))
    library.tadm_create_multistage_upsampler_from_dir.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.tadm_create_multistage_upsampler_from_dir.restype = ctypes.c_void_p

    library.tadm_destroy_multistage_upsampler.argtypes = [ctypes.c_void_p]
    library.tadm_destroy_multistage_upsampler.restype = None

    library.tadm_multistage_output_length.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    library.tadm_multistage_output_length.restype = ctypes.c_size_t

    library.tadm_multistage_process_block.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.tadm_multistage_process_block.restype = ctypes.c_size_t
    return library


def _build_capi_library(cpp_project_dir: Path, cpp_build_dir: Path) -> None:
    project_dir = cpp_project_dir.resolve()
    build_dir = cpp_build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    _run_subprocess(
        ["cmake", "-S", str(project_dir), "-B", str(build_dir)],
        fail_message="Failed to configure C++ Stage 2 project.",
    )
    _run_subprocess(
        ["cmake", "--build", str(build_dir), "--target", "tadm_dsp_capi"],
        fail_message="Failed to build C++ Stage 2 C API library.",
    )


def _run_subprocess(command: list[str], fail_message: str) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{fail_message} Command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        stdout = exc.stdout.strip()
        details = stderr if stderr != "" else stdout
        raise RuntimeError(f"{fail_message} {details}") from exc


def _find_shared_library(cpp_build_dir: Path) -> Path:
    patterns = (
        "libtadm_dsp_capi.so",
        "libtadm_dsp_capi.dylib",
        "tadm_dsp_capi.dll",
    )
    for pattern in patterns:
        direct = cpp_build_dir / pattern
        if direct.exists():
            return direct

    matches = sorted(cpp_build_dir.rglob("*tadm_dsp_capi*.so"))
    if matches:
        return matches[0]
    matches = sorted(cpp_build_dir.rglob("*tadm_dsp_capi*.dylib"))
    if matches:
        return matches[0]
    matches = sorted(cpp_build_dir.rglob("*tadm_dsp_capi*.dll"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "C++ Stage 2 C API shared library not found in build directory: "
        f"{cpp_build_dir}"
    )


def _validate_signal(signal: np.ndarray) -> None:
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got {signal.ndim}D.")
    if signal.size == 0:
        raise ValueError("signal must not be empty.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must contain only finite values.")
