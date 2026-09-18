#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Shared Python dispatcher utilities for GEMM and grouped convolution.

Extracted from ctypes_utils.py (GEMM) + compile_grouped_conv_examples.py (grouped conv).
Both ctypes_utils.py and grouped_conv_utils.py import from here to
eliminate duplication.

Best-of-both:
  - Validation and auto-correction return typed objects (GEMM pattern)
  - Colors class with cross-platform ANSI handling (conv pattern)
  - Phased output helpers (conv pattern)
  - logging module instead of bare print() (shared improvement)
"""

import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ============================================================================
# Path Configuration
# ============================================================================


def get_dispatcher_root() -> Path:
    """Get the dispatcher root directory (parent of python/)."""
    return Path(__file__).parent.parent


def get_ck_root() -> Path:
    """Get the CK root directory (parent of dispatcher/)."""
    return get_dispatcher_root().parent


def get_build_dir() -> Path:
    """Get the build directory."""
    return get_dispatcher_root() / "build"


def get_generated_kernels_dir() -> Path:
    """Get the generated kernels directory."""
    return get_build_dir() / "generated_kernels"


def get_codegen_dir() -> Path:
    """Get the codegen scripts directory."""
    return get_dispatcher_root() / "codegen"


def _detect_gpu_arch_via_amd_smi() -> Optional[str]:
    """Best-effort arch via the shared amd-smi-first smi_utils wrapper.

    Single source of truth for the amd-smi bridge: the other dispatcher arch
    helpers (``gemm_utils``, ``ctypes_utils``, ``grouped_conv_utils``) import
    this rather than re-implementing the wrapper reach.

    Returns a ``gfxNNN`` string, or ``None`` if the wrapper is unavailable or
    neither amd-smi nor rocm-smi resolves an arch (callers fall back to
    rocminfo).
    """
    try:
        import sys as _sys
        _common = Path(__file__).resolve().parents[2] / "tile_engine" / "ops" / "common"
        if str(_common) not in _sys.path:
            _sys.path.insert(0, str(_common))
        import smi_utils  # noqa: E402
        return smi_utils.detect_gpu_arch()
    except Exception:  # noqa: BLE001 - optional wrapper + external CLI; degrade to rocminfo
        return None


def detect_gpu_arch(fallback: str = "gfx942") -> str:
    """Detect the GPU architecture, preferring amd-smi (smi_utils wrapper).

    Falls back to rocminfo, then to the given default.
    """
    arch = _detect_gpu_arch_via_amd_smi()
    if arch:
        return arch
    import subprocess

    try:
        out = subprocess.check_output(
            ["rocminfo"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if "Name:" in line and "gfx" in line:
                return line.split()[-1].strip()
    except Exception:
        pass
    return fallback


# ============================================================================
# fp8 / bf8 encoding format per architecture
# ============================================================================
#
# CK has two incompatible 8-bit float encodings and picks between them per arch:
#
#   OCP  (gfx950, gfx12): e4m3fn   / e5m2    -- exponent bias 7 / 15
#   FNUZ (everything else, notably gfx942): e4m3fnuz / e5m2fnuz -- bias 8 / 16
#
# include/ck_tile/core/config.hpp:361-371 resolves this at compile time, but only
# the *device* pass sees __gfx950__; the host pass of the very same header falls
# back to FNUZ. So a host-side reference encoder that guesses from the header ends
# up disagreeing with the kernel it is validating. Every caller must therefore
# decide from the target arch string, and pass -DCK_TILE_USE_OCP_FP8 explicitly so
# both compiler passes agree. These helpers are the single source of truth for
# both halves of that contract.


def normalize_arch(arch: Optional[str]) -> str:
    """Lowercase `arch` and strip the target-feature suffix, if any.

    rocminfo, hipcc and the HSA runtime all hand back full target triples --
    ``gfx950:sramecc+:xnack-`` -- while configs, CI parameters and these helpers
    are written in terms of the bare name.  Every arch predicate below matches on
    a prefix, so a triple happens to survive unnormalized, but anything carrying
    a leading component (an ``amdgcn-amd-amdhsa--gfx950`` offload target, say)
    silently falls through to the FNUZ default.  That miss drops
    ``-DCK_USE_OCP_FP8``, and host and device then disagree on the fp8 encoding
    with no diagnostic at all -- the kernel builds and returns wrong numbers.

    Normalizing in one place, up front, is what keeps that from depending on
    which spelling of the arch a given caller happened to be handed.
    """
    a = (arch or "").lower().strip()
    # Feature suffix: gfx950:sramecc+:xnack- -> gfx950
    a = a.split(":", 1)[0]
    # Offload-target prefix: amdgcn-amd-amdhsa--gfx950 -> gfx950
    idx = a.rfind("gfx")
    return a[idx:] if idx > 0 else a


def fp8_uses_ocp(arch: Optional[str]) -> bool:
    """True iff `arch` uses the OCP fp8/bf8 encoding rather than FNUZ.

    Mirrors the __gfx950__ / __gfx12__ test in include/ck_tile/core/config.hpp.
    Use this to select the host-side codec (ml_dtypes.float8_e4m3fn vs
    float8_e4m3fnuz) so it matches the bytes the kernel actually produces.

    Accepts bare names and full target triples alike; see `normalize_arch`.
    """
    a = normalize_arch(arch)
    return a.startswith("gfx950") or a.startswith("gfx12")


def ocp_arch_defines(arch: Optional[str]) -> List[str]:
    """hipcc defines that pin the fp8/bf8 encoding for `arch`.

    Returned for OCP archs only; FNUZ is the default for both compiler passes and
    needs no define. Passing these makes the host pass agree with the device pass
    instead of silently falling back to FNUZ.
    """
    if not fp8_uses_ocp(arch):
        return []
    return ["-DCK_USE_OCP_FP8", "-DCK_TILE_USE_OCP_FP8"]


def arch_feature_defines(arch: Optional[str]) -> List[str]:
    """`ocp_arch_defines` plus the per-arch feature-enablement defines.

    The top-level CMakeLists.txt (:456-512) sets these for a normal build; they
    are absent in the standalone hipcc JIT path, so any bridge that compiles a
    kernel out-of-tree has to re-supply them.  This mirrors that block for the
    arches the dispatcher targets:

        gfx950  -> CK_USE_NATIVE_MX_SUPPORT, CK_GFX950_SUPPORT
        gfx1250 -> CK_USE_GFX1250, CK_USE_NATIVE_MX_SUPPORT, CK_GFX1250_SUPPORT
        gfx11/gfx12 -> CK_TILE_USE_WMMA=1  (gfx12 also CK_GFX12_SUPPORT)

    CK_TILE_USE_WMMA is the one that must be passed even when it is 0: CMake
    always defines it (:480), and the JIT path leaving it undefined only happens
    to work because the preprocessor reads an undefined identifier as 0, which is
    the right answer on gfx942/gfx950 and the wrong one on every WMMA part.

    CK_USE_GFX950 is deliberately *not* emitted -- CMake sets it, but it is read
    only by the legacy ck/library conv instances, never by ck_tile, so it is
    dead weight on this path.

    On the warp tile: these defines do select the branch of
    tile_gemm_shape.hpp get_k_warp_tile() (CK_TILE_USE_WMMA outermost, then
    CK_USE_GFX1250 / CK_GFX950_SUPPORT), but the emitted kernels never *call*
    that function -- codegen writes a literal warp_tile_k, mirroring it in Python
    via codegen_common.fp8_warp_tile_k_for_arch.  The hazard is therefore a
    mismatch between that literal and the warp-gemm the target arch actually has:
    there is no 16x16x128 fp8 warp-gemm on gfx942, and asking for one compiles
    cleanly and returns all zeros.  Keep warp_tile_k arch-aware alongside these
    defines, and derive it from the one helper rather than a local copy.
    """
    a = normalize_arch(arch)
    defines = ocp_arch_defines(arch)
    if a.startswith("gfx950"):
        defines = defines + ["-DCK_USE_NATIVE_MX_SUPPORT", "-DCK_GFX950_SUPPORT"]
    elif a.startswith("gfx11") or a.startswith("gfx12"):
        defines = defines + ["-DCK_TILE_USE_WMMA=1"]
        if a.startswith("gfx12"):
            defines = defines + ["-DCK_GFX12_SUPPORT"]
        if a.startswith("gfx1250"):
            defines = defines + [
                "-DCK_USE_GFX1250",
                "-DCK_USE_NATIVE_MX_SUPPORT",
                "-DCK_GFX1250_SUPPORT",
            ]
    return defines


# ============================================================================
# Architecture Filter Data
# ============================================================================

_arch_data_cache: Optional[Dict[str, Any]] = None



def get_arch_filter_data() -> Dict[str, Any]:
    """Load arch filter data from arch_specs_generated if available.

    Returns dict with keys: trait_unsupported, warp_combos,
    warp_tile_combos, supported_archs.
    """
    global _arch_data_cache
    if _arch_data_cache is not None:
        return _arch_data_cache

    codegen_dir = get_dispatcher_root() / "codegen"
    sys.path.insert(0, str(codegen_dir))

    try:
        from arch_specs_generated import (
            TRAIT_UNSUPPORTED_COMBINATIONS,
            WARP_SUPPORTED_COMBINATIONS,
            WARP_TILE_SUPPORTED_COMBINATIONS,
            get_supported_archs,
        )

        _arch_data_cache = {
            "trait_unsupported": TRAIT_UNSUPPORTED_COMBINATIONS,
            "warp_combos": WARP_SUPPORTED_COMBINATIONS,
            "warp_tile_combos": WARP_TILE_SUPPORTED_COMBINATIONS,
            "supported_archs": get_supported_archs(),
        }
    except ImportError:
        _arch_data_cache = {
            "trait_unsupported": {
                ("compv3", "cshuffle", "interwave"),
                ("compv3", "default", "interwave"),
                ("compv4", "cshuffle", "interwave"),
                ("compv4", "default", "interwave"),
            },
            "warp_combos": {
                "gfx942": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
                "gfx90a": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
            },
            "warp_tile_combos": {
                "gfx942": {"fp16_fp16_fp32": [[16, 16, 16], [32, 32, 16]]},
                "gfx90a": {"fp16_fp16_fp32": [[16, 16, 16], [32, 32, 16]]},
            },
            "supported_archs": ["gfx90a", "gfx942", "gfx950"],
        }

    return _arch_data_cache


# ============================================================================
# Validation Result
# ============================================================================


@dataclass
class ValidationResultBase:
    """Result of kernel config validation (shared base for GEMM and conv)."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggested_fixes: Dict[str, Any] = field(default_factory=dict)

    def print_result(self, indent: str = "  "):
        if self.is_valid:
            print(f"{indent}OK Configuration valid")
        else:
            print(f"{indent}WARNING Configuration has issues:")
            for err in self.errors:
                print(f"{indent}  - {err}")
        if self.warnings:
            for warn in self.warnings:
                print(f"{indent}  Warning: {warn}")
        if self.suggested_fixes:
            print(f"{indent}  Suggested fixes:")
            for key, val in self.suggested_fixes.items():
                print(f"{indent}    {key}: {val}")


# ============================================================================
# Validation Helpers
# ============================================================================


def validate_wave_config(wave_cfg: List[int], arch: str) -> Tuple[bool, str]:
    """Validate a [wave_m, wave_n, wave_k] config for *arch*.

    Returns (is_valid, error_message). Empty string on success.
    """
    data = get_arch_filter_data()
    valid_waves = data["warp_combos"].get(arch, [[2, 2, 1]])
    if wave_cfg in valid_waves:
        return True, ""
    valid_str = ", ".join(f"[{c[0]},{c[1]},{c[2]}]" for c in valid_waves)
    return (
        False,
        f"Unsupported wave configuration {wave_cfg} for {arch}. "
        f"Valid wave configs: {valid_str}",
    )


def validate_warp_tile_config(
    warp_cfg: List[int], arch: str, dtype: str
) -> Tuple[bool, str]:
    """Validate a [warp_m, warp_n, warp_k] config for *arch*/*dtype*.

    Returns (is_valid, error_message). Empty string on success.
    """
    data = get_arch_filter_data()
    acc = "int32" if dtype == "int8" else "fp32"
    dtype_key = f"{dtype}_{dtype}_{acc}"
    valid_tiles = (
        data["warp_tile_combos"]
        .get(arch, {})
        .get(dtype_key, [[32, 32, 16], [16, 16, 16]])
    )
    if warp_cfg in valid_tiles:
        return True, ""
    valid_str = ", ".join(f"[{c[0]},{c[1]},{c[2]}]" for c in valid_tiles[:5])
    return (
        False,
        f"Unsupported warp tile {warp_cfg} for {arch}/{dtype}. "
        f"Valid warp tiles: {valid_str}",
    )


def validate_trait_combo(
    pipeline: str, epilogue: str, scheduler: str
) -> Tuple[bool, str]:
    """Validate a (pipeline, epilogue, scheduler) combination.

    Returns (is_valid, error_message). Empty string on success.
    """
    data = get_arch_filter_data()
    combo = (pipeline, epilogue, scheduler)
    if combo in data["trait_unsupported"]:
        return (
            False,
            f"Unsupported trait combination: pipeline={pipeline}, "
            f"epilogue={epilogue}, scheduler={scheduler}",
        )
    return True, ""


# ============================================================================
# Auto-Correction Helpers
# ============================================================================


def auto_correct_wave(wave_cfg: List[int], arch: str) -> List[int]:
    """Return the first valid wave config for *arch*.

    If *wave_cfg* is already valid, returns it unchanged.
    """
    data = get_arch_filter_data()
    valid_waves = data["warp_combos"].get(arch, [[2, 2, 1]])
    if wave_cfg in valid_waves:
        return wave_cfg
    return valid_waves[0] if valid_waves else [2, 2, 1]


def auto_correct_trait(pipeline: str, scheduler: str) -> Tuple[str, str]:
    """Return a corrected (pipeline, scheduler) pair.

    If the compute pipeline doesn't support interwave, switch to intrawave.
    """
    data = get_arch_filter_data()
    for epilogue in ("cshuffle", "default"):
        if (pipeline, epilogue, scheduler) in data["trait_unsupported"]:
            return pipeline, "intrawave"
    return pipeline, scheduler


# ============================================================================
# Colors (adopted from compile_grouped_conv_examples.py -- cross-platform)
# ============================================================================


class Colors:
    """Cross-platform ANSI color support.

    Respects sys.platform (no ANSI on Windows) and isatty() check so
    piped/redirected output stays clean.
    """

    _GREEN = "\033[0;32m"
    _YELLOW = "\033[1;33m"
    _RED = "\033[0;31m"
    _CYAN = "\033[0;36m"
    _BOLD = "\033[1m"
    _NC = "\033[0m"

    @classmethod
    def _use_color(cls) -> bool:
        return (
            sys.platform != "win32"
            and hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
        )

    @classmethod
    def green(cls, text: str) -> str:
        if cls._use_color():
            return f"{cls._GREEN}{text}{cls._NC}"
        return text

    @classmethod
    def red(cls, text: str) -> str:
        if cls._use_color():
            return f"{cls._RED}{text}{cls._NC}"
        return text

    @classmethod
    def yellow(cls, text: str) -> str:
        if cls._use_color():
            return f"{cls._YELLOW}{text}{cls._NC}"
        return text

    @classmethod
    def cyan(cls, text: str) -> str:
        if cls._use_color():
            return f"{cls._CYAN}{text}{cls._NC}"
        return text

    @classmethod
    def bold(cls, text: str) -> str:
        if cls._use_color():
            return f"{cls._BOLD}{text}{cls._NC}"
        return text


# ============================================================================
# Phased Output Helpers
# ============================================================================


def print_phase(number: int, description: str) -> None:
    """Print a phase header (e.g. 'Phase 1: Codegen')."""
    print(f"\n{'=' * 60}")
    print(f"  Phase {number}: {description}")
    print(f"{'=' * 60}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"  OK {Colors.green(message)}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"  FAIL {Colors.red(message)}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"  {Colors.cyan(message)}")


# ============================================================================
# Cleanup Helpers
# ============================================================================


def cleanup_generated_kernels(gen_dir: Optional[Path] = None) -> None:
    """Remove generated kernel directory if it exists."""
    if gen_dir is None:
        gen_dir = get_generated_kernels_dir()
    if gen_dir.exists():
        shutil.rmtree(gen_dir, ignore_errors=True)
        log.info("Cleaned up generated kernels at %s", gen_dir)


# ============================================================================
# Tool Helpers
# ============================================================================


def find_hipcc() -> Optional[str]:
    """Find the hipcc compiler."""
    import os

    candidates = [
        os.environ.get("HIPCC"),
        "/opt/rocm/bin/hipcc",
        shutil.which("hipcc"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None
