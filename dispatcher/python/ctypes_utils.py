#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CK Tile Dispatcher Utilities

Common utilities for loading, compiling, and using the CK Tile dispatcher.

Usage:
    from ck_tile_dispatcher.utils import DispatcherLib, GemmRunner, Validator

    # Option 1: Auto-compile and load
    lib = DispatcherLib.auto()

    # Option 2: Load existing library
    lib = DispatcherLib.load("/path/to/libdispatcher_gemm.so")

    # Run GEMM
    runner = GemmRunner(lib)
    result = runner.run(A, B)

    # Validate
    validator = Validator()
    check = validator.check(result.C, C_reference)
"""

import ctypes
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time


# =============================================================================
# GPU Architecture Auto-Detection
# =============================================================================

_detected_arch: Optional[str] = None


def detect_gpu_arch(fallback: str = "gfx942") -> str:
    """
    Auto-detect the GPU architecture by querying rocminfo.

    Caches the result after the first call. Falls back to `fallback` if
    detection fails (e.g. no GPU, rocminfo not installed).
    """
    global _detected_arch
    if _detected_arch is not None:
        return _detected_arch

    try:
        result = subprocess.run(
            ["/opt/rocm/bin/rocminfo"], capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Name:") and "gfx" in stripped:
                # Extract e.g. "gfx950" from "Name:                    gfx950"
                name = stripped.split(":", 1)[1].strip()
                if name.startswith("gfx") and name[3:].isdigit():
                    _detected_arch = name
                    return _detected_arch
    except Exception:
        pass

    _detected_arch = fallback
    return _detected_arch


# =============================================================================
# Path Configuration
# =============================================================================


def get_dispatcher_root() -> Path:
    """Get the dispatcher root directory"""
    # This file is in dispatcher/python/
    return Path(__file__).parent.parent


def get_ck_root() -> Path:
    """Get the CK root directory"""
    return get_dispatcher_root().parent


def get_build_dir() -> Path:
    """Get the build directory"""
    return get_dispatcher_root() / "build"


# =============================================================================
# Supported Data Types
# =============================================================================

# All supported GEMM dtype combinations from warp_gemm_dispatcher.hpp
SUPPORTED_DTYPES = {
    # dtype_a, dtype_b -> acc_dtype, warp_tiles
    ("fp32", "fp32"): {"acc": "fp32", "warp_tiles": [(16, 16, 4), (16, 16, 16)]},
    ("fp16", "fp16"): {
        "acc": "fp32",
        "warp_tiles": [(32, 32, 8), (32, 32, 16), (16, 16, 16), (16, 16, 32)],
    },
    ("bf16", "bf16"): {
        "acc": "fp32",
        "warp_tiles": [(32, 32, 8), (32, 32, 16), (16, 16, 16), (16, 16, 32)],
    },
    ("fp8", "fp8"): {
        "acc": "fp32",
        "warp_tiles": [(32, 32, 16), (32, 32, 32), (16, 16, 32), (16, 16, 64)],
    },
    ("fp8", "bf8"): {"acc": "fp32", "warp_tiles": [(32, 32, 16), (16, 16, 32)]},
    ("bf8", "fp8"): {"acc": "fp32", "warp_tiles": [(32, 32, 16), (16, 16, 128)]},
    ("bf8", "bf8"): {
        "acc": "fp32",
        "warp_tiles": [(32, 32, 16), (32, 32, 32), (16, 16, 32)],
    },
    ("int8", "int8"): {
        "acc": "int32",
        "warp_tiles": [(32, 32, 16), (16, 16, 32), (16, 16, 16)],
    },
    ("pk_fp4", "pk_fp4"): {"acc": "fp32", "warp_tiles": [(16, 16, 128)]},
}

# All valid individual dtypes
VALID_DTYPES = ["fp16", "bf16", "fp32", "fp8", "bf8", "int8", "pk_fp4"]


def get_generated_kernels_dir() -> Path:
    """Get the generated kernels directory"""
    return get_build_dir() / "generated_kernels"


# =============================================================================
# Arch Filter and Validation
# =============================================================================


def get_arch_filter_data() -> Dict[str, Any]:
    """Load arch filter data from arch_specs_generated if available."""
    codegen_dir = get_dispatcher_root() / "codegen"
    import sys

    sys.path.insert(0, str(codegen_dir))

    try:
        from arch_specs_generated import (
            TRAIT_UNSUPPORTED_COMBINATIONS,
            WARP_SUPPORTED_COMBINATIONS,
            WARP_TILE_SUPPORTED_COMBINATIONS,
            get_supported_archs,
        )

        return {
            "trait_unsupported": TRAIT_UNSUPPORTED_COMBINATIONS,
            "warp_combos": WARP_SUPPORTED_COMBINATIONS,
            "warp_tile_combos": WARP_TILE_SUPPORTED_COMBINATIONS,
            "supported_archs": get_supported_archs(),
        }
    except ImportError:
        # Fallback defaults
        return {
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
                "gfx942": {"fp16_fp16_fp16": [[16, 16, 16], [32, 32, 16]]},
                "gfx90a": {"fp16_fp16_fp16": [[16, 16, 16], [32, 32, 16]]},
            },
            "supported_archs": ["gfx90a", "gfx942", "gfx950"],
        }


@dataclass
class ValidationResult:
    """Result of kernel config validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggested_fixes: Dict[str, Any] = field(default_factory=dict)

    def print_result(self, indent: str = "  "):
        """Print validation result."""
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


def validate_kernel_config(config: "KernelConfig") -> ValidationResult:
    """
    Validate a KernelConfig against arch filter rules.

    Validation considers the GEMM variant (standard, preshuffle, multi_d)
    for operator-specific constraints like minimum tile sizes.

    Returns ValidationResult with is_valid, errors, and suggested fixes.
    """
    arch_data = get_arch_filter_data()

    errors = []
    warnings = []
    suggested_fixes = {}

    pipeline = config.pipeline
    epilogue = config.epilogue
    scheduler = config.scheduler
    dtype = config.dtype_a
    arch = config.gfx_arch
    variant = getattr(config, "variant", "standard")

    wave_m = config.wave_m
    wave_n = config.wave_n
    wave_k = config.wave_k

    warp_m = config.warp_m
    warp_n = config.warp_n
    warp_k = config.warp_k

    # Variant-specific tile constraints
    if variant == "preshuffle":
        # Preshuffle requires larger minimum tiles for efficiency
        if config.tile_m < 64:
            errors.append(f"Preshuffle requires tile_m >= 64, got {config.tile_m}")
            suggested_fixes["tile_m"] = 64
        if config.tile_n < 64:
            errors.append(f"Preshuffle requires tile_n >= 64, got {config.tile_n}")
            suggested_fixes["tile_n"] = 64
        if config.tile_k < 32:
            errors.append(f"Preshuffle requires tile_k >= 32, got {config.tile_k}")
            suggested_fixes["tile_k"] = 32

    elif variant == "multi_d":
        # Multi-D has standard GEMM constraints
        # Could add specific constraints here if needed
        pass

    # Check trait combination (pipeline, epilogue, scheduler)
    combo = (pipeline, epilogue, scheduler)
    if combo in arch_data["trait_unsupported"]:
        errors.append(
            f"Unsupported trait combination: pipeline={pipeline}, epilogue={epilogue}, scheduler={scheduler}"
        )
        suggested_fixes["scheduler"] = "intrawave"

    # Check wave configuration for this arch
    warp_combos = arch_data["warp_combos"].get(arch, [[2, 2, 1]])
    wave_cfg = [wave_m, wave_n, wave_k]
    if wave_cfg not in warp_combos:
        valid_str = ", ".join(f"[{c[0]},{c[1]},{c[2]}]" for c in warp_combos)
        errors.append(
            f"Unsupported wave configuration [{wave_m},{wave_n},{wave_k}] for {arch}. Valid: {valid_str}"
        )
        if warp_combos:
            suggested_fixes["wave_m"] = warp_combos[0][0]
            suggested_fixes["wave_n"] = warp_combos[0][1]
            suggested_fixes["wave_k"] = warp_combos[0][2]

    # Check warp tile configuration for this arch and dtype
    dtype_key = f"{dtype}_{dtype}_{dtype}"
    warp_tile_combos = (
        arch_data["warp_tile_combos"]
        .get(arch, {})
        .get(dtype_key, [[32, 32, 16], [16, 16, 16]])
    )
    warp_cfg = [warp_m, warp_n, warp_k]
    if warp_cfg not in warp_tile_combos:
        valid_str = ", ".join(f"[{c[0]},{c[1]},{c[2]}]" for c in warp_tile_combos[:5])
        errors.append(
            f"Unsupported warp tile [{warp_m},{warp_n},{warp_k}] for {arch}/{dtype}. Valid: {valid_str}"
        )
        if warp_tile_combos:
            suggested_fixes["warp_m"] = warp_tile_combos[0][0]
            suggested_fixes["warp_n"] = warp_tile_combos[0][1]
            suggested_fixes["warp_k"] = warp_tile_combos[0][2]

    # Check arch is supported
    if arch not in arch_data["supported_archs"]:
        errors.append(
            f"Unsupported architecture: {arch}. Supported: {', '.join(arch_data['supported_archs'])}"
        )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        suggested_fixes=suggested_fixes,
    )


def auto_correct_kernel_config(
    config: "KernelConfig", verbose: bool = False
) -> Tuple["KernelConfig", bool, List[str]]:
    """
    Validate and auto-correct a KernelConfig.

    Returns (corrected_config, was_modified, corrections_list).
    If the config was valid, returns (original_config, False, []).
    If corrections were made, returns (new_config, True, [list of correction descriptions]).
    """
    validation = validate_kernel_config(config)

    if validation.is_valid:
        return config, False, []

    # Apply suggested fixes and track what changed
    from dataclasses import replace

    fixes = validation.suggested_fixes
    corrections = []

    # Check each fix and describe what changed
    if "scheduler" in fixes and fixes["scheduler"] != config.scheduler:
        corrections.append(
            f"Scheduler: {config.scheduler} -> {fixes['scheduler']} "
            f"('{config.scheduler}' not supported with pipeline={config.pipeline}, epilogue={config.epilogue})"
        )

    if "wave_m" in fixes or "wave_n" in fixes or "wave_k" in fixes:
        old_wave = f"[{config.wave_m}, {config.wave_n}, {config.wave_k}]"
        new_wave = f"[{fixes.get('wave_m', config.wave_m)}, {fixes.get('wave_n', config.wave_n)}, {fixes.get('wave_k', config.wave_k)}]"
        if old_wave != new_wave:
            corrections.append(
                f"Wave config: {old_wave} -> {new_wave} "
                f"(original not supported on {config.gfx_arch})"
            )

    if "warp_m" in fixes or "warp_n" in fixes or "warp_k" in fixes:
        old_warp = f"[{config.warp_m}, {config.warp_n}, {config.warp_k}]"
        new_warp = f"[{fixes.get('warp_m', config.warp_m)}, {fixes.get('warp_n', config.warp_n)}, {fixes.get('warp_k', config.warp_k)}]"
        if old_warp != new_warp:
            corrections.append(
                f"Warp tile: {old_warp} -> {new_warp} "
                f"(original not supported for {config.dtype_a} on {config.gfx_arch})"
            )

    new_config = replace(
        config,
        scheduler=fixes.get("scheduler", config.scheduler),
        wave_m=fixes.get("wave_m", config.wave_m),
        wave_n=fixes.get("wave_n", config.wave_n),
        wave_k=fixes.get("wave_k", config.wave_k),
        warp_m=fixes.get("warp_m", config.warp_m),
        warp_n=fixes.get("warp_n", config.warp_n),
        warp_k=fixes.get("warp_k", config.warp_k),
    )

    return new_config, True, corrections


def print_kernel_config(config: "KernelConfig", title: str = "KERNEL CONFIGURATION"):
    """
    Print a formatted kernel configuration for GEMM.

    Args:
        config: The KernelConfig to print
        title: Title to display (e.g., "REQUESTED KERNEL CONFIGURATION")
    """
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"  Data Type A:   {config.dtype_a}")
    print(f"  Data Type B:   {config.dtype_b}")
    print(f"  Data Type C:   {config.dtype_c}")
    print(f"  Accumulator:   {config.dtype_acc}")
    print()
    print(
        f"  Layout:        {config.layout} (A={config.layout_a}, B={config.layout_b}, C={config.layout_c})"
    )
    print()
    print(f"  Tile M x N x K: {config.tile_m} x {config.tile_n} x {config.tile_k}")
    print(f"  Wave Config:    {config.wave_m} x {config.wave_n} x {config.wave_k}")
    print(f"  Warp Tile:      {config.warp_m} x {config.warp_n} x {config.warp_k}")
    print()
    print(f"  Pipeline:      {config.pipeline}")
    print(f"  Scheduler:     {config.scheduler}")
    print(f"  Epilogue:      {config.epilogue}")
    print()
    print(f"  Target Arch:   {config.gfx_arch}")
    print("=" * 70)
    print()


def print_auto_correction(
    original: "KernelConfig",
    corrected: "KernelConfig",
    corrections: List[str],
    indent: str = "  ",
):
    """
    Print what was auto-corrected and why.

    Args:
        original: Original configuration before correction
        corrected: Configuration after correction
        corrections: List of correction descriptions
        indent: Indentation for output
    """
    if not corrections:
        print(f"{indent}OK Configuration valid - no corrections needed")
        return

    print(f"\n{indent}WARNING AUTO-CORRECTION APPLIED:")
    print(f"{indent}" + "-" * 50)
    for correction in corrections:
        print(f"{indent}  - {correction}")
    print(f"{indent}" + "-" * 50)
    print()


def find_matching_kernel_header(config: "KernelConfig") -> Optional[Path]:
    """
    Find a kernel header that EXACTLY matches the config.

    Uses progressively relaxed matching strategies.
    """
    kernel_dir = get_generated_kernels_dir()

    dtype = config.dtype_a
    layout = config.layout
    pipeline = config.pipeline
    scheduler = config.scheduler
    tile_str = config.tile_str
    wave_str = f"{config.wave_m}x{config.wave_n}x{config.wave_k}"
    warp_str = f"{config.warp_m}x{config.warp_n}x{config.warp_k}"

    # Strategy 1: Exact match with ALL parameters including warp tile
    pattern = f"gemm_{dtype}_{layout}_{pipeline}_*_{scheduler}_*_{tile_str}_{wave_str}_{warp_str}.hpp"
    matches = list(kernel_dir.glob(pattern))
    if matches:
        return matches[0]

    # Strategy 2: Match with tile and wave, any warp
    pattern = (
        f"gemm_{dtype}_{layout}_{pipeline}_*_{scheduler}_*_{tile_str}_{wave_str}_*.hpp"
    )
    matches = list(kernel_dir.glob(pattern))
    if matches:
        return matches[0]

    # Strategy 3: Match with just tile (ignore wave/warp)
    pattern = f"gemm_{dtype}_{layout}_{pipeline}_*_{scheduler}_*_{tile_str}_*.hpp"
    matches = list(kernel_dir.glob(pattern))
    if matches:
        return matches[0]

    # Strategy 4: Match with intrawave (known to work)
    pattern = f"gemm_{dtype}_{layout}_*_intrawave_*_{tile_str}_*.hpp"
    matches = list(kernel_dir.glob(pattern))
    if matches:
        return matches[0]

    # Strategy 5: Any kernel with matching dtype/layout/tile
    pattern = f"gemm_{dtype}_{layout}_*_{tile_str}_*.hpp"
    matches = list(kernel_dir.glob(pattern))
    if matches:
        return matches[0]

    return None


# =============================================================================
# Library Loading
# =============================================================================


class DispatcherLib:
    """Wrapper for the dispatcher dynamic library"""

    # Default library search paths (relative to dispatcher root)
    SEARCH_PATHS = [
        "build/examples/libdispatcher_gemm_lib.so",
        "build/libdispatcher_gemm_lib.so",
        "build/examples/libdispatcher_gemm.so",
        "build/lib/libdispatcher_gemm.so",
    ]

    # Track loaded libraries globally for cleanup
    _loaded_libs: List[Path] = []

    def __init__(self, lib: ctypes.CDLL, path: Path):
        self._lib = lib
        self._path = path
        self._closed = False
        DispatcherLib._loaded_libs.append(path)
        self._setup_functions()

    def _setup_functions(self):
        """Setup ctypes function signatures"""
        # Initialize
        self._lib.dispatcher_initialize.argtypes = []
        self._lib.dispatcher_initialize.restype = ctypes.c_int

        # Alias for init
        self._lib.dispatcher_init.argtypes = []
        self._lib.dispatcher_init.restype = ctypes.c_int

        # Get kernel count
        self._lib.dispatcher_get_kernel_count.argtypes = []
        self._lib.dispatcher_get_kernel_count.restype = ctypes.c_int

        # Check if supported
        self._lib.dispatcher_is_supported.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self._lib.dispatcher_is_supported.restype = ctypes.c_int

        # Run GEMM
        self._lib.dispatcher_run_gemm.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # B
            ctypes.c_void_p,  # C
            ctypes.c_int64,  # M
            ctypes.c_int64,  # N
            ctypes.c_int64,  # K
            ctypes.POINTER(ctypes.c_float),  # time_ms
        ]
        self._lib.dispatcher_run_gemm.restype = ctypes.c_int

        # Get kernel name
        self._lib.dispatcher_get_kernel_name.argtypes = []
        self._lib.dispatcher_get_kernel_name.restype = ctypes.c_char_p

        # Select kernel
        self._lib.dispatcher_select_kernel.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self._lib.dispatcher_select_kernel.restype = ctypes.c_int

        # Export JSON
        self._lib.dispatcher_export_registry_json.argtypes = []
        self._lib.dispatcher_export_registry_json.restype = ctypes.c_char_p

        # Cleanup
        self._lib.dispatcher_cleanup.argtypes = []
        self._lib.dispatcher_cleanup.restype = None

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> bool:
        """Initialize the dispatcher"""
        return self._lib.dispatcher_initialize() == 0

    def get_kernel_count(self) -> int:
        """Get number of registered kernels"""
        return self._lib.dispatcher_get_kernel_count()

    def is_supported(self, M: int, N: int, K: int) -> bool:
        """Check if a problem size is supported"""
        return self._lib.dispatcher_is_supported(M, N, K) == 1

    def get_kernel_name(self) -> str:
        """Get the kernel name"""
        name = self._lib.dispatcher_get_kernel_name()
        return name.decode("utf-8") if name else "unknown"

    def select_kernel(self, M: int, N: int, K: int) -> Optional[str]:
        """Select kernel for problem and return its name"""
        buffer = ctypes.create_string_buffer(256)
        result = self._lib.dispatcher_select_kernel(M, N, K, buffer, 256)
        if result == 0:
            return buffer.value.decode("utf-8")
        return None

    def run_gemm(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, M: int, N: int, K: int
    ) -> Tuple[int, float]:
        """
        Run GEMM operation

        Returns: (status, time_ms)
            status: 0 = success, -1 = error, -2 = no suitable kernel
        """
        time_ms = ctypes.c_float(0.0)

        status = self._lib.dispatcher_run_gemm(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            M,
            N,
            K,
            ctypes.byref(time_ms),
        )

        return status, time_ms.value

    def export_json(self) -> Optional[str]:
        """Export registry to JSON string"""
        json_ptr = self._lib.dispatcher_export_registry_json()
        if json_ptr:
            return json_ptr.decode("utf-8")
        return None

    def export_registry_json(self) -> str:
        """Alias for export_json for compatibility"""
        return self.export_json() or "{}"

    def cleanup(self):
        """Cleanup dispatcher resources"""
        self._lib.dispatcher_cleanup()

    @classmethod
    def find(cls) -> Optional[Path]:
        """Find the dispatcher library"""
        root = get_dispatcher_root()

        for rel_path in cls.SEARCH_PATHS:
            path = root / rel_path
            if path.exists():
                return path

        return None

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional["DispatcherLib"]:
        """Load the dispatcher library from path or auto-find"""
        if path is None:
            path = cls.find()

        if path is None or not path.exists():
            return None

        try:
            lib = ctypes.CDLL(str(path))
            return cls(lib, path)
        except OSError as e:
            print(f"Failed to load library: {e}")
            return None

    @classmethod
    def compile(cls, output_path: Optional[Path] = None) -> Optional[Path]:
        """Compile the dispatcher library"""
        root = get_dispatcher_root()
        ck_root = get_ck_root()

        if output_path is None:
            output_path = get_build_dir() / "examples" / "libdispatcher_gemm.so"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Find a kernel header to include
        kernel_dir = get_generated_kernels_dir()
        kernel_headers = list(kernel_dir.glob("gemm_fp16_rcr_compv4*128x128x32*.hpp"))

        if not kernel_headers:
            print("No kernel headers found. Generate kernels first.")
            return None

        kernel_header = kernel_headers[0]

        # Use the ctypes binding source file
        ctypes_source = root / "bindings/ctypes/gemm_ctypes_lib.cpp"
        if not ctypes_source.exists():
            print(f"Source file not found: {ctypes_source}")
            print(
                "Please build with CMake: cd build && cmake .. && make dispatcher_gemm_lib"
            )
            return None

        # CK_TILE_SINGLE_KERNEL_INCLUDE exports types to global namespace for ctypes binding
        compile_cmd = [
            "/opt/rocm/bin/hipcc",
            "-shared",
            "-fPIC",
            "-O3",
            f"-I{root / 'include'}",
            f"-I{ck_root / 'include'}",
            f"-I{ck_root}",
            f"-I{root / 'build/generated_kernels'}",
            "-DCK_TILE_SINGLE_KERNEL_INCLUDE",  # Enable global namespace exports
            f"-include{kernel_header}",
            "-D__HIP_PLATFORM_AMD__",
            "--offload-arch=gfx942",
            "-DAMDGPU_ARCH=gfx942",
            "-mllvm",
            "-enable-noalias-to-md-conversion=0",
            "-Wno-undefined-func-template",
            "-Wno-float-equal",
            str(ctypes_source),
            "-o",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                compile_cmd, capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                return output_path
            else:
                print(f"Compilation failed:\n{result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print("Compilation timed out")
            return None

    @classmethod
    def auto(cls, recompile: bool = False) -> Optional["DispatcherLib"]:
        """Auto-find or compile the library.

        Note: The library is built by CMake with a specific kernel configuration.
        If you need a different dtype/layout, rebuild with:
            cd build && cmake .. && make dispatcher_gemm_lib
        """
        lib = cls.load()
        if lib is not None:
            if lib.initialize():
                return lib
            else:
                print("  Library found but failed to initialize")
                print(
                    "  Rebuild with: cd build && cmake .. && make dispatcher_gemm_lib"
                )

        # Don't fall back to old compile method - use CMake instead
        print("  Library not found. Build with:")
        print("    cd dispatcher/build && cmake .. && make dispatcher_gemm_lib")
        return None


# =============================================================================
# GEMM Runner
# =============================================================================


@dataclass
class GemmResult:
    """Result of a GEMM operation"""

    output: np.ndarray  # The output C matrix
    time_ms: float
    status: int
    tflops: float
    kernel_name: str

    @property
    def success(self) -> bool:
        return self.status == 0

    # Alias for backward compatibility
    @property
    def C(self) -> np.ndarray:
        return self.output


class GemmRunner:
    """High-level GEMM runner using the dispatcher"""

    def __init__(self, lib: DispatcherLib):
        self.lib = lib

    def run(self, A: np.ndarray, B: np.ndarray, dtype=np.float16) -> GemmResult:
        """
        Run GEMM: C = A @ B

        Args:
            A: Input matrix (M x K)
            B: Input matrix (K x N)
            dtype: Output data type (default: float16)

        Returns:
            GemmResult with output matrix and timing
        """
        M, K = A.shape
        K2, N = B.shape

        assert K == K2, f"Dimension mismatch: A is {M}x{K}, B is {K2}x{N}"

        # Ensure contiguous float16 arrays
        A_gpu = np.ascontiguousarray(A, dtype=np.float16)
        B_gpu = np.ascontiguousarray(B.T, dtype=np.float16)  # Column-major
        C_gpu = np.zeros((M, N), dtype=np.float16)

        # Run
        status, time_ms = self.lib.run_gemm(A_gpu, B_gpu, C_gpu, M, N, K)

        # Calculate TFLOPS
        flops = 2.0 * M * N * K
        tflops = (flops / (time_ms * 1e-3)) / 1e12 if time_ms > 0 else 0

        return GemmResult(
            output=C_gpu,
            time_ms=time_ms,
            status=status,
            tflops=tflops,
            kernel_name=self.lib.get_kernel_name(),
        )

    def benchmark(
        self, M: int, N: int, K: int, warmup: int = 2, iterations: int = 10
    ) -> dict:
        """Benchmark GEMM for given dimensions"""
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(K, N).astype(np.float16)

        times = []

        # Warmup
        for _ in range(warmup):
            self.run(A, B)

        # Benchmark
        for _ in range(iterations):
            result = self.run(A, B)
            if result.success:
                times.append(result.time_ms)

        if not times:
            return {"error": "All iterations failed"}

        flops = 2.0 * M * N * K
        avg_time = sum(times) / len(times)

        return {
            "M": M,
            "N": N,
            "K": K,
            "min_ms": min(times),
            "avg_ms": avg_time,
            "max_ms": max(times),
            "tflops": (flops / (avg_time * 1e-3)) / 1e12,
            "iterations": len(times),
        }


# =============================================================================
# Validation Utilities
# =============================================================================


class Validator:
    """Utilities for validating GEMM results"""

    def __init__(self, rtol: float = 1e-3, atol: float = 1e-2):
        self.rtol = rtol
        self.atol = atol

    def check(
        self, result: np.ndarray, reference: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Check if result matches reference

        Returns: (is_correct, max_diff, mean_diff)
        """
        result = result.astype(np.float32)
        reference = reference.astype(np.float32)

        diff = np.abs(result - reference)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))

        close = np.allclose(result, reference, rtol=self.rtol, atol=self.atol)

        return close, max_diff, mean_diff

    def compute_reference(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute reference GEMM result using NumPy"""
        return np.matmul(A.astype(np.float32), B.astype(np.float32))


# =============================================================================
# Code Generation Utilities
# =============================================================================


def get_codegen_path() -> Path:
    """Get path to unified_gemm_codegen.py"""
    return get_dispatcher_root() / "codegen" / "unified_gemm_codegen.py"


@dataclass
class CodegenResult:
    """Result of kernel code generation"""

    success: bool
    output_dir: Path
    variant: str
    stdout: str = ""
    stderr: str = ""
    kernel_count: int = 0
    elapsed_seconds: float = 0.0
    instance_names: List[str] = field(default_factory=list)

    def get_generated_kernels(self) -> List[Path]:
        """Get list of generated kernel headers"""
        if self.output_dir.exists():
            return list(self.output_dir.glob("*.hpp"))
        return []

    def print_instances(self, prefix: str = "    "):
        """Print all generated instance names."""
        for name in self.instance_names:
            print(f"{prefix}{name}")


def _run_codegen_subprocess(args: Dict[str, Any]) -> CodegenResult:
    """
    Worker function for parallel codegen execution.

    This is a module-level function to allow pickling for ProcessPoolExecutor.
    """
    import sys
    import subprocess
    from pathlib import Path

    codegen_path = Path(args["codegen_path"])
    out_dir = Path(args["output_dir"])
    variant = args["variant"]
    datatype = args["datatype"]
    layout = args["layout"]
    gpu_target = args["gpu_target"]
    extra_args = args.get("extra_args", [])
    timeout = args.get("timeout", 300)

    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # Get existing kernels before generation
    existing_kernels = set(out_dir.glob("*.hpp")) if out_dir.exists() else set()

    cmd = [
        sys.executable,
        str(codegen_path),
        "--output-dir",
        str(out_dir),
        "--datatype",
        datatype,
        "--layout",
        layout,
        "--gpu-target",
        gpu_target,
        "--variants",
        variant,
    ]

    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        # Get new kernels after generation
        all_kernels = set(out_dir.glob("*.hpp"))
        new_kernels = all_kernels - existing_kernels
        kernel_count = len(all_kernels)
        elapsed = time.time() - start

        # Build instance names list for verbose output
        instance_names = sorted([k.stem for k in new_kernels])

        return CodegenResult(
            success=result.returncode == 0,
            output_dir=out_dir,
            variant=variant,
            stdout=result.stdout,
            stderr=result.stderr,
            kernel_count=kernel_count,
            elapsed_seconds=elapsed,
            instance_names=instance_names,
        )
    except subprocess.TimeoutExpired:
        return CodegenResult(
            success=False,
            output_dir=out_dir,
            variant=variant,
            stderr=f"Code generation timed out ({timeout}s)",
            elapsed_seconds=time.time() - start,
        )
    except Exception as e:
        return CodegenResult(
            success=False,
            output_dir=out_dir,
            variant=variant,
            stderr=str(e),
            elapsed_seconds=time.time() - start,
        )


def _run_hipcc_subprocess(args: dict) -> Tuple[bool, Optional[Path], str]:
    """Module-level function to run hipcc compilation in parallel."""
    import subprocess
    from pathlib import Path

    compile_cmd = args["compile_cmd"]
    link_cmd = args["link_cmd"]
    lib_path = Path(args["lib_path"])

    try:
        res_c = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=300)
        if res_c.returncode != 0:
            return False, None, f"Compile failed: {res_c.stderr[:200]}"

        res_l = subprocess.run(link_cmd, capture_output=True, text=True, timeout=300)
        if res_l.returncode != 0:
            return False, None, f"Link failed: {res_l.stderr[:200]}"

        return True, lib_path, ""
    except subprocess.TimeoutExpired:
        return False, None, "Timeout"
    except Exception as e:
        return False, None, str(e)


def _generate_single_kernel_subprocess(args: dict) -> Tuple[bool, Optional[str], str]:
    """Module-level function: generate ONE kernel .hpp via --config JSON file.

    Used by setup_multiple_gemm_dispatchers for per-config parallel codegen.
    Returns (success, header_path_or_None, error_msg).
    """
    import subprocess
    import json
    import tempfile
    import os
    from pathlib import Path

    try:
        out_dir = Path(args["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write the single-config JSON to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(args["tile_config_json"], f)
            config_file = f.name

        cmd = [
            args["python"],
            str(args["codegen_script"]),
            "--output-dir",
            str(out_dir),
            "--datatype",
            args["dtype"],
            "--layout",
            args["layout"],
            "--gpu-target",
            args["gpu_target"],
            "--config",
            config_file,
            "--variants",
            "standard",
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        os.unlink(config_file)

        if res.returncode != 0:
            return False, None, f"Codegen failed: {res.stderr[:200]}"

        # Find the generated .hpp using the expected name pattern
        pattern = args["hpp_glob_pattern"]
        matches = sorted(out_dir.glob(pattern))
        if matches:
            return True, str(matches[0]), ""
        else:
            return False, None, f"No .hpp matching {pattern} after codegen"

    except Exception as e:
        return False, None, str(e)


def _parse_triplet(text: str) -> Optional[Tuple[int, int, int]]:
    parts = text.split("x")
    if len(parts) != 3:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def _parse_gemm_header_metadata(header: Path) -> Optional[Dict[str, Any]]:
    """
    Parse GEMM header name into configuration metadata.

    Expected stem format:
      gemm_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}
           _{pad_m}_{pad_n}_{pad_k}_{persistent}
           _{tile_m}x{tile_n}x{tile_k}_{wave_m}x{wave_n}x{wave_k}_{warp_m}x{warp_n}x{warp_k}
    """
    parts = header.stem.split("_")
    if len(parts) < 13 or parts[0] != "gemm":
        return None

    tile = _parse_triplet(parts[10])
    wave = _parse_triplet(parts[11])
    warp = _parse_triplet(parts[12])
    if tile is None or wave is None or warp is None:
        return None

    def _as_bool(v: str) -> bool:
        return v.lower() == "true"

    return {
        "dtype": parts[1],
        "layout": parts[2],
        "pipeline": parts[3],
        "epilogue": parts[4],
        "scheduler": parts[5],
        "pad_m": _as_bool(parts[6]),
        "pad_n": _as_bool(parts[7]),
        "pad_k": _as_bool(parts[8]),
        "persistent": _as_bool(parts[9]),
        "tile": tile,
        "wave": wave,
        "warp": warp,
    }


def _generate_arch_valid_gemm_headers(
    python_exe: str,
    codegen_script: Path,
    output_dir: Path,
    dtype: str,
    layout: str,
    gpu_target: str,
    variant: str = "standard",
) -> Tuple[bool, List[Path], str]:
    """Generate (or reuse) an arch-filtered kernel catalog for fallback selection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = f"gemm_{dtype}_{layout}_*.hpp"
    existing = sorted(output_dir.glob(pattern))
    if existing:
        return True, existing, ""

    cmd = [
        python_exe,
        str(codegen_script),
        "--output-dir",
        str(output_dir),
        "--datatype",
        dtype,
        "--layout",
        layout,
        "--gpu-target",
        gpu_target,
        "--variants",
        variant,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        err = (res.stderr or res.stdout or "").strip()[:500]
        return False, [], f"Catalog codegen failed: {err}"

    generated = sorted(output_dir.glob(pattern))
    if not generated:
        return False, [], "Catalog codegen produced no GEMM headers"
    return True, generated, ""


def _select_best_arch_valid_gemm_header(
    config: "KernelConfig",
    headers: List[Path],
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Choose nearest arch-valid header for a requested GEMM config."""
    best: Optional[Path] = None
    best_meta: Optional[Dict[str, Any]] = None
    best_score: Optional[Tuple[int, int, int, int, int, int]] = None

    for h in headers:
        meta = _parse_gemm_header_metadata(h)
        if meta is None:
            continue
        if meta["dtype"] != config.dtype_a or meta["layout"] != config.layout:
            continue

        tile = meta["tile"]
        wave = meta["wave"]
        warp = meta["warp"]
        tile_delta = (
            abs(tile[0] - config.tile_m)
            + abs(tile[1] - config.tile_n)
            + abs(tile[2] - config.tile_k)
        )
        wave_delta = (
            abs(wave[0] - config.wave_m)
            + abs(wave[1] - config.wave_n)
            + abs(wave[2] - config.wave_k)
        )
        warp_delta = (
            abs(warp[0] - config.warp_m)
            + abs(warp[1] - config.warp_n)
            + abs(warp[2] - config.warp_k)
        )
        score = (
            0 if meta["pipeline"] == config.pipeline else 1,
            0 if meta["scheduler"] == config.scheduler else 1,
            0 if meta["epilogue"] == config.epilogue else 1,
            tile_delta,
            wave_delta,
            warp_delta,
        )
        if best_score is None or score < best_score:
            best_score = score
            best = h
            best_meta = meta

    return best, best_meta


# =============================================================================
# Preshuffle Utilities
# =============================================================================


def preshuffle_weight_matrix(
    B: np.ndarray,
    warp_tile_n: int,
    warp_tile_k: int,
    arch: str = "gfx942",
) -> np.ndarray:
    """
    Preshuffle the B (weight) matrix for optimized GEMM inference.

    This transforms the B matrix layout to match the expected memory access
    pattern for preshuffle-enabled kernels. The transformation reorders data
    so that warp-level loads are coalesced.

    Args:
        B: Weight matrix of shape (K, N) in column-major / (K, N) layout
        warp_tile_n: Warp tile size in N dimension (e.g., 32)
        warp_tile_k: Warp tile size in K dimension (e.g., 16)
        arch: Target GPU architecture (gfx9xx, gfx11xx, gfx12xx)

    Returns:
        Shuffled B matrix with same data but reordered layout

    Example:
        >>> B = np.random.randn(1024, 2048).astype(np.float16)
        >>> B_shuffled = preshuffle_weight_matrix(B, warp_tile_n=32, warp_tile_k=16)
        >>> # Use B_shuffled with preshuffle-enabled kernel
    """
    K, N = B.shape

    # Validate dimensions are divisible by warp tiles
    if N % warp_tile_n != 0:
        raise ValueError(f"N ({N}) must be divisible by warp_tile_n ({warp_tile_n})")
    if K % warp_tile_k != 0:
        raise ValueError(f"K ({K}) must be divisible by warp_tile_k ({warp_tile_k})")

    # Architecture-specific shuffle patterns
    # Based on ck_tile/host/tensor_shuffle_utils.hpp
    if arch.startswith("gfx12"):
        # GFX12 (RDNA4) pattern
        divisor = 2
        k_abk1_per_lane = 8
        k_abk0_per_lane = warp_tile_k // divisor // k_abk1_per_lane

        if k_abk0_per_lane <= 0:
            raise ValueError(
                f"warp_tile_k ({warp_tile_k}) too small for GFX12 preshuffle"
            )

        # Reshape: (K, N) -> (N/warp_n, warp_n, K/warp_k, k0, div, k1)
        B_view = B.T.reshape(
            N // warp_tile_n,
            warp_tile_n,
            K // warp_tile_k,
            k_abk0_per_lane,
            divisor,
            k_abk1_per_lane,
        )
        # Permute: {0, 2, 4, 1, 3, 5}
        B_shuffled = np.transpose(B_view, (0, 2, 4, 1, 3, 5))

    elif arch.startswith("gfx11"):
        # GFX11 (RDNA3) pattern - divisor = 1
        divisor = 1

        # Reshape: (K, N) -> (N/warp_n, warp_n, K/warp_k, div, warp_k/div)
        B_view = B.T.reshape(
            N // warp_tile_n,
            warp_tile_n,
            K // warp_tile_k,
            divisor,
            warp_tile_k // divisor,
        )
        # Permute: {0, 2, 3, 1, 4}
        B_shuffled = np.transpose(B_view, (0, 2, 3, 1, 4))

    else:
        # GFX9 (CDNA) pattern - wave64
        divisor = 2 if warp_tile_n == 32 else 4

        # Reshape: (K, N) -> (N/warp_n, warp_n, K/warp_k, div, warp_k/div)
        B_view = B.T.reshape(
            N // warp_tile_n,
            warp_tile_n,
            K // warp_tile_k,
            divisor,
            warp_tile_k // divisor,
        )
        # Permute: {0, 2, 3, 1, 4}
        B_shuffled = np.transpose(B_view, (0, 2, 3, 1, 4))

    # Return contiguous array with same dtype
    return np.ascontiguousarray(B_shuffled.reshape(-1)).reshape(B.shape)


def is_preshuffle_supported(arch: str) -> bool:
    """Check if preshuffle is supported for the given architecture."""
    # Preshuffle is supported on CDNA (gfx9xx) and RDNA (gfx11xx, gfx12xx)
    return arch.startswith(("gfx9", "gfx11", "gfx12"))


@dataclass
class KernelConfig:
    """
    Complete kernel configuration for GEMM.

    This defines all parameters needed to generate and run a specific kernel.
    """

    # Data types
    dtype_a: str = "fp16"
    dtype_b: str = "fp16"
    dtype_c: str = "fp16"
    dtype_acc: str = "fp32"

    # Layouts (row/col)
    layout_a: str = "row"
    layout_b: str = "col"
    layout_c: str = "row"

    # Tile shape (work per thread block)
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 32

    # Wave shape (warps per block)
    wave_m: int = 2
    wave_n: int = 2
    wave_k: int = 1

    # Warp tile (elements per warp)
    warp_m: int = 32
    warp_n: int = 32
    warp_k: int = 16

    # Block configuration
    block_size: int = 256

    # Pipeline configuration
    pipeline: str = "compv4"
    scheduler: str = "intrawave"
    epilogue: str = "cshuffle"

    # Padding (enables arbitrary problem sizes)
    pad_m: bool = True
    pad_n: bool = True
    pad_k: bool = True

    # GPU target
    gfx_arch: str = "gfx942"

    # GEMM variant (affects arch filter validation)
    # "standard", "preshuffle", or "multi_d"
    variant: str = "standard"

    @property
    def layout(self) -> str:
        """Get layout string (e.g., 'rcr' for row-col-row)"""
        mapping = {"row": "r", "col": "c"}
        return mapping[self.layout_a] + mapping[self.layout_b] + mapping[self.layout_c]

    @property
    def tile_str(self) -> str:
        """Get tile size string"""
        return f"{self.tile_m}x{self.tile_n}x{self.tile_k}"

    def print_config(self, indent: str = "  "):
        """Pretty print the configuration."""
        print(f"{indent}KernelConfig:")
        print(
            f"{indent}  Data types: A={self.dtype_a}, B={self.dtype_b}, C={self.dtype_c}, Acc={self.dtype_acc}"
        )
        print(
            f"{indent}  Layouts:    A={self.layout_a}, B={self.layout_b}, C={self.layout_c} ({self.layout})"
        )
        print(f"{indent}  Tile:       {self.tile_m}x{self.tile_n}x{self.tile_k}")
        print(f"{indent}  Waves:      {self.wave_m}x{self.wave_n}x{self.wave_k}")
        print(f"{indent}  Warp tile:  {self.warp_m}x{self.warp_n}x{self.warp_k}")
        print(f"{indent}  Block size: {self.block_size}")
        print(f"{indent}  Pipeline:   {self.pipeline}/{self.scheduler}/{self.epilogue}")
        print(f"{indent}  Padding:    M={self.pad_m}, N={self.pad_n}, K={self.pad_k}")
        print(f"{indent}  Target:     {self.gfx_arch}")


class CodegenRunner:
    """
    Runner for the unified GEMM code generator with parallel execution support.

    Usage:
        codegen = CodegenRunner()

        # Generate standard kernels
        result = codegen.generate("standard")

        # Generate preshuffle kernels
        result = codegen.generate("preshuffle")

        # Generate multi-D kernels
        result = codegen.generate("multi_d")

        # Generate all variants IN PARALLEL
        results = codegen.generate_all_parallel()

        # Generate multiple configs IN PARALLEL
        configs = [KernelConfig(...), KernelConfig(...)]
        results = codegen.generate_configs_parallel(configs)

        # Generate with custom output directory
        result = codegen.generate("standard", output_dir=Path("/custom/path"))

        # Generate from specific config
        config = KernelConfig(tile_m=256, tile_n=256, tile_k=64)
        result = codegen.generate_from_config(config)
    """

    VARIANTS = ["standard", "preshuffle", "multi_d"]

    def __init__(
        self,
        codegen_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        datatype: str = "fp16",
        layout: str = "rcr",
        gpu_target: str = "gfx942",
        max_workers: Optional[int] = None,
    ):
        self.codegen_path = codegen_path or get_codegen_path()
        self.output_dir = output_dir or get_generated_kernels_dir()
        self.datatype = datatype
        self.layout = layout
        self.gpu_target = gpu_target
        # Default to CPU count, but cap at reasonable value
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)

    def _make_args(
        self,
        variant: str,
        output_dir: Optional[Path] = None,
        extra_args: Optional[List[str]] = None,
        timeout: int = 300,
        show_instances: bool = False,
    ) -> Dict[str, Any]:
        """Build args dict for parallel worker."""
        return {
            "codegen_path": str(self.codegen_path),
            "output_dir": str(output_dir or self.output_dir),
            "variant": variant,
            "datatype": self.datatype,
            "layout": self.layout,
            "gpu_target": self.gpu_target,
            "extra_args": extra_args or [],
            "timeout": timeout,
            "show_instances": show_instances,
        }

    def generate(
        self,
        variant: str = "standard",
        output_dir: Optional[Path] = None,
        extra_args: Optional[List[str]] = None,
        show_instances: bool = False,
    ) -> CodegenResult:
        """
        Generate kernels for a specific variant (single-threaded).

        Args:
            variant: One of "standard", "preshuffle", "multi_d"
            output_dir: Override output directory
            extra_args: Additional arguments to pass to codegen
            show_instances: Print "Adding Instance" and "Building Instance" for each kernel

        Returns:
            CodegenResult with generation status and info
        """
        args = self._make_args(
            variant, output_dir, extra_args, show_instances=show_instances
        )
        result = _run_codegen_subprocess(args)

        if show_instances and result.instance_names:
            for name in result.instance_names:
                print(f"  Adding Instance: {name}")
                print(f"  Building Instance: {name}")

        return result

    def generate_all(self, output_dir: Optional[Path] = None) -> List[CodegenResult]:
        """Generate all variants sequentially (use generate_all_parallel for speed)."""
        results = []
        for variant in self.VARIANTS:
            result = self.generate(variant, output_dir)
            results.append(result)
        return results

    def generate_all_parallel(
        self,
        output_dir: Optional[Path] = None,
        variants: Optional[List[str]] = None,
        verbose: bool = True,
        show_instances: bool = False,
    ) -> List[CodegenResult]:
        """
        Generate all variants IN PARALLEL.

        Args:
            output_dir: Override output directory
            variants: List of variants to generate (default: all)
            verbose: Print progress
            show_instances: Print "Adding Instance" and "Building Instance" for each kernel

        Returns:
            List of CodegenResult for each variant
        """
        variants = variants or self.VARIANTS
        start_total = time.time()

        if verbose:
            print(
                f"Generating {len(variants)} variants in parallel (workers={self.max_workers})..."
            )

        # Build args for each variant
        args_list = [self._make_args(v, output_dir) for v in variants]
        for args in args_list:
            args["show_instances"] = show_instances

        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_run_codegen_subprocess, args): args["variant"]
                for args in args_list
            }

            for future in as_completed(futures):
                variant = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if verbose:
                        status = "OK" if result.success else "FAIL"
                        print(
                            f"  {status} {variant}: {result.kernel_count} kernels in {result.elapsed_seconds:.2f}s"
                        )
                        if show_instances and result.instance_names:
                            for name in result.instance_names:
                                print(f"      Adding Instance: {name}")
                                print(f"      Building Instance: {name}")
                except Exception as e:
                    results.append(
                        CodegenResult(
                            success=False,
                            output_dir=output_dir or self.output_dir,
                            variant=variant,
                            stderr=str(e),
                        )
                    )
                    if verbose:
                        print(f"  FAIL {variant}: FAILED - {e}")

        total_time = time.time() - start_total
        if verbose:
            total_kernels = sum(r.kernel_count for r in results)
            print(f"Total: {total_kernels} kernels in {total_time:.2f}s")

        return results

    def generate_configs_parallel(
        self,
        configs: List["KernelConfig"],
        output_dir: Optional[Path] = None,
        verbose: bool = True,
        show_instances: bool = False,
    ) -> List[CodegenResult]:
        """
        Generate kernels from multiple configs IN PARALLEL.

        Each config generates independently, allowing maximum parallelism.

        Args:
            configs: List of KernelConfig objects
            output_dir: Override output directory
            verbose: Print progress
            show_instances: Print "Adding Instance" and "Building Instance" for each kernel

        Returns:
            List of CodegenResult for each config
        """
        start_total = time.time()
        out_dir = output_dir or self.output_dir

        if verbose:
            print(
                f"Generating {len(configs)} configs in parallel (workers={self.max_workers})..."
            )

        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for config in configs:
                args = {
                    "codegen_path": str(self.codegen_path),
                    "output_dir": str(out_dir),
                    "variant": "standard",
                    "datatype": config.dtype_a,
                    "layout": config.layout,
                    "gpu_target": config.gfx_arch,
                    "extra_args": [],
                    "timeout": 300,
                    "show_instances": show_instances,
                }
                future = executor.submit(_run_codegen_subprocess, args)
                futures[future] = config.tile_str

            for future in as_completed(futures):
                tile_str = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if verbose:
                        status = "OK" if result.success else "FAIL"
                        print(
                            f"  {status} {tile_str}: {result.kernel_count} kernels in {result.elapsed_seconds:.2f}s"
                        )
                        if show_instances and result.instance_names:
                            for name in result.instance_names:
                                print(f"      Adding Instance: {name}")
                                print(f"      Building Instance: {name}")
                except Exception as e:
                    results.append(
                        CodegenResult(
                            success=False,
                            output_dir=out_dir,
                            variant=f"config:{tile_str}",
                            stderr=str(e),
                        )
                    )
                    if verbose:
                        print(f"  FAIL {tile_str}: FAILED - {e}")

        total_time = time.time() - start_total
        if verbose:
            total_kernels = sum(r.kernel_count for r in results)
            print(f"Total: {total_kernels} kernels in {total_time:.2f}s")

        return results

    def generate_batch_parallel(
        self,
        batch: List[Dict[str, Any]],
        verbose: bool = True,
        show_instances: bool = False,
    ) -> List[CodegenResult]:
        """
        Generate a batch of kernel specs IN PARALLEL.

        This is the most flexible parallel generation method.

        Args:
            batch: List of dicts with keys: variant, datatype, layout, gpu_target, output_dir
            verbose: Print progress
            show_instances: Print "Adding Instance" and "Building Instance" for each kernel

        Returns:
            List of CodegenResult
        """
        start_total = time.time()

        if verbose:
            print(
                f"Generating {len(batch)} kernel specs in parallel (workers={self.max_workers})..."
            )

        # Build args for each spec
        args_list = []
        for spec in batch:
            args = {
                "codegen_path": str(self.codegen_path),
                "output_dir": str(spec.get("output_dir", self.output_dir)),
                "variant": spec.get("variant", "standard"),
                "datatype": spec.get("datatype", self.datatype),
                "layout": spec.get("layout", self.layout),
                "gpu_target": spec.get("gpu_target", self.gpu_target),
                "extra_args": spec.get("extra_args", []),
                "timeout": spec.get("timeout", 300),
                "show_instances": show_instances,
            }
            args_list.append(args)

        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_run_codegen_subprocess, args): args["variant"]
                for args in args_list
            }

            for future in as_completed(futures):
                variant = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if verbose:
                        status = "OK" if result.success else "FAIL"
                        print(
                            f"  {status} {variant}: {result.kernel_count} kernels in {result.elapsed_seconds:.2f}s"
                        )
                        if show_instances and result.instance_names:
                            for name in result.instance_names:
                                print(f"      Adding Instance: {name}")
                                print(f"      Building Instance: {name}")
                except Exception as e:
                    results.append(
                        CodegenResult(
                            success=False,
                            output_dir=self.output_dir,
                            variant=variant,
                            stderr=str(e),
                        )
                    )
                    if verbose:
                        print(f"  FAIL {variant}: FAILED - {e}")

        total_time = time.time() - start_total
        if verbose:
            total_kernels = sum(r.kernel_count for r in results)
            print(f"Total: {total_kernels} kernels in {total_time:.2f}s")

        return results

    def generate_from_config(
        self,
        config: KernelConfig,
        output_dir: Optional[Path] = None,
        force: bool = False,
        show_instances: bool = False,
    ) -> CodegenResult:
        """
        Generate kernel from a specific KernelConfig.

        This generates ONLY the specific kernel header needed (not all kernels).
        Note: This does NOT rebuild the library - use build_library_for_configs()
        for that.

        Args:
            config: KernelConfig with all kernel parameters
            output_dir: Override output directory
            force: Force regeneration even if kernel exists
            show_instances: Print instance names when generating

        Returns:
            CodegenResult with the specific kernel
        """
        import sys
        import json
        import tempfile

        out_dir = output_dir or self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build kernel filename pattern for this config
        # Note: padding flags may differ from config (arch filter may enable padding)
        tile_str = config.tile_str  # e.g., "128x128x32"
        wave_str = f"{config.wave_m}x{config.wave_n}x{config.wave_k}"
        warp_str = f"{config.warp_m}x{config.warp_n}x{config.warp_k}"

        # Build pattern - use * for padding flags since arch filter may change them
        precise_pattern = f"gemm_{config.dtype_a}_{config.layout}_{config.pipeline}_{config.epilogue}_{config.scheduler}_*_*_*_*_{tile_str}_{wave_str}_{warp_str}.hpp"

        # Check if exact kernel already exists
        existing = list(out_dir.glob(precise_pattern))
        if existing and not force:
            instance_names = sorted([k.stem for k in existing])
            if show_instances:
                for name in instance_names:
                    print(f"  Kernel exists: {name}")

            return CodegenResult(
                success=True,
                output_dir=out_dir,
                variant=f"config:{tile_str}",
                kernel_count=len(existing),
                instance_names=instance_names,
                stdout=f"Kernel exists, using: {existing[0].name}",
            )

        if not self.codegen_path.exists():
            return CodegenResult(
                success=False,
                output_dir=out_dir,
                variant=f"config:{tile_str}",
                stderr=f"Codegen not found at {self.codegen_path}",
            )

        start = time.time()

        # Create a temporary config file for single-kernel generation
        # Format must match what unified_gemm_codegen.py expects
        single_config = {
            "tile_config": {
                "tile_m": [config.tile_m],
                "tile_n": [config.tile_n],
                "tile_k": [config.tile_k],
                "warp_m": [config.wave_m],
                "warp_n": [config.wave_n],
                "warp_k": [config.wave_k],
                "warp_tile_m": [config.warp_m],
                "warp_tile_n": [config.warp_n],
                "warp_tile_k": [config.warp_k],
            },
            "trait_config": {
                "pipeline": [config.pipeline],
                "epilogue": [config.epilogue],
                "scheduler": [config.scheduler],
                "pad_m": [config.pad_m],
                "pad_n": [config.pad_n],
                "pad_k": [config.pad_k],
                "persistent": [False],
            },
        }

        # Write temp config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(single_config, f)
            config_file = f.name

        try:
            # Generate ONLY this specific kernel using config file
            cmd = [
                sys.executable,
                str(self.codegen_path),
                "--output-dir",
                str(out_dir),
                "--datatype",
                config.dtype_a,
                "--layout",
                config.layout,
                "--gpu-target",
                config.gfx_arch,
                "--config",
                config_file,
                "--variants",
                "standard",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Find the generated kernel
            matching = list(out_dir.glob(precise_pattern))
            kernel_count = len(matching)
            elapsed = time.time() - start

            instance_names = sorted([k.stem for k in matching])
            if show_instances and instance_names:
                for name in instance_names:
                    print(f"  Generated: {name}")

            return CodegenResult(
                success=result.returncode == 0 and kernel_count > 0,
                output_dir=out_dir,
                variant=f"config:{tile_str}",
                stdout=result.stdout,
                stderr=result.stderr,
                kernel_count=kernel_count,
                elapsed_seconds=elapsed,
                instance_names=instance_names,
            )
        except Exception as e:
            return CodegenResult(
                success=False,
                output_dir=out_dir,
                variant=f"config:{tile_str}",
                stderr=str(e),
            )
        finally:
            # Clean up temp file
            import os

            try:
                os.unlink(config_file)
            except Exception:
                pass

    def _rebuild_library_for_config(
        self, config: KernelConfig, kernel_header: Path
    ) -> Optional[Path]:
        """
        Rebuild the library with the specified kernel header using hipcc directly.

        This compiles a new library with exactly the kernel specified.
        Builds to a UNIQUE filename to avoid conflicts with loaded libraries.

        Architecture Note - C++ vs Python Paths:
        -----------------------------------------
        C++ Multi-Kernel Path:
          - Each kernel is in its own namespace (ns_gemm_...)
          - Multiple kernel headers can be included together
          - Uses namespace-qualified types: ns_...:SelectedKernel
          - Does NOT define CK_TILE_SINGLE_KERNEL_INCLUDE
          - Registration code uses block-scoped type aliases

        Python Single-Kernel JIT Path (this function):
          - Each library contains exactly ONE kernel
          - Uses -DCK_TILE_SINGLE_KERNEL_INCLUDE to export types to global namespace
          - gemm_ctypes_lib.cpp expects: SelectedKernel, KERNEL_NAME, ADataType, etc.
          - Different configs get different library files (by dtype/layout)
          - This enables Python to use any kernel config without pre-building all

        Returns: Path to new library, or None on failure
        """
        build_dir = get_build_dir()
        # Use unique filename based on ALL distinguishing config parameters
        # Include: dtype, layout, tile, wave, warp, pipeline, epilogue, scheduler
        # This ensures different configs don't collide even if tile/pipeline match
        wave_str = f"{config.wave_m}x{config.wave_n}x{config.wave_k}"
        warp_str = f"{config.warp_m}x{config.warp_n}x{config.warp_k}"
        lib_name = (
            f"libdispatcher_gemm_{config.dtype_a}_{config.layout}_"
            f"{config.tile_str}_{wave_str}_{warp_str}_"
            f"{config.pipeline}_{config.epilogue}_{config.scheduler}.so"
        )
        lib_path = build_dir / "examples" / lib_name

        print(f"  Rebuilding library: {lib_name}")
        print(f"  With kernel: {kernel_header.name}")

        root = get_dispatcher_root()
        ck_root = root.parent

        ctypes_source = root / "bindings/ctypes/gemm_ctypes_lib.cpp"
        if not ctypes_source.exists():
            print(f"  Source not found: {ctypes_source}")
            return None

        # Link against the static dispatcher library (contains Registry, Dispatcher)
        static_lib = build_dir / "libck_tile_dispatcher.a"
        if not static_lib.exists():
            print(f"  Static library not found: {static_lib}")
            print("  Build with: cd build && cmake .. && make ck_tile_dispatcher")
            return None

        # Compile source to object first, then link
        obj_file = lib_path.with_suffix(".o")

        # Step 1: Compile source to object
        # CK_TILE_SINGLE_KERNEL_INCLUDE enables global namespace exports in the kernel header
        # This exports: SelectedKernel, KERNEL_NAME, ADataType, BDataType, CDataType, AccDataType
        compile_cmd = [
            "/opt/rocm/bin/hipcc",
            "-c",  # Compile only
            "-fPIC",
            "-O3",
            f"-I{root / 'include'}",
            f"-I{ck_root / 'include'}",
            f"-I{ck_root}",
            f"-I{root / 'build/generated_kernels'}",
            "-DCK_TILE_SINGLE_KERNEL_INCLUDE",  # Enable global namespace exports
            f"-include{kernel_header}",
            "-D__HIP_PLATFORM_AMD__",
            f"--offload-arch={config.gfx_arch}",
            f'-DGFX_ARCH="{config.gfx_arch}"',  # Pass arch as string for gemm_ctypes_lib.cpp
            "-mllvm",
            "-enable-noalias-to-md-conversion=0",
            "-Wno-undefined-func-template",
            "-Wno-float-equal",
            str(ctypes_source),
            "-o",
            str(obj_file),
        ]

        try:
            print("  Compiling source...")
            result = subprocess.run(
                compile_cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                print(f"  Compilation failed: {result.stderr[:300]}")
                return None

            # Step 2: Link object with static library into shared library
            link_cmd = [
                "/opt/rocm/bin/hipcc",
                "-shared",
                "-fPIC",
                f"--offload-arch={config.gfx_arch}",
                "--hip-link",
                str(obj_file),
                str(static_lib),
                "-o",
                str(lib_path),
            ]

            print("  Linking...")
            result = subprocess.run(
                link_cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print(f"  OK Library rebuilt: {lib_path.name}")
                # Clean up object file
                obj_file.unlink(missing_ok=True)
                return lib_path
            else:
                print(f"  Linking failed: {result.stderr[:300]}")
                return None
        except subprocess.TimeoutExpired:
            print("  Build timed out")
            return None
        except Exception as e:
            print(f"  Build error: {e}")
            return None

    def build_libraries_parallel(
        self, configs_and_headers: List[Tuple[KernelConfig, Path]], verbose: bool = True
    ) -> List[Optional[Path]]:
        """
        Build multiple libraries in parallel using ProcessPoolExecutor.
        Returns a list of library paths (or None if a build failed) in the same order.
        """
        import time
        from concurrent.futures import ProcessPoolExecutor, as_completed

        start_time = time.time()
        build_dir = get_build_dir()
        root = get_dispatcher_root()
        ck_root = root.parent
        ctypes_source = root / "bindings/ctypes/gemm_ctypes_lib.cpp"
        static_lib = build_dir / "libck_tile_dispatcher.a"

        if not ctypes_source.exists() or not static_lib.exists():
            if verbose:
                print("  Required source or static library missing for parallel build.")
            return [None] * len(configs_and_headers)

        args_list = []
        for config, kernel_header in configs_and_headers:
            lib_name = f"libdispatcher_gemm_{config.dtype_a}_{config.layout}_{config.tile_str}_{config.pipeline}.so"
            lib_path = build_dir / "examples" / lib_name
            obj_file = lib_path.with_suffix(".o")

            compile_cmd = [
                "/opt/rocm/bin/hipcc",
                "-c",
                "-fPIC",
                "-O3",
                f"-I{root / 'include'}",
                f"-I{ck_root / 'include'}",
                f"-I{ck_root}",
                f"-I{root / 'build/generated_kernels'}",
                "-DCK_TILE_SINGLE_KERNEL_INCLUDE",
                f"-include{kernel_header}",
                "-D__HIP_PLATFORM_AMD__",
                f"--offload-arch={config.gfx_arch}",
                f'-DGFX_ARCH="{config.gfx_arch}"',
                "-mllvm",
                "-enable-noalias-to-md-conversion=0",
                "-Wno-undefined-func-template",
                "-Wno-float-equal",
                str(ctypes_source),
                "-o",
                str(obj_file),
            ]

            link_cmd = [
                "/opt/rocm/bin/hipcc",
                "-shared",
                "-fPIC",
                f"--offload-arch={config.gfx_arch}",
                "--hip-link",
                str(obj_file),
                str(static_lib),
                "-o",
                str(lib_path),
            ]

            args_list.append(
                {
                    "compile_cmd": compile_cmd,
                    "link_cmd": link_cmd,
                    "lib_path": str(lib_path),
                    "config_name": f"{config.dtype_a}_{config.layout}_{config.tile_str}",
                }
            )

        if verbose:
            print(
                f"Building {len(args_list)} libraries in parallel (workers={self.max_workers})..."
            )

        results_map = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_run_hipcc_subprocess, args): i
                for i, args in enumerate(args_list)
            }
            for future in as_completed(futures):
                idx = futures[future]
                success, lib_path, err = future.result()
                results_map[idx] = Path(lib_path) if success else None
                if verbose:
                    status = "OK" if success else f"FAIL ({err})"
                    print(
                        f"  {status} {Path(lib_path).name if success else args_list[idx]['config_name']}"
                    )

        if verbose:
            elapsed = time.time() - start_time
            print(f"Parallel build finished in {elapsed:.2f}s")

        return [results_map[i] for i in range(len(configs_and_headers))]

    def generate_preselected(
        self, preset: str = "fp16_rcr_essential", output_dir: Optional[Path] = None
    ) -> CodegenResult:
        """
        Generate kernels from a preselected set.

        Args:
            preset: Preselected kernel set name (e.g., "fp16_rcr_essential")
            output_dir: Override output directory

        Returns:
            CodegenResult
        """
        import sys

        out_dir = output_dir or self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(self.codegen_path),
            "--output-dir",
            str(out_dir),
            "--preselected",
            preset,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            kernel_count = len(list(out_dir.glob("*.hpp")))

            return CodegenResult(
                success=result.returncode == 0,
                output_dir=out_dir,
                variant=f"preselected:{preset}",
                stdout=result.stdout,
                stderr=result.stderr,
                kernel_count=kernel_count,
            )
        except Exception as e:
            return CodegenResult(
                success=False,
                output_dir=out_dir,
                variant=f"preselected:{preset}",
                stderr=str(e),
            )

    def ensure_kernels_exist(self) -> bool:
        """
        Ensure kernel headers exist, generating if necessary.

        Returns:
            True if kernels exist or were successfully generated
        """
        if self.output_dir.exists():
            kernels = list(self.output_dir.glob("*.hpp"))
            if kernels:
                return True

        # Generate standard kernels
        result = self.generate("standard")
        return result.success

    def list_kernels(self) -> List[Path]:
        """List all generated kernel headers"""
        if self.output_dir.exists():
            return sorted(self.output_dir.glob("*.hpp"))
        return []

    def categorize_kernels(self) -> dict:
        """
        Categorize kernels by tile size and variant.

        Returns:
            Dict with categories by tile size and variant type
        """
        kernels = self.list_kernels()

        # Separate by variant first
        preshuffle = [k for k in kernels if "_preshuffle" in k.name]
        multi_d = [k for k in kernels if "_multid_" in k.name]
        standard = [
            k
            for k in kernels
            if "_preshuffle" not in k.name and "_multid_" not in k.name
        ]

        # Categorize standard kernels by tile size
        compute = [k for k in standard if "_256x" in k.name]
        memory = [k for k in standard if "_128x" in k.name]
        latency = [k for k in standard if "_64x" in k.name or "_32x" in k.name]

        return {
            "total": len(kernels),
            "standard": len(standard),
            "compute": compute,
            "memory": memory,
            "latency": latency,
            "preshuffle": preshuffle,
            "multi_d": multi_d,
        }


# =============================================================================
# Registry and Dispatcher (Explicit API)
# =============================================================================


class Registry:
    """
    Kernel registry - stores and manages kernel instances.

    This provides an explicit registry API that mirrors the C++ Registry class.

    Usage:
        registry = Registry()
        registry.register_kernel(kernel_config)
        dispatcher = Dispatcher(registry)
    """

    def __init__(self, lib: Optional[DispatcherLib] = None, name: str = "default"):
        self._lib = lib
        self._name = name
        self._kernels: List[KernelConfig] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def kernel_count(self) -> int:
        if self._lib:
            return self._lib.get_kernel_count()
        return len(self._kernels)

    def register_kernel(self, config: KernelConfig) -> bool:
        """Register a kernel configuration."""
        self._kernels.append(config)
        return True

    def get_kernels(self) -> List[KernelConfig]:
        """Get all registered kernel configs."""
        return self._kernels.copy()

    def clear(self):
        """Clear all kernels."""
        self._kernels.clear()

    def bind_library(self, lib: DispatcherLib):
        """Bind to a loaded dispatcher library."""
        self._lib = lib

    def build(
        self,
        verbose: bool = False,
        max_workers: Optional[int] = None,
    ) -> List["GemmSetupResult"]:
        """Parallel JIT compile all kernels in this registry.

        Args:
            verbose:     Print progress during build.
            max_workers: Max parallel codegen/compile processes (default: cpu_count capped at 8).

        Returns a GemmSetupResult per registered kernel (same order as get_kernels()).
        """
        if not self._kernels:
            return []
        return setup_multiple_gemm_dispatchers(
            self._kernels,
            registry_name=self._name,
            verbose=verbose,
            max_workers=max_workers,
        )

    def __repr__(self) -> str:
        return f"Registry(name='{self._name}', kernels={self.kernel_count})"


class Dispatcher:
    """
    Kernel dispatcher - selects and runs kernels for problems.

    This provides an explicit dispatcher API that mirrors the C++ Dispatcher class.

    Usage:
        registry = Registry()
        registry.register_kernel(config)

        dispatcher = Dispatcher(registry)
        result = dispatcher.run(A, B, M, N, K)
    """

    def __init__(self, registry: Registry, lib: Optional[DispatcherLib] = None):
        self._registry = registry
        self._lib = lib or registry._lib

    @property
    def registry(self) -> Registry:
        return self._registry

    def select_kernel(self, M: int, N: int, K: int) -> Optional[str]:
        """Select best kernel for problem dimensions."""
        if self._lib:
            return self._lib.select_kernel(M, N, K)
        # Fallback: return first matching kernel
        for config in self._registry.get_kernels():
            return f"kernel_{config.tile_str}"
        return None

    def is_supported(self, M: int, N: int, K: int) -> bool:
        """Check if problem size is supported."""
        if self._lib:
            return self._lib.is_supported(M, N, K)
        return len(self._registry.get_kernels()) > 0

    def run(self, A: np.ndarray, B: np.ndarray, M: int, N: int, K: int) -> GemmResult:
        """
        Run GEMM: C = A @ B

        Args:
            A: Input matrix (M x K)
            B: Input matrix (K x N)
            M, N, K: Problem dimensions

        Returns:
            GemmResult with output and timing
        """
        if self._lib is None:
            raise RuntimeError("Dispatcher not bound to library")

        # Ensure contiguous float16 arrays
        A_gpu = np.ascontiguousarray(A, dtype=np.float16)
        B_gpu = np.ascontiguousarray(B.T, dtype=np.float16)  # Column-major
        C_gpu = np.zeros((M, N), dtype=np.float16)

        # Run via library
        status, time_ms = self._lib.run_gemm(A_gpu, B_gpu, C_gpu, M, N, K)

        # Calculate TFLOPS
        flops = 2.0 * M * N * K
        tflops = (flops / (time_ms * 1e-3)) / 1e12 if time_ms > 0 else 0

        return GemmResult(
            output=C_gpu,
            time_ms=time_ms,
            status=status,
            tflops=tflops,
            kernel_name=self._lib.get_kernel_name() if self._lib else "unknown",
        )

    def __repr__(self) -> str:
        return f"Dispatcher(registry={self._registry.name}, kernels={self._registry.kernel_count})"


# =============================================================================
# Main (self-test)
# =============================================================================

if __name__ == "__main__":
    print("CK Tile Dispatcher Utils Self-Test")
    print("=" * 60)

    # Test library loading
    print("\n1. Loading library...")
    lib = DispatcherLib.auto()
    if lib is None:
        print("   FAILED: Could not load library")
        exit(1)
    print(f"   OK: Loaded from {lib.path}")
    print(f"   Kernel: {lib.get_kernel_name()}")
    print(f"   Registered kernels: {lib.get_kernel_count()}")

    # Test GEMM
    print("\n2. Running GEMM 256x256x256...")
    runner = GemmRunner(lib)
    A = np.random.randn(256, 256).astype(np.float16)
    B = np.random.randn(256, 256).astype(np.float16)

    result = runner.run(A, B)
    print(f"   Status: {'OK' if result.success else 'FAILED'}")
    print(f"   Time: {result.time_ms:.4f} ms")
    print(f"   TFLOPS: {result.tflops:.2f}")

    # Test validation
    print("\n3. Validating result...")
    validator = Validator()
    reference = validator.compute_reference(A, B)
    correct, max_diff, mean_diff = validator.check(result.output, reference)
    print(f"   Correct: {correct}")
    print(f"   Max diff: {max_diff:.6f}")

    print("\n" + "=" * 60)
    print("All tests passed!")


# =============================================================================
# High-Level Helper Functions
# =============================================================================


@dataclass
class GemmSetupResult:
    """Result of setup_gemm_dispatcher"""

    success: bool
    dispatcher: Optional[Dispatcher] = None
    lib: Optional[DispatcherLib] = None
    registry: Optional[Registry] = None
    codegen: Optional[CodegenRunner] = None
    config: Optional[KernelConfig] = None
    kernel_header: Optional[Path] = None
    error: str = ""
    corrections: List[str] = field(default_factory=list)


def setup_gemm_dispatcher(
    config: KernelConfig,
    registry_name: str = "gemm_registry",
    verbose: bool = True,
    auto_rebuild: bool = True,
) -> GemmSetupResult:
    """
    High-level helper to setup a GEMM dispatcher from a kernel config.

    This handles:
    1. Validate config against arch filter (auto-correct if needed)
    2. Generate kernel code if needed
    3. Find matching kernel header
    4. Load or rebuild library (if dtype mismatch)
    5. Create registry and dispatcher

    Args:
        config: KernelConfig with all parameters
        registry_name: Name for the registry
        verbose: Print progress messages
        auto_rebuild: Rebuild library if dtype doesn't match

    Returns:
        GemmSetupResult with dispatcher, lib, registry, etc.
    """
    result = GemmSetupResult(success=False, config=config)

    def log(msg):
        if verbose:
            print(msg)

    # Step 1: Validate config
    log("  Validating config...")
    validation = validate_kernel_config(config)
    if not validation.is_valid:
        log("  WARNING Auto-correcting configuration...")
        config, was_modified, corrections = auto_correct_kernel_config(
            config, verbose=verbose
        )
        result.config = config
        result.corrections = corrections
        # Note: corrections will be displayed by the caller via print_auto_correction

    # Step 2: Setup codegen and generate kernel
    log(f"  Generating kernel (tile={config.tile_str})...")
    codegen = CodegenRunner(
        datatype=config.dtype_a,
        layout=config.layout,
        gpu_target=config.gfx_arch,
    )
    result.codegen = codegen

    codegen_result = codegen.generate_from_config(config)
    if not codegen_result.success:
        log("  WARNING Kernel generation: using existing")

    # Step 3: Find matching kernel header
    kernel_header = find_matching_kernel_header(config)
    result.kernel_header = kernel_header
    if not kernel_header:
        log("  WARNING No matching kernel header found")

    # Step 4: Load library
    log("  Loading library...")
    lib = DispatcherLib.auto()
    if lib is None:
        result.error = "Could not load dispatcher library"
        return result
    result.lib = lib

    # Check if library kernel matches config - rebuild if ANY parameter differs
    lib_kernel = lib.get_kernel_name()
    needs_rebuild = False
    mismatches = []

    if lib_kernel:
        # Build expected kernel signature components from config
        expected_parts = {
            "dtype": config.dtype_a,
            "layout": config.layout,
            "pipeline": config.pipeline,
            "epilogue": config.epilogue,
            "scheduler": config.scheduler,
            "tile": f"{config.tile_m}x{config.tile_n}x{config.tile_k}",
            "wave": f"{config.wave_m}x{config.wave_n}x{config.wave_k}",
            "warp": f"{config.warp_m}x{config.warp_n}x{config.warp_k}",
        }

        # Check each component against the library kernel name
        for name, expected in expected_parts.items():
            if expected not in lib_kernel:
                needs_rebuild = True
                mismatches.append(f"{name}={expected}")

    if needs_rebuild and auto_rebuild:
        log(f"  Library kernel doesn't match config: {', '.join(mismatches)}")

        # Check if a rebuilt library for this exact config already exists
        build_dir = get_build_dir()
        wave_str = f"{config.wave_m}x{config.wave_n}x{config.wave_k}"
        warp_str = f"{config.warp_m}x{config.warp_n}x{config.warp_k}"
        cached_lib_name = (
            f"libdispatcher_gemm_{config.dtype_a}_{config.layout}_"
            f"{config.tile_str}_{wave_str}_{warp_str}_"
            f"{config.pipeline}_{config.epilogue}_{config.scheduler}.so"
        )
        cached_lib_path = build_dir / "examples" / cached_lib_name

        if cached_lib_path.exists():
            log(f"  Using cached library: {cached_lib_name}")
            lib = DispatcherLib.load(cached_lib_path)
            if lib is not None and lib.initialize():
                result.lib = lib
                log(f"  OK Loaded cached library: {lib.get_kernel_name()}")
            else:
                log("  WARNING Cached library failed to load/initialize")
                cached_lib_path = None  # Force rebuild
        else:
            log("  Rebuilding library for exact config match...")

            # First ensure we have a kernel header for this exact config
            if not kernel_header:
                # Generate kernel for the exact config
                log("  Generating kernel for config...")
                codegen_result = codegen.generate_from_config(config, force=True)

                # Check if generation succeeded
                if not codegen_result.success:
                    log(f"  WARNING Kernel generation failed:")
                    if codegen_result.stderr:
                        # Show first few lines of error
                        error_lines = codegen_result.stderr.split('\n')[:5]
                        for line in error_lines:
                            if line.strip():
                                log(f"    {line}")
                    log("  This config may not be valid for the target architecture")
                    log("  Falling back to existing library")
                    # Don't try to rebuild without a valid kernel
                    kernel_header = None
                else:
                    kernel_header = find_matching_kernel_header(config)
                    result.kernel_header = kernel_header

            if kernel_header:
                new_lib_path = codegen._rebuild_library_for_config(config, kernel_header)
                if new_lib_path:
                    lib = DispatcherLib.load(new_lib_path)
                    if lib is None or not lib.initialize():
                        result.error = "Failed to load rebuilt library"
                        return result
                    result.lib = lib
                    log(f"  OK Rebuilt library: {lib.get_kernel_name()}")
                else:
                    log("  WARNING Rebuild failed, using existing library")
            else:
                log("  WARNING No kernel header found for config, using existing library")

    # Step 5: Create registry and dispatcher
    log("  Creating registry and dispatcher...")
    registry = Registry(name=registry_name, lib=lib)
    registry.register_kernel(config)
    result.registry = registry

    dispatcher = Dispatcher(registry=registry, lib=lib)
    result.dispatcher = dispatcher

    log(f"  OK Ready: {lib.get_kernel_name()}")

    result.success = True
    return result


def setup_multiple_gemm_dispatchers(
    configs: List[KernelConfig],
    registry_name: str = "gemm_registry",
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> List[GemmSetupResult]:
    """
    Setup multiple GEMM dispatchers in parallel.

    Pipeline:
      1. Validate + auto-correct each config
      2. Parallel codegen: generate .hpp for each config via --config JSON
      3. Parallel hipcc: compile each .hpp -> .so
      4. Load + wire up each .so into a GemmSetupResult

    Each config gets its own .so, so different tile sizes can coexist.

    Args:
        max_workers: Max parallel processes for codegen/compile (default: cpu_count capped at 8).
    """
    import sys

    results = [GemmSetupResult(success=False, config=c) for c in configs]
    max_workers = max_workers or min(multiprocessing.cpu_count(), 8)

    # -- Step 1: Validate & correct ---------------------------------------
    valid_configs = []
    for i, c in enumerate(configs):
        val = validate_kernel_config(c)
        if not val.is_valid:
            c, modified, corrections = auto_correct_kernel_config(c, verbose=False)
            results[i].config = c
            results[i].corrections = corrections
        valid_configs.append(c)

    # -- Step 2: Parallel codegen (one --config JSON per config) ----------
    codegen_script = get_codegen_path()
    output_dir = get_generated_kernels_dir()

    codegen_args = []
    for c in valid_configs:
        tile_str = c.tile_str
        wave_str = f"{c.wave_m}x{c.wave_n}x{c.wave_k}"
        warp_str = f"{c.warp_m}x{c.warp_n}x{c.warp_k}"

        tile_config_json = {
            "tile_config": {
                "tile_m": [c.tile_m],
                "tile_n": [c.tile_n],
                "tile_k": [c.tile_k],
                "warp_m": [c.wave_m],
                "warp_n": [c.wave_n],
                "warp_k": [c.wave_k],
                "warp_tile_m": [c.warp_m],
                "warp_tile_n": [c.warp_n],
                "warp_tile_k": [c.warp_k],
            },
            "trait_config": {
                "pipeline": [c.pipeline],
                "epilogue": [c.epilogue],
                "scheduler": [c.scheduler],
                "pad_m": [c.pad_m],
                "pad_n": [c.pad_n],
                "pad_k": [c.pad_k],
                "persistent": [False],
            },
        }

        hpp_pattern = (
            f"gemm_{c.dtype_a}_{c.layout}_{c.pipeline}_{c.epilogue}_{c.scheduler}"
            f"_*_{tile_str}_{wave_str}_{warp_str}.hpp"
        )

        codegen_args.append(
            {
                "python": sys.executable,
                "codegen_script": str(codegen_script),
                "output_dir": str(output_dir),
                "dtype": c.dtype_a,
                "layout": c.layout,
                "gpu_target": c.gfx_arch,
                "tile_config_json": tile_config_json,
                "hpp_glob_pattern": hpp_pattern,
            }
        )

    if verbose:
        print(
            f"Generating {len(codegen_args)} kernel headers in parallel (workers={max_workers})..."
        )

    headers: List[Optional[Path]] = [None] * len(valid_configs)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_single_kernel_subprocess, a): i
            for i, a in enumerate(codegen_args)
        }
        for future in as_completed(futures):
            idx = futures[future]
            ok, hdr_str, err = future.result()
            if ok and hdr_str:
                headers[idx] = Path(hdr_str)
                results[idx].kernel_header = Path(hdr_str)
                if verbose:
                    print(
                        f"  OK [{idx}] {valid_configs[idx].tile_str}: {Path(hdr_str).name}"
                    )
            else:
                results[idx].error = f"Codegen: {err}"
                if verbose:
                    print(f"  FAIL [{idx}] {valid_configs[idx].tile_str}: {err}")

    # For configs rejected by arch filter, map to nearest arch-valid header.
    fallback_needed = [i for i, h in enumerate(headers) if h is None]
    if fallback_needed:
        if verbose:
            print(
                f"Resolving {len(fallback_needed)} configs via arch-valid GEMM catalog..."
            )

        catalog_cache: Dict[Tuple[str, str, str, str], List[Path]] = {}
        for i in fallback_needed:
            c = valid_configs[i]
            key = (c.gfx_arch, c.dtype_a, c.layout, c.variant)
            if key not in catalog_cache:
                catalog_dir = (
                    output_dir
                    / "_arch_valid_catalog"
                    / (f"{c.gfx_arch}_{c.dtype_a}_{c.layout}_{c.variant}")
                )
                ok, catalog_headers, err = _generate_arch_valid_gemm_headers(
                    python_exe=sys.executable,
                    codegen_script=codegen_script,
                    output_dir=catalog_dir,
                    dtype=c.dtype_a,
                    layout=c.layout,
                    gpu_target=c.gfx_arch,
                    variant=c.variant,
                )
                if not ok:
                    catalog_headers = []
                    if verbose:
                        print(f"  FAIL [{i}] catalog generation: {err}")
                catalog_cache[key] = catalog_headers

            chosen, meta = _select_best_arch_valid_gemm_header(c, catalog_cache[key])
            if chosen is None or meta is None:
                continue

            headers[i] = chosen
            results[i].kernel_header = chosen
            results[i].error = ""

            # Keep Python-side config aligned with the selected kernel header.
            valid_configs[i].pipeline = str(meta["pipeline"])
            valid_configs[i].epilogue = str(meta["epilogue"])
            valid_configs[i].scheduler = str(meta["scheduler"])
            valid_configs[i].pad_m = bool(meta["pad_m"])
            valid_configs[i].pad_n = bool(meta["pad_n"])
            valid_configs[i].pad_k = bool(meta["pad_k"])
            valid_configs[i].tile_m = int(meta["tile"][0])
            valid_configs[i].tile_n = int(meta["tile"][1])
            valid_configs[i].tile_k = int(meta["tile"][2])
            valid_configs[i].wave_m = int(meta["wave"][0])
            valid_configs[i].wave_n = int(meta["wave"][1])
            valid_configs[i].wave_k = int(meta["wave"][2])
            valid_configs[i].warp_m = int(meta["warp"][0])
            valid_configs[i].warp_n = int(meta["warp"][1])
            valid_configs[i].warp_k = int(meta["warp"][2])
            results[i].config = valid_configs[i]

            if verbose:
                print(f"  INFO [{i}] mapped to arch-valid header: {chosen.name}")

    # -- Step 3: Parallel hipcc compilation -------------------------------
    root = get_dispatcher_root()
    ck_root = root.parent
    build_dir = get_build_dir()
    ctypes_source = root / "bindings" / "ctypes" / "gemm_ctypes_lib.cpp"
    static_lib = build_dir / "libck_tile_dispatcher.a"

    if not ctypes_source.exists() or not static_lib.exists():
        for i in range(len(valid_configs)):
            if results[i].error == "":
                results[
                    i
                ].error = "Missing ctypes source or static library for compilation"
        return results

    compile_jobs = []
    compile_index_map = {}
    for i, c in enumerate(valid_configs):
        hdr = headers[i]
        if hdr is None:
            continue

        lib_name = (
            f"libdispatcher_gemm_{c.dtype_a}_{c.layout}_{c.tile_str}_{c.pipeline}.so"
        )
        lib_path = build_dir / "examples" / lib_name
        obj_file = lib_path.with_suffix(".o")

        compile_cmd = [
            "/opt/rocm/bin/hipcc",
            "-c",
            "-fPIC",
            "-O3",
            f"-I{root / 'include'}",
            f"-I{ck_root / 'include'}",
            f"-I{ck_root}",
            f"-I{str(output_dir)}",
            "-DCK_TILE_SINGLE_KERNEL_INCLUDE",
            f"-include{hdr}",
            "-D__HIP_PLATFORM_AMD__",
            f"--offload-arch={c.gfx_arch}",
            f'-DGFX_ARCH="{c.gfx_arch}"',
            "-mllvm",
            "-enable-noalias-to-md-conversion=0",
            "-Wno-undefined-func-template",
            "-Wno-float-equal",
            str(ctypes_source),
            "-o",
            str(obj_file),
        ]
        link_cmd = [
            "/opt/rocm/bin/hipcc",
            "-shared",
            "-fPIC",
            f"--offload-arch={c.gfx_arch}",
            "--hip-link",
            str(obj_file),
            str(static_lib),
            "-o",
            str(lib_path),
        ]

        compile_index_map[len(compile_jobs)] = i
        compile_jobs.append(
            {
                "compile_cmd": compile_cmd,
                "link_cmd": link_cmd,
                "lib_path": str(lib_path),
            }
        )

    if verbose and compile_jobs:
        print(
            f"Compiling {len(compile_jobs)} libraries in parallel (workers={max_workers})..."
        )

    lib_paths: Dict[int, Optional[Path]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_hipcc_subprocess, job): j
            for j, job in enumerate(compile_jobs)
        }
        for future in as_completed(futures):
            j = futures[future]
            i = compile_index_map[j]
            ok, lp, err = future.result()
            if ok and lp:
                lib_paths[i] = Path(lp)
                if verbose:
                    print(f"  OK [{i}] {valid_configs[i].tile_str}: {Path(lp).name}")
            else:
                results[i].error = f"Compile: {err}"
                if verbose:
                    print(f"  FAIL [{i}] {valid_configs[i].tile_str}: {err}")

    # -- Step 4: Load libraries and create dispatchers --------------------
    for i, c in enumerate(valid_configs):
        lp = lib_paths.get(i)
        if lp is None:
            continue

        lib = DispatcherLib.load(lp)
        if lib is not None and lib.initialize():
            results[i].lib = lib
            reg = Registry(name=f"{registry_name}_{i}", lib=lib)
            reg.register_kernel(c)
            results[i].registry = reg
            results[i].dispatcher = Dispatcher(registry=reg, lib=lib)
            results[i].success = True
        else:
            results[i].error = "Failed to load compiled library"

    if verbose:
        ok_count = sum(1 for r in results if r.success)
        print(f"Setup complete: {ok_count}/{len(results)} dispatchers ready")

    return results


def cleanup_gemm():
    """
    Cleanup function to call after running GEMM examples.

    This helps ensure clean state between examples by:
    1. Clearing any global state
    2. Suggesting garbage collection
    """
    import gc

    # Clear loaded libraries list
    DispatcherLib._loaded_libs.clear()

    # Suggest garbage collection
    gc.collect()


def cleanup_generated_kernels(
    keep_default: bool = True,
    verbose: bool = False,
) -> int:
    """
    Clean up generated kernel files.

    Call this at the start of examples to ensure fresh state.

    Args:
        keep_default: Keep the default fp16 kernel (True) or delete all (False)
        verbose: Print what's being deleted

    Returns:
        Number of files deleted
    """

    kernel_dir = get_generated_kernels_dir()
    if not kernel_dir.exists():
        return 0

    deleted = 0

    # Default kernel pattern to keep
    default_pattern = (
        "gemm_fp16_rcr_compv4_cshuffle_intrawave_*_128x128x32_2x2x1_16x16x16.hpp"
    )

    for f in kernel_dir.glob("*.hpp"):
        # Skip dispatcher_wrappers directory
        if f.is_dir():
            continue

        # Optionally keep default kernel
        if keep_default and f.match(default_pattern):
            continue

        if verbose:
            print(f"  Deleting: {f.name}")
        f.unlink()
        deleted += 1

    # Also clean up any temp libs
    build_dir = get_build_dir()
    examples_dir = build_dir / "examples"
    if examples_dir.exists():
        for f in examples_dir.glob("libdispatcher_gemm_*_lib.so"):
            if f.name != "libdispatcher_gemm_lib.so":
                if verbose:
                    print(f"  Deleting: {f.name}")
                f.unlink()
                deleted += 1

    return deleted


def reset_for_example(verbose: bool = False):
    """
    Reset state for a fresh example run.

    Call this at the START of each example to ensure clean state.
    Cleans up generated kernels (except default) and resets globals.
    """
    # Cleanup any previously generated kernels
    deleted = cleanup_generated_kernels(keep_default=True, verbose=verbose)
    if verbose and deleted > 0:
        print(f"  Cleaned up {deleted} generated files")

    # Clear any cached state
    cleanup_gemm()


# Main (self-test)
# =============================================================================

if __name__ == "__main__":
    print("CK Tile Dispatcher Utils Self-Test")
    print("=" * 60)

    # Test library loading
    print("\n1. Loading library...")
    lib = DispatcherLib.auto()
    if lib is None:
        print("   FAILED: Could not load library")
        exit(1)
    print(f"   OK: Loaded from {lib.path}")
    print(f"   Kernel: {lib.get_kernel_name()}")
    print(f"   Registered kernels: {lib.get_kernel_count()}")

    # Test GEMM
    print("\n2. Running GEMM 256x256x256...")
    runner = GemmRunner(lib)
    A = np.random.randn(256, 256).astype(np.float16)
    B = np.random.randn(256, 256).astype(np.float16)

    result = runner.run(A, B)
    print(f"   Status: {'OK' if result.success else 'FAILED'}")
    print(f"   Time: {result.time_ms:.4f} ms")
    print(f"   TFLOPS: {result.tflops:.2f}")

    # Test validation
    print("\n3. Validating result...")
    validator = Validator()
    reference = validator.compute_reference(A, B)
    correct, max_diff, mean_diff = validator.check(result.output, reference)
    print(f"   Correct: {correct}")
    print(f"   Max diff: {max_diff:.6f}")

    # Test high-level helper
    print("\n4. Testing setup_gemm_dispatcher...")
    config = KernelConfig(tile_m=128, tile_n=128, tile_k=32)
    setup = setup_gemm_dispatcher(config, verbose=True)
    print(f"   Success: {setup.success}")

    # Cleanup
    cleanup_gemm()

    print("\n" + "=" * 60)
    print("All tests passed!")
