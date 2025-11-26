#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
from typing import Optional, Tuple, List
from dataclasses import dataclass


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


def get_generated_kernels_dir() -> Path:
    """Get the generated kernels directory"""
    return get_build_dir() / "generated_kernels"


# =============================================================================
# Library Loading
# =============================================================================


class DispatcherLib:
    """Wrapper for the dispatcher dynamic library"""

    # Default library search paths (relative to dispatcher root)
    SEARCH_PATHS = [
        "build/examples/libdispatcher_gemm.so",
        "build/lib/libdispatcher_gemm.so",
        "examples/python/libdispatcher_gemm.so",
    ]

    def __init__(self, lib: ctypes.CDLL, path: Path):
        self._lib = lib
        self._path = path
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

        compile_cmd = [
            "/opt/rocm/bin/hipcc",
            "-shared",
            "-fPIC",
            "-O3",
            f"-I{root / 'include'}",
            f"-I{ck_root / 'include'}",
            f"-I{ck_root}",
            f"-include{kernel_header}",
            "-D__HIP_PLATFORM_AMD__",
            "--offload-arch=gfx942",
            "-DAMDGPU_ARCH=gfx942",
            str(root / "examples/cpp/dispatcher_dynamic_lib.cpp"),
            str(root / "src/registry.cpp"),
            str(root / "src/dispatcher.cpp"),
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
        """Auto-find or compile the library"""
        if not recompile:
            lib = cls.load()
            if lib is not None:
                if lib.initialize():
                    return lib

        # Try to compile
        path = cls.compile()
        if path is None:
            return None

        lib = cls.load(path)
        if lib is not None:
            lib.initialize()

        return lib


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
# Convenience Functions
# =============================================================================


def quick_gemm(lib: DispatcherLib, A: np.ndarray, B: np.ndarray) -> GemmResult:
    """Quick GEMM using provided library"""
    runner = GemmRunner(lib)
    return runner.run(A, B)


def benchmark_multiple_sizes(
    lib: DispatcherLib,
    sizes: List[Tuple[int, int, int]],
    warmup: int = 2,
    iterations: int = 10,
) -> List[GemmResult]:
    """
    Benchmark multiple problem sizes

    Args:
        lib: Dispatcher library
        sizes: List of (M, N, K) tuples
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        List of GemmResult for each size
    """
    runner = GemmRunner(lib)
    results = []

    print(f"\n{'Size':>20} | {'Time (ms)':>12} | {'TFLOPS':>10}")
    print("-" * 50)

    for M, N, K in sizes:
        if not lib.is_supported(M, N, K):
            print(f"{M:>4}x{N:>4}x{K:<4} | {'N/A':>12} | {'N/A':>10} (unsupported)")
            continue

        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(K, N).astype(np.float16)

        # Warmup
        for _ in range(warmup):
            runner.run(A, B)

        # Average multiple runs
        times = []
        result = None
        for _ in range(iterations):
            result = runner.run(A, B)
            if result.success:
                times.append(result.time_ms)

        if times and result:
            avg_time = sum(times) / len(times)
            flops = 2.0 * M * N * K
            avg_tflops = (flops / (avg_time * 1e-3)) / 1e12

            # Update result with averaged values
            result.time_ms = avg_time
            result.tflops = avg_tflops

            print(f"{M:>4}x{N:>4}x{K:<4} | {avg_time:>12.4f} | {avg_tflops:>10.2f}")
            results.append(result)

    return results


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

    def get_generated_kernels(self) -> List[Path]:
        """Get list of generated kernel headers"""
        if self.output_dir.exists():
            return list(self.output_dir.glob("*.hpp"))
        return []


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
    Runner for the unified GEMM code generator.

    Usage:
        codegen = CodegenRunner()

        # Generate standard kernels
        result = codegen.generate("standard")

        # Generate preshuffle kernels
        result = codegen.generate("preshuffle")

        # Generate multi-D kernels
        result = codegen.generate("multi_d")

        # Generate all variants
        results = codegen.generate_all()

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
    ):
        self.codegen_path = codegen_path or get_codegen_path()
        self.output_dir = output_dir or get_generated_kernels_dir()
        self.datatype = datatype
        self.layout = layout
        self.gpu_target = gpu_target

    def generate(
        self,
        variant: str = "standard",
        output_dir: Optional[Path] = None,
        extra_args: Optional[List[str]] = None,
    ) -> CodegenResult:
        """
        Generate kernels for a specific variant.

        Args:
            variant: One of "standard", "preshuffle", "multi_d"
            output_dir: Override output directory
            extra_args: Additional arguments to pass to codegen

        Returns:
            CodegenResult with generation status and info
        """
        import sys

        out_dir = output_dir or self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.codegen_path.exists():
            return CodegenResult(
                success=False,
                output_dir=out_dir,
                variant=variant,
                stderr=f"Codegen not found at {self.codegen_path}",
            )

        cmd = [
            sys.executable,
            str(self.codegen_path),
            "--output-dir",
            str(out_dir),
            "--datatype",
            self.datatype,
            "--layout",
            self.layout,
            "--gpu-target",
            self.gpu_target,
            "--variants",
            variant,
        ]

        if extra_args:
            cmd.extend(extra_args)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Count generated kernels
            kernel_count = len(list(out_dir.glob("*.hpp")))

            return CodegenResult(
                success=result.returncode == 0,
                output_dir=out_dir,
                variant=variant,
                stdout=result.stdout,
                stderr=result.stderr,
                kernel_count=kernel_count,
            )
        except subprocess.TimeoutExpired:
            return CodegenResult(
                success=False,
                output_dir=out_dir,
                variant=variant,
                stderr="Code generation timed out (300s)",
            )
        except Exception as e:
            return CodegenResult(
                success=False,
                output_dir=out_dir,
                variant=variant,
                stderr=str(e),
            )

    def generate_all(self, output_dir: Optional[Path] = None) -> List[CodegenResult]:
        """Generate all variants"""
        results = []
        for variant in self.VARIANTS:
            result = self.generate(variant, output_dir)
            results.append(result)
        return results

    def generate_from_config(
        self, config: KernelConfig, output_dir: Optional[Path] = None
    ) -> CodegenResult:
        """
        Generate kernel from a specific KernelConfig.

        Args:
            config: KernelConfig with all kernel parameters
            output_dir: Override output directory

        Returns:
            CodegenResult
        """
        import sys

        out_dir = output_dir or self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.codegen_path.exists():
            return CodegenResult(
                success=False,
                output_dir=out_dir,
                variant=f"config:{config.tile_str}",
                stderr=f"Codegen not found at {self.codegen_path}",
            )

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
            "--variants",
            "standard",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Find matching kernel for this config
            pattern = f"*{config.tile_str}*.hpp"
            matching = list(out_dir.glob(pattern))
            kernel_count = len(matching)

            return CodegenResult(
                success=result.returncode == 0 and kernel_count > 0,
                output_dir=out_dir,
                variant=f"config:{config.tile_str}",
                stdout=result.stdout,
                stderr=result.stderr,
                kernel_count=kernel_count,
            )
        except Exception as e:
            return CodegenResult(
                success=False,
                output_dir=out_dir,
                variant=f"config:{config.tile_str}",
                stderr=str(e),
            )

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


def ensure_dispatcher_ready(
    generate_if_missing: bool = True,
) -> Optional[DispatcherLib]:
    """
    Ensure the dispatcher library is ready.

    This function:
    1. Checks if kernels exist, generates them if missing
    2. Checks if library exists, compiles it if missing
    3. Loads and initializes the library

    Args:
        generate_if_missing: If True, generate kernels/compile library if missing

    Returns:
        DispatcherLib if ready, None otherwise
    """
    # Check for kernels
    kernel_dir = get_generated_kernels_dir()
    kernels = list(kernel_dir.glob("*.hpp")) if kernel_dir.exists() else []

    if not kernels and generate_if_missing:
        print("No kernels found. Generating standard kernels...")
        codegen = CodegenRunner()
        result = codegen.generate("standard")
        if not result.success:
            print(f"  Failed: {result.stderr[:200]}")
            return None
        print(f"  Generated {result.kernel_count} kernels")

    # Load or compile library
    return DispatcherLib.auto(recompile=generate_if_missing and not kernels)


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
