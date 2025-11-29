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
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time


# =============================================================================
# Path Configuration
# =============================================================================


def get_dispatcher_root() -> Path:
    """Get the dispatcher root directory"""
    # This file is in dispatcher/examples/gemm/python/
    return Path(__file__).parent.parent.parent.parent


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
        "build/examples/libdispatcher_gemm_lib.so",
        "build/libdispatcher_gemm_lib.so",
        "build/examples/libdispatcher_gemm.so",
        "build/lib/libdispatcher_gemm.so",
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
                        status = "✓" if result.success else "✗"
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
                        print(f"  ✗ {variant}: FAILED - {e}")

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
                        status = "✓" if result.success else "✗"
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
                        print(f"  ✗ {tile_str}: FAILED - {e}")

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
                        status = "✓" if result.success else "✗"
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
                        print(f"  ✗ {variant}: FAILED - {e}")

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

        This method is smart: it checks if the specific kernel already exists
        and skips generation if so (unless force=True).

        Args:
            config: KernelConfig with all kernel parameters
            output_dir: Override output directory
            force: Force regeneration even if kernel exists
            show_instances: Print instance names when generating

        Returns:
            CodegenResult with only the EXACT matching kernel counted
        """
        import sys

        out_dir = output_dir or self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build PRECISE kernel filename pattern for this specific config
        # Format: gemm_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_{pads}_{tile}_{wave}_{warp}
        tile_str = config.tile_str  # e.g., "128x128x32"
        wave_str = f"{config.wave_m}x{config.wave_n}x{config.wave_k}"  # e.g., "2x2x1"
        warp_str = (
            f"{config.warp_m}x{config.warp_n}x{config.warp_k}"  # e.g., "32x32x16"
        )

        # Build precise pattern including pipeline and epilogue
        # Format: gemm_fp16_rcr_compv4_cshuffle_intrawave_*_128x128x32_2x2x1_32x32x16.hpp
        # Matches standard kernels ending with .hpp (NOT _preshuffle.hpp or _multid_*.hpp)
        precise_pattern = f"gemm_{config.dtype_a}_{config.layout}_{config.pipeline}_{config.epilogue}_{config.scheduler}_*_{tile_str}_{wave_str}_{warp_str}.hpp"

        # Check if exact kernel already exists - skip expensive generation
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
                stdout=f"Kernel already exists ({len(existing)} variants), skipped generation",
            )

        if not self.codegen_path.exists():
            return CodegenResult(
                success=False,
                output_dir=out_dir,
                variant=f"config:{tile_str}",
                stderr=f"Codegen not found at {self.codegen_path}",
            )

        start = time.time()

        # Generate standard kernels (codegen generates all tile sizes)
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

            # Find ONLY the EXACT matching kernel(s) for this specific config
            matching = list(out_dir.glob(precise_pattern))
            kernel_count = len(matching)
            elapsed = time.time() - start

            instance_names = sorted([k.stem for k in matching])
            if show_instances and instance_names:
                for name in instance_names:
                    print(f"  Adding Instance: {name}")
                    print(f"  Building Instance: {name}")

            return CodegenResult(
                success=result.returncode == 0 and kernel_count > 0,
                output_dir=out_dir,
                variant=f"config:{tile_str}",
                stdout=result.stdout,
                stderr=result.stderr,
                kernel_count=kernel_count,  # Only count EXACT matching kernels
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
