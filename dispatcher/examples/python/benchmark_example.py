#!/usr/bin/env python3
"""
Benchmark Example

Comprehensive benchmarking of dispatcher GEMM performance from Python.
Tests various problem sizes and reports detailed metrics.
"""

import sys
import numpy as np
import ctypes
from pathlib import Path
import subprocess
import time
from dataclasses import dataclass
from typing import List, Tuple

# Setup paths
DISPATCHER_ROOT = Path(__file__).parent.parent.parent
BUILD_DIR = DISPATCHER_ROOT / "build"
KERNELS_DIR = BUILD_DIR / "generated_kernels"
EXAMPLES_BUILD_DIR = BUILD_DIR / "examples"


@dataclass
class BenchmarkResult:
    M: int
    N: int
    K: int
    min_ms: float
    max_ms: float
    avg_ms: float
    median_ms: float
    tflops: float
    bandwidth_gb: float


def ensure_library():
    """Ensure the dynamic library exists"""
    lib_path = EXAMPLES_BUILD_DIR / "libdispatcher_gemm.so"
    
    if lib_path.exists():
        return lib_path
    
    print("Compiling dynamic library...")
    lib_source = DISPATCHER_ROOT / "examples" / "cpp" / "dispatcher_dynamic_lib.cpp"
    kernel_header = KERNELS_DIR / "gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_128x128x32_2x2x1_32x32x16.hpp"
    
    if not kernel_header.exists():
        print(f"Kernel header not found: {kernel_header}")
        return None
    
    EXAMPLES_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    compile_cmd = [
        '/opt/rocm/bin/hipcc',
        '-std=c++17', '-O3', '-shared', '-fPIC',
        f'-I{DISPATCHER_ROOT}/include',
        f'-I{DISPATCHER_ROOT.parent}/include',
        f'-I{KERNELS_DIR}',
        f'-include', str(kernel_header),
        '-mllvm', '-enable-noalias-to-md-conversion=0',
        '-Wno-undefined-func-template', '-Wno-float-equal',
        '--offload-arch=gfx942', '--offload-compress',
        str(lib_source),
        f'-L{BUILD_DIR}', '-lck_tile_dispatcher',
        '-o', str(lib_path)
    ]
    
    result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)
    
    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        return None
    
    return lib_path


def load_library(lib_path):
    """Load the dispatcher library"""
    lib = ctypes.CDLL(str(lib_path))
    
    lib.dispatcher_initialize.argtypes = []
    lib.dispatcher_initialize.restype = ctypes.c_int
    
    lib.dispatcher_run_gemm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.dispatcher_run_gemm.restype = ctypes.c_int
    
    lib.dispatcher_cleanup.argtypes = []
    lib.dispatcher_cleanup.restype = None
    
    return lib


def benchmark_size(lib, M: int, N: int, K: int, warmup_runs: int = 3, bench_runs: int = 10) -> BenchmarkResult:
    """Benchmark a single problem size"""
    
    # Create test matrices
    A = np.ones((M, K), dtype=np.float16, order='C')
    B = np.ones((K, N), dtype=np.float16, order='F')
    C = np.zeros((M, N), dtype=np.float16, order='C')
    
    A_ptr = A.ctypes.data_as(ctypes.c_void_p)
    B_ptr = B.ctypes.data_as(ctypes.c_void_p)
    C_ptr = C.ctypes.data_as(ctypes.c_void_p)
    time_ms = ctypes.c_float()
    
    # Warmup
    for _ in range(warmup_runs):
        lib.dispatcher_run_gemm(A_ptr, B_ptr, C_ptr, M, N, K, ctypes.byref(time_ms))
    
    # Benchmark
    times = []
    for _ in range(bench_runs):
        status = lib.dispatcher_run_gemm(A_ptr, B_ptr, C_ptr, M, N, K, ctypes.byref(time_ms))
        if status == 0:
            times.append(time_ms.value)
    
    if not times:
        return BenchmarkResult(M, N, K, 0, 0, 0, 0, 0, 0)
    
    # Calculate statistics
    times.sort()
    min_ms = times[0]
    max_ms = times[-1]
    avg_ms = sum(times) / len(times)
    median_ms = times[len(times) // 2]
    
    # Performance metrics
    flops = 2.0 * M * N * K
    tflops = flops / (min_ms * 1e9)
    
    # Memory bandwidth
    bytes_transferred = (M * K + K * N + M * N) * 2  # FP16 = 2 bytes
    bandwidth_gb = bytes_transferred / (min_ms * 1e6)
    
    return BenchmarkResult(M, N, K, min_ms, max_ms, avg_ms, median_ms, tflops, bandwidth_gb)


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a nice table"""
    print()
    print(f"{'Size':>20} {'Min (ms)':>12} {'Avg (ms)':>12} {'Med (ms)':>12} {'Max (ms)':>12} {'TFLOPS':>12} {'BW (GB/s)':>12}")
    print("-" * 92)
    
    for r in results:
        size_str = f"{r.M}x{r.N}x{r.K}"
        print(f"{size_str:>20} {r.min_ms:>12.4f} {r.avg_ms:>12.4f} {r.median_ms:>12.4f} {r.max_ms:>12.4f} {r.tflops:>12.2f} {r.bandwidth_gb:>12.2f}")


def main():
    print("=" * 70)
    print("CK Tile Dispatcher - Python Benchmark Example")
    print("=" * 70)
    print()
    
    # Ensure library exists
    lib_path = ensure_library()
    if lib_path is None:
        print("Failed to get library")
        return 1
    
    print(f"Library: {lib_path}")
    
    # Load library
    lib = load_library(lib_path)
    
    # Initialize
    status = lib.dispatcher_initialize()
    if status != 0:
        print("Initialization failed")
        return 1
    
    print("Dispatcher initialized")
    
    # Benchmark configuration
    warmup_runs = 3
    bench_runs = 10
    
    print(f"Warmup runs: {warmup_runs}")
    print(f"Benchmark runs: {bench_runs}")
    
    # Test sizes
    sizes = [
        # Square sizes
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        
        # Rectangular sizes
        (512, 512, 2048),
        (512, 2048, 512),
        (2048, 512, 512),
        
        # Common deep learning sizes
        (1024, 4096, 1024),
        (4096, 1024, 1024),
    ]
    
    print("\nRunning benchmarks...")
    
    results = []
    for M, N, K in sizes:
        print(f"  {M}x{N}x{K}...", end="", flush=True)
        result = benchmark_size(lib, M, N, K, warmup_runs, bench_runs)
        results.append(result)
        print(f" {result.tflops:.2f} TFLOPS")
    
    # Print results
    print_results(results)
    
    # Summary
    max_tflops = max(r.tflops for r in results)
    
    print()
    print("=" * 70)
    print(f"Peak Performance: {max_tflops:.2f} TFLOPS")
    print("=" * 70)
    
    # Cleanup
    lib.dispatcher_cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

