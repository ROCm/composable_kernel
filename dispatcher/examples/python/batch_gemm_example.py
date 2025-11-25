#!/usr/bin/env python3
"""
Batch GEMM Example

Demonstrates running multiple GEMM operations with different sizes,
simulating a typical deep learning workload with varying tensor shapes.
"""

import sys
import numpy as np
import ctypes
from pathlib import Path
import subprocess
from typing import List
from dataclasses import dataclass

# Setup paths
DISPATCHER_ROOT = Path(__file__).parent.parent.parent
BUILD_DIR = DISPATCHER_ROOT / "build"
KERNELS_DIR = BUILD_DIR / "generated_kernels"
EXAMPLES_BUILD_DIR = BUILD_DIR / "examples"


@dataclass
class GemmResult:
    name: str
    M: int
    N: int
    K: int
    time_ms: float
    tflops: float
    correct: bool


def ensure_library():
    """Ensure the dynamic library exists"""
    lib_path = EXAMPLES_BUILD_DIR / "libdispatcher_gemm.so"

    if lib_path.exists():
        return lib_path

    print("Compiling dynamic library...")
    lib_source = DISPATCHER_ROOT / "examples" / "cpp" / "dispatcher_dynamic_lib.cpp"
    kernel_header = (
        KERNELS_DIR
        / "gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_128x128x32_2x2x1_32x32x16.hpp"
    )

    if not kernel_header.exists():
        print(f"Kernel header not found: {kernel_header}")
        return None

    EXAMPLES_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    compile_cmd = [
        "/opt/rocm/bin/hipcc",
        "-std=c++17",
        "-O3",
        "-shared",
        "-fPIC",
        f"-I{DISPATCHER_ROOT}/include",
        f"-I{DISPATCHER_ROOT.parent}/include",
        f"-I{KERNELS_DIR}",
        "-include",
        str(kernel_header),
        "-mllvm",
        "-enable-noalias-to-md-conversion=0",
        "-Wno-undefined-func-template",
        "-Wno-float-equal",
        "--offload-arch=gfx942",
        "--offload-compress",
        str(lib_source),
        f"-L{BUILD_DIR}",
        "-lck_tile_dispatcher",
        "-o",
        str(lib_path),
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
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.dispatcher_run_gemm.restype = ctypes.c_int

    # New: check if size is supported
    lib.dispatcher_is_supported.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.dispatcher_is_supported.restype = ctypes.c_int

    lib.dispatcher_cleanup.argtypes = []
    lib.dispatcher_cleanup.restype = None

    return lib


def run_gemm(lib, name: str, A: np.ndarray, B: np.ndarray) -> GemmResult:
    """Run a single GEMM and validate result"""

    M, K = A.shape
    _, N = B.shape

    # First check if this size is supported
    is_supported = lib.dispatcher_is_supported(M, N, K)
    if not is_supported:
        # Return a result indicating unsupported size
        return GemmResult(name, M, N, K, -1, 0, False)

    # Output matrix
    C = np.zeros((M, N), dtype=np.float16, order="C")

    # Get pointers
    A_ptr = A.ctypes.data_as(ctypes.c_void_p)
    B_ptr = B.ctypes.data_as(ctypes.c_void_p)
    C_ptr = C.ctypes.data_as(ctypes.c_void_p)
    time_ms = ctypes.c_float()

    # Run GEMM
    status = lib.dispatcher_run_gemm(
        A_ptr, B_ptr, C_ptr, M, N, K, ctypes.byref(time_ms)
    )

    if status == -2:
        # No suitable kernel - return unsupported
        return GemmResult(name, M, N, K, -1, 0, False)
    elif status != 0:
        # Other error
        return GemmResult(name, M, N, K, 0, 0, False)

    # Calculate performance
    flops = 2.0 * M * N * K
    tflops = flops / (time_ms.value * 1e9) if time_ms.value > 0 else 0

    # Validate: for all-ones matrices, result should be K
    expected = float(K)
    correct_count = np.sum(np.abs(C - expected) < 1.0)
    correct = correct_count > (M * N * 0.99)  # 99% correct

    return GemmResult(name, M, N, K, time_ms.value, tflops, correct)


def main():
    print("=" * 70)
    print("CK Tile Dispatcher - Batch GEMM Example")
    print("=" * 70)
    print()
    print("Simulating a deep learning workload with various GEMM sizes")
    print()

    # Ensure library exists
    lib_path = ensure_library()
    if lib_path is None:
        print("Failed to get library")
        return 1

    # Load library
    lib = load_library(lib_path)

    # Initialize
    status = lib.dispatcher_initialize()
    if status != 0:
        print("Initialization failed")
        return 1

    print("Dispatcher initialized")
    print()

    # Define batch of GEMM operations (simulating a transformer layer)
    # Note: Dimensions must be compatible with tile sizes (multiples of 128 for this kernel)
    batch_operations = [
        # QKV projection: (batch*seq, hidden) x (hidden, 3*hidden)
        ("QKV Projection", 1024, 3072, 1024),
        # Attention: Q x K^T (adjusted for tile compatibility)
        ("Attention QK", 256, 256, 128),
        # Attention: scores x V (adjusted for tile compatibility)
        ("Attention V", 256, 128, 256),
        # Output projection: (batch*seq, hidden) x (hidden, hidden)
        ("Output Projection", 1024, 1024, 1024),
        # FFN layer 1: (batch*seq, hidden) x (hidden, 4*hidden)
        ("FFN Expand", 1024, 4096, 1024),
        # FFN layer 2: (batch*seq, 4*hidden) x (4*hidden, hidden)
        ("FFN Contract", 1024, 1024, 4096),
        # Additional operations (adjusted for tile compatibility)
        ("Embedding Lookup", 512, 1024, 256),
        ("Classification Head", 256, 1024, 1024),
    ]

    print(f"Running {len(batch_operations)} GEMM operations:")
    print("-" * 70)

    results: List[GemmResult] = []
    total_time = 0.0
    total_flops = 0

    for name, M, N, K in batch_operations:
        # Create test matrices (all ones for easy validation)
        A = np.ones((M, K), dtype=np.float16, order="C")
        B = np.ones((K, N), dtype=np.float16, order="F")

        result = run_gemm(lib, name, A, B)
        results.append(result)

        # Handle unsupported sizes (time_ms == -1)
        if result.time_ms >= 0:
            total_time += result.time_ms
            total_flops += 2 * M * N * K
            status = "OK" if result.correct else "FAIL"
            print(
                f"  {name:20s} {M:5d}x{N:5d}x{K:5d}  {result.time_ms:8.4f} ms  {result.tflops:6.2f} TFLOPS  [{status}]"
            )
        else:
            print(
                f"  {name:20s} {M:5d}x{N:5d}x{K:5d}  {'skipped':>8s}     {'---':>6s} TFLOPS  [UNSUPPORTED]"
            )

    print("-" * 70)

    # Summary
    supported_results = [r for r in results if r.time_ms >= 0]
    unsupported_count = len(results) - len(supported_results)
    all_correct = (
        all(r.correct for r in supported_results) if supported_results else False
    )
    avg_tflops = (total_flops / total_time) / 1e9 if total_time > 0 else 0

    print()
    print("Summary:")
    print(f"  Total operations: {len(batch_operations)}")
    print(f"  Executed: {len(supported_results)}")
    if unsupported_count > 0:
        print(
            f"  Unsupported sizes: {unsupported_count} (need additional kernel configs)"
        )
    print(f"  Total time: {total_time:.4f} ms")
    print(f"  Average TFLOPS: {avg_tflops:.2f}")
    print(f"  All correct: {'Yes' if all_correct else 'No'}")
    print()

    # Per-operation breakdown
    print("Performance breakdown:")
    print()
    print(
        f"{'Operation':25s} {'Size':20s} {'Time (ms)':>12s} {'% Total':>10s} {'TFLOPS':>10s}"
    )
    print("-" * 80)

    for r in results:
        pct = (r.time_ms / total_time * 100) if total_time > 0 else 0
        size_str = f"{r.M}x{r.N}x{r.K}"
        print(
            f"{r.name:25s} {size_str:20s} {r.time_ms:>12.4f} {pct:>10.1f}% {r.tflops:>10.2f}"
        )

    print()
    print("=" * 70)
    print("Batch GEMM Example Complete")
    print("=" * 70)

    # Cleanup
    lib.dispatcher_cleanup()

    return 0 if all_correct else 1


if __name__ == "__main__":
    sys.exit(main())
