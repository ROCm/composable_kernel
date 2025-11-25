#!/usr/bin/env python3
"""
Validation Example

Comprehensive validation of GPU GEMM results against NumPy reference.
Tests various input patterns and validates numerical accuracy.
"""

import sys
import numpy as np
import ctypes
from pathlib import Path
import subprocess
from typing import Tuple

# Setup paths
DISPATCHER_ROOT = Path(__file__).parent.parent.parent
BUILD_DIR = DISPATCHER_ROOT / "build"
KERNELS_DIR = BUILD_DIR / "generated_kernels"
EXAMPLES_BUILD_DIR = BUILD_DIR / "examples"


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

    lib.dispatcher_cleanup.argtypes = []
    lib.dispatcher_cleanup.restype = None

    return lib


def run_gpu_gemm(lib, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    """Run GEMM on GPU"""
    M, K = A.shape
    _, N = B.shape

    C = np.zeros((M, N), dtype=np.float16, order="C")

    A_ptr = A.ctypes.data_as(ctypes.c_void_p)
    B_ptr = B.ctypes.data_as(ctypes.c_void_p)
    C_ptr = C.ctypes.data_as(ctypes.c_void_p)
    time_ms = ctypes.c_float()

    status = lib.dispatcher_run_gemm(
        A_ptr, B_ptr, C_ptr, M, N, K, ctypes.byref(time_ms)
    )

    if status != 0:
        raise RuntimeError("GEMM execution failed")

    return C, time_ms.value


def validate_test(
    lib, name: str, A: np.ndarray, B: np.ndarray, expected: np.ndarray = None
) -> bool:
    """Run a validation test"""
    print(f"\nTest: {name}")
    print(f"  Size: A{A.shape} x B{B.shape}")

    # GPU GEMM
    C_gpu, time_ms = run_gpu_gemm(lib, A, B)

    # NumPy reference
    if expected is None:
        expected = np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(
            np.float16
        )

    # Compare
    diff = np.abs(C_gpu.astype(np.float32) - expected.astype(np.float32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Use relative tolerance based on expected magnitude
    expected_abs = np.abs(expected.astype(np.float32))
    rel_tol = np.maximum(expected_abs * 0.01, 0.5)  # 1% relative or 0.5 absolute
    correct_count = np.sum(diff < rel_tol)
    accuracy = 100.0 * correct_count / (A.shape[0] * B.shape[1])

    print(f"  GPU Time: {time_ms:.4f} ms")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Accuracy: {accuracy:.2f}%")

    passed = accuracy > 95.0
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    print("=" * 70)
    print("CK Tile Dispatcher - Validation Example")
    print("=" * 70)
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

    tests_passed = 0
    tests_total = 0

    # Test 1: All ones
    print("\n" + "-" * 70)
    print("Test Category: Simple Patterns")
    print("-" * 70)

    M, N, K = 256, 256, 256
    A = np.ones((M, K), dtype=np.float16, order="C")
    B = np.ones((K, N), dtype=np.float16, order="F")
    expected = np.full((M, N), K, dtype=np.float16)

    tests_total += 1
    if validate_test(lib, "All Ones", A, B, expected):
        tests_passed += 1

    # Test 2: Identity matrix
    A = np.eye(M, K, dtype=np.float16, order="C")
    B = np.ones((K, N), dtype=np.float16, order="F")

    tests_total += 1
    if validate_test(lib, "Identity x Ones", A, B):
        tests_passed += 1

    # Test 3: Small integer values
    A = (np.arange(M * K).reshape(M, K) % 10).astype(np.float16, order="C")
    B = (np.arange(K * N).reshape(K, N) % 10).astype(np.float16, order="F")

    tests_total += 1
    if validate_test(lib, "Small Integers (0-9)", A, B):
        tests_passed += 1

    # Test 4: Random uniform
    print("\n" + "-" * 70)
    print("Test Category: Random Data")
    print("-" * 70)

    np.random.seed(42)
    A = np.random.uniform(-1, 1, (M, K)).astype(np.float16, order="C")
    B = np.random.uniform(-1, 1, (K, N)).astype(np.float16, order="F")

    tests_total += 1
    if validate_test(lib, "Random Uniform [-1, 1]", A, B):
        tests_passed += 1

    # Test 5: Random normal
    A = np.random.randn(M, K).astype(np.float16, order="C")
    B = np.random.randn(K, N).astype(np.float16, order="F")

    tests_total += 1
    if validate_test(lib, "Random Normal", A, B):
        tests_passed += 1

    # Test 6: Different sizes
    print("\n" + "-" * 70)
    print("Test Category: Various Sizes")
    print("-" * 70)

    sizes = [
        (128, 128, 128),
        (512, 512, 512),
        (256, 512, 128),
        (512, 128, 256),
        (1024, 1024, 256),
    ]

    for M, N, K in sizes:
        A = np.random.randn(M, K).astype(np.float16, order="C") * 0.1
        B = np.random.randn(K, N).astype(np.float16, order="F") * 0.1

        tests_total += 1
        if validate_test(lib, f"Size {M}x{N}x{K}", A, B):
            tests_passed += 1

    # Test 7: Edge cases
    print("\n" + "-" * 70)
    print("Test Category: Edge Cases")
    print("-" * 70)

    # Very small values
    M, N, K = 256, 256, 256
    A = np.ones((M, K), dtype=np.float16, order="C") * 0.001
    B = np.ones((K, N), dtype=np.float16, order="F") * 0.001

    tests_total += 1
    if validate_test(lib, "Very Small Values (0.001)", A, B):
        tests_passed += 1

    # Mixed positive/negative
    A = np.ones((M, K), dtype=np.float16, order="C")
    A[::2, :] = -1  # Alternate rows
    B = np.ones((K, N), dtype=np.float16, order="F")

    tests_total += 1
    if validate_test(lib, "Mixed Signs", A, B):
        tests_passed += 1

    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    print(f"Pass rate: {100.0 * tests_passed / tests_total:.1f}%")

    if tests_passed == tests_total:
        print("\nAll validation tests PASSED!")
        result = 0
    else:
        print(f"\nWARNING: {tests_total - tests_passed} test(s) FAILED")
        result = 1

    print("=" * 70)

    # Cleanup
    lib.dispatcher_cleanup()

    return result


if __name__ == "__main__":
    sys.exit(main())
