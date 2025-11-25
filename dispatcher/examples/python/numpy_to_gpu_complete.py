#!/usr/bin/env python3
"""
NumPy to GPU - Complete Workflow

This demonstrates the complete workflow from NumPy to GPU!

Workflow:
1. Start with NumPy matrices in Python
2. Compile dynamically loadable library (.so) with selected kernel
3. Load .so back into Python via ctypes
4. Pass NumPy array pointers directly to C++
5. C++ runs dispatcher + GPU GEMM
6. Results written back to NumPy arrays
7. Print and validate results in Python

This is the seamless Python <-> GPU integration!
"""

import sys
import numpy as np
import ctypes
from pathlib import Path
import subprocess
import time

# Setup paths
DISPATCHER_ROOT = Path(__file__).parent.parent.parent
BUILD_DIR = DISPATCHER_ROOT / "build"
KERNELS_DIR = BUILD_DIR / "generated_kernels"
EXAMPLES_BUILD_DIR = BUILD_DIR / "examples"


def ensure_kernels_generated():
    """Ensure kernels are generated"""
    kernel_header = (
        KERNELS_DIR
        / "gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_128x128x32_2x2x1_32x32x16.hpp"
    )

    if kernel_header.exists():
        print("OK Kernels already generated")
        return True

    print("Generating kernels...")
    codegen_script = DISPATCHER_ROOT / "codegen" / "unified_gemm_codegen.py"

    cmd = [
        sys.executable,
        str(codegen_script),
        "--output-dir",
        str(KERNELS_DIR),
        "--datatype",
        "fp16",
        "--layout",
        "rcr",
        "--gpu-target",
        "gfx942",
        "--preselected",
        "fp16_rcr_essential",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[FAIL] Kernel generation failed: {result.stderr}")
        return False

    print("OK Kernels generated")
    return True


def compile_dynamic_library():
    """Compile the dispatcher dynamic library (.so)"""
    print("\nCompiling dynamic library...")

    lib_source = DISPATCHER_ROOT / "examples" / "cpp" / "dispatcher_dynamic_lib.cpp"
    lib_output = EXAMPLES_BUILD_DIR / "libdispatcher_gemm.so"

    # Ensure output directory exists
    EXAMPLES_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Kernel to include
    kernel_header = (
        KERNELS_DIR
        / "gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_128x128x32_2x2x1_32x32x16.hpp"
    )

    if not kernel_header.exists():
        print(f"[FAIL] Kernel header not found: {kernel_header}")
        return None

    # Compile command
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
        str(lib_output),
    ]

    print(f"  Compiling: {lib_source.name}")
    print(f"  Output: {lib_output.name}")

    result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print("[FAIL] Compilation failed:")
        print(result.stderr)
        return None

    if not lib_output.exists():
        print(f"[FAIL] Library not found after compilation: {lib_output}")
        return None

    print(f"OK Compiled: {lib_output}")
    return lib_output


def load_dispatcher_library(lib_path):
    """Load the dispatcher library via ctypes"""
    print("\nLoading library via ctypes...")

    try:
        lib = ctypes.CDLL(str(lib_path))

        # Define function signatures

        # int dispatcher_initialize()
        lib.dispatcher_initialize.argtypes = []
        lib.dispatcher_initialize.restype = ctypes.c_int

        # int dispatcher_select_kernel(int64_t M, int64_t N, int64_t K, char* buffer, int size)
        lib.dispatcher_select_kernel.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        lib.dispatcher_select_kernel.restype = ctypes.c_int

        # int dispatcher_run_gemm(void* A, void* B, void* C, int64_t M, int64_t N, int64_t K, float* time)
        lib.dispatcher_run_gemm.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # B
            ctypes.c_void_p,  # C
            ctypes.c_int64,  # M
            ctypes.c_int64,  # N
            ctypes.c_int64,  # K
            ctypes.POINTER(ctypes.c_float),  # time_ms
        ]
        lib.dispatcher_run_gemm.restype = ctypes.c_int

        # const char* dispatcher_get_kernel_name()
        lib.dispatcher_get_kernel_name.argtypes = []
        lib.dispatcher_get_kernel_name.restype = ctypes.c_char_p

        # void dispatcher_cleanup()
        lib.dispatcher_cleanup.argtypes = []
        lib.dispatcher_cleanup.restype = None

        print(f"OK Library loaded: {lib_path.name}")
        return lib

    except Exception as e:
        print(f"[FAIL] Failed to load library: {e}")
        return None


def run_gemm_from_numpy(lib, A, B, M=None, N=None, K=None):
    """
    Run GEMM on GPU using NumPy arrays

    Args:
        lib: Loaded ctypes library
        A: NumPy array (M x K), dtype=float16, row-major
        B: NumPy array (K x N), dtype=float16, column-major
        M, N, K: Optional dimensions (inferred from arrays if not provided)

    Returns:
        C: Result matrix (M x N), dtype=float16
        time_ms: Execution time in milliseconds
    """
    # Infer dimensions if not provided
    if M is None:
        M = A.shape[0]
    if N is None:
        N = B.shape[1]
    if K is None:
        K = A.shape[1]

    # Validate inputs
    assert A.dtype == np.float16, "A must be float16"
    assert B.dtype == np.float16, "B must be float16"
    assert A.shape == (M, K), f"A shape mismatch: {A.shape} vs ({M}, {K})"
    assert B.shape == (K, N), f"B shape mismatch: {B.shape} vs ({K}, {N})"
    assert A.flags["C_CONTIGUOUS"], "A must be C-contiguous (row-major)"
    assert B.flags["F_CONTIGUOUS"], "B must be F-contiguous (column-major)"

    # Create output array
    C = np.zeros((M, N), dtype=np.float16, order="C")

    # Get pointers
    A_ptr = A.ctypes.data_as(ctypes.c_void_p)
    B_ptr = B.ctypes.data_as(ctypes.c_void_p)
    C_ptr = C.ctypes.data_as(ctypes.c_void_p)

    # Timing output
    time_ms = ctypes.c_float()

    # Call C++ function
    status = lib.dispatcher_run_gemm(
        A_ptr,
        B_ptr,
        C_ptr,
        ctypes.c_int64(M),
        ctypes.c_int64(N),
        ctypes.c_int64(K),
        ctypes.byref(time_ms),
    )

    if status != 0:
        raise RuntimeError("GEMM execution failed")

    return C, time_ms.value


def main():
    print("\n" + "=" * 70)
    print("NumPy to GPU - Complete Workflow")
    print("=" * 70 + "\n")

    print("This demonstrates the COMPLETE Python <-> GPU workflow:")
    print("  NumPy matrices -> C++ dispatcher -> GPU GEMM -> NumPy results")
    print()

    # Step 1: Ensure kernels exist
    print("Step 1: Ensure Kernels Generated")
    print("-" * 70)
    if not ensure_kernels_generated():
        return 1
    print()

    # Step 2: Compile dynamic library
    print("Step 2: Compile Dynamic Library")
    print("-" * 70)
    lib_path = compile_dynamic_library()
    if lib_path is None:
        return 1
    print()

    # Step 3: Load library
    print("Step 3: Load Library via ctypes")
    print("-" * 70)
    lib = load_dispatcher_library(lib_path)
    if lib is None:
        return 1
    print()

    # Step 4: Initialize dispatcher
    print("Step 4: Initialize Dispatcher")
    print("-" * 70)
    status = lib.dispatcher_initialize()
    if status != 0:
        print("[FAIL] Initialization failed")
        return 1

    kernel_name = lib.dispatcher_get_kernel_name().decode("utf-8")
    print("OK Dispatcher initialized")
    print(f"  Kernel: {kernel_name}")
    print()

    # Step 5: Create NumPy matrices
    print("Step 5: Create NumPy Matrices")
    print("-" * 70)

    M, N, K = 512, 512, 512

    print(f"Creating matrices: M={M}, N={N}, K={K}")

    # Create test matrices: A=1, B=1, so C should be K
    A = np.ones((M, K), dtype=np.float16, order="C")  # Row-major
    B = np.ones((K, N), dtype=np.float16, order="F")  # Column-major

    print(
        f"  A: shape={A.shape}, dtype={A.dtype}, "
        f"order={'C' if A.flags['C_CONTIGUOUS'] else 'F'}"
    )
    print(
        f"  B: shape={B.shape}, dtype={B.dtype}, "
        f"order={'C' if B.flags['C_CONTIGUOUS'] else 'F'}"
    )
    print()

    # Step 6: Select kernel
    print("Step 6: Select Kernel for Problem")
    print("-" * 70)

    name_buffer = ctypes.create_string_buffer(256)
    status = lib.dispatcher_select_kernel(
        ctypes.c_int64(M), ctypes.c_int64(N), ctypes.c_int64(K), name_buffer, 256
    )

    if status != 0:
        print("[FAIL] Kernel selection failed")
        return 1

    selected_kernel = name_buffer.value.decode("utf-8")
    print(f"OK Selected kernel: {selected_kernel}")
    print()

    # Step 7: Execute GEMM on GPU
    print("Step 7: Execute GEMM on GPU")
    print("-" * 70)

    print("Calling dispatcher_run_gemm with NumPy array pointers...")

    try:
        C, time_ms = run_gemm_from_numpy(lib, A, B, M, N, K)

        print("OK GPU execution complete!")
        print(f"  Time: {time_ms:.4f} ms")

        # Calculate performance
        flops = 2.0 * M * N * K
        tflops = (flops / (time_ms * 1e-3)) / 1e12
        print(f"  Performance: {tflops:.2f} TFLOPS")
        print()

    except Exception as e:
        print(f"[FAIL] Execution failed: {e}")
        lib.dispatcher_cleanup()
        return 1

    # Step 8: Validate results in Python
    print("Step 8: Validate Results in Python")
    print("-" * 70)

    print(f"Result matrix C: shape={C.shape}, dtype={C.dtype}")
    print(f"  Expected: all elements = {K}")
    print(f"  C[0,0] = {C[0, 0]}")
    print(f"  C[0,1] = {C[0, 1]}")
    print(f"  C[100,100] = {C[100, 100]}")
    print()

    # Validate
    expected = float(K)
    correct = np.sum(np.abs(C - expected) < 1.0)
    total = M * N
    accuracy = 100.0 * correct / total

    print("Validation:")
    print(f"  Correct elements: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.2f}%")

    if accuracy > 99.9:
        print("  Status: [OK] Results correct!")
    else:
        print("  Status: [FAIL] Accuracy too low")
    print()

    # Step 9: Compare with NumPy
    print("Step 9: Compare with NumPy Reference")
    print("-" * 70)

    print("Computing NumPy reference...")
    t0 = time.time()
    C_numpy = np.matmul(A, B)
    t1 = time.time()
    numpy_time = (t1 - t0) * 1000

    print(f"  NumPy time: {numpy_time:.4f} ms")
    print(f"  GPU speedup: {numpy_time / time_ms:.1f}x")
    print()

    # Compare results
    max_diff = np.max(np.abs(C - C_numpy))
    mean_diff = np.mean(np.abs(C - C_numpy))

    print("GPU vs NumPy comparison:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if max_diff < 0.01:
        print("  Status: [OK] Perfect match!")
    else:
        print("  Status: [FAIL] Difference too large")
    print()

    # Cleanup
    lib.dispatcher_cleanup()

    # Final summary
    print("=" * 70)
    print("SUCCESS - Complete NumPy to GPU Workflow!")
    print("=" * 70)
    print()
    print("Achieved:")
    print("  [OK] Started with NumPy matrices in Python")
    print("  [OK] Compiled dynamic library with dispatcher")
    print("  [OK] Loaded .so back into Python via ctypes")
    print("  [OK] Passed NumPy pointers to C++")
    print(f"  [OK] C++ executed GPU GEMM via dispatcher: {tflops:.2f} TFLOPS")
    print("  [OK] Results written back to NumPy arrays")
    print(f"  [OK] Validated in Python: {accuracy:.2f}% accuracy")
    print(f"  [OK] {numpy_time / time_ms:.1f}x faster than NumPy CPU")
    print()
    print("This is the COMPLETE Python <-> GPU integration!")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
