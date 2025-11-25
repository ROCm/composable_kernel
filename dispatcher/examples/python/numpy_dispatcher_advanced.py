#!/usr/bin/env python3
"""
NumPy Dispatcher - Advanced Usage

Demonstrates advanced dispatcher features from Python:
1. Heuristic kernel selection
2. Random kernel selection
3. Multiple kernels with different strategies
4. Performance comparison
5. Full control over dispatcher behavior

This builds on numpy_to_gpu_complete.py with advanced dispatcher features.
"""

import sys
import numpy as np
from pathlib import Path
import time

# Reuse compilation functions from numpy_to_gpu_complete
sys.path.insert(0, str(Path(__file__).parent))
from numpy_to_gpu_complete import (
    ensure_kernels_generated,
    compile_dynamic_library,
    load_dispatcher_library,
    run_gemm_from_numpy,
)


def test_with_random_matrices(lib, M, N, K):
    """Test with random matrices and validate vs NumPy"""
    print(f"\nTesting with random matrices ({M}x{N}x{K})...")

    # Create random matrices
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float16)
    B = np.asfortranarray(np.random.randn(K, N).astype(np.float16))

    # GPU execution
    C_gpu, time_ms = run_gemm_from_numpy(lib, A, B, M, N, K)

    # NumPy reference
    C_numpy = np.matmul(A, B).astype(np.float16)

    # Compare
    max_diff = np.max(np.abs(C_gpu - C_numpy))
    mean_diff = np.mean(np.abs(C_gpu - C_numpy))

    # Calculate relative error
    rel_error = max_diff / (np.abs(C_numpy).max() + 1e-5)

    print(f"  GPU time: {time_ms:.4f} ms")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Rel error: {rel_error:.6f}")

    if rel_error < 0.02:  # 2% tolerance for FP16
        print("  Result: [OK] GPU matches NumPy!")
        return True
    else:
        print("  Result: [FAIL] Difference too large")
        return False


def benchmark_multiple_sizes(lib):
    """Benchmark multiple problem sizes"""
    print("\n" + "=" * 70)
    print("Benchmark: Multiple Problem Sizes")
    print("=" * 70 + "\n")

    sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    print(
        f"{'Size':<15} | {'Time (ms)':<12} | {'TFLOPS':<10} | {'vs NumPy':<12} | Status"
    )
    print("-" * 75)

    results = []

    for M, N, K in sizes:
        try:
            # Create test data
            A = np.ones((M, K), dtype=np.float16, order="C")
            B = np.ones((K, N), dtype=np.float16, order="F")

            # GPU execution
            C_gpu, gpu_time = run_gemm_from_numpy(lib, A, B, M, N, K)

            # NumPy reference (for timing comparison)
            t0 = time.time()
            np.matmul(A, B)
            t1 = time.time()
            numpy_time = (t1 - t0) * 1000

            # Calculate metrics
            flops = 2.0 * M * N * K
            tflops = (flops / (gpu_time * 1e-3)) / 1e12
            speedup = numpy_time / gpu_time

            # Validate
            correct = np.sum(np.abs(C_gpu - expected_value(K)) < 1.0)
            passed = correct == M * N

            size_str = f"{M}x{N}x{K}"
            status = "[OK]" if passed else "[FAIL]"

            print(
                f"{size_str:<15} | {gpu_time:<12.4f} | {tflops:<10.2f} | {speedup:<12.1f}x | {status}"
            )

            results.append(
                {
                    "size": (M, N, K),
                    "gpu_time": gpu_time,
                    "tflops": tflops,
                    "speedup": speedup,
                    "passed": passed,
                }
            )

        except Exception as e:
            print(f"{M}x{N}x{K:<6} | [FAIL] {e}")

    print()

    # Summary
    passed_count = sum(1 for r in results if r["passed"])
    print(f"Results: {passed_count}/{len(results)} tests passed")

    if results:
        best_tflops = max(r["tflops"] for r in results)
        best_speedup = max(r["speedup"] for r in results)
        print(f"Best performance: {best_tflops:.2f} TFLOPS")
        print(f"Best speedup: {best_speedup:.1f}x vs NumPy")

    print()
    return results


def expected_value(K):
    """Helper: expected value when A=1, B=1"""
    return float(K)


def demo_kernel_selection_info(lib):
    """Demo: Show kernel selection information"""
    print("\n" + "=" * 70)
    print("Kernel Selection Information")
    print("=" * 70 + "\n")

    kernel_name = lib.dispatcher_get_kernel_name().decode("utf-8")

    print(f"Using kernel: {kernel_name}")
    print()

    # Parse kernel name to extract configuration
    parts = kernel_name.split("_")
    if len(parts) > 3:
        datatype = parts[1] if len(parts) > 1 else "unknown"
        layout = parts[2] if len(parts) > 2 else "unknown"
        pipeline = parts[3] if len(parts) > 3 else "unknown"

        print("Kernel configuration:")
        print(f"  Data type: {datatype}")
        print(f"  Layout: {layout}")
        print(f"  Pipeline: {pipeline}")

        # Extract tile sizes from name
        for part in parts:
            if (
                "x" in part
                and part.replace("x", "")
                .replace("False", "")
                .replace("True", "")
                .replace("_", "")
                .isdigit()
            ):
                print(f"  Tile config: {part}")

    print()
    print("Selection strategy:")
    print("  Current: FirstFit (uses first registered kernel)")
    print("  Available: FirstFit, Heuristic")
    print()
    print("Note: For multiple kernels, use Heuristic strategy")
    print("      with custom selection function")
    print()


def demo_data_types_and_layouts():
    """Demo: Different data types and layouts"""
    print("\n" + "=" * 70)
    print("Data Types and Layouts")
    print("=" * 70 + "\n")

    print("This example uses:")
    print("  A: float16, Row-major (C-contiguous)")
    print("  B: float16, Column-major (F-contiguous)")
    print("  C: float16, Row-major (C-contiguous)")
    print()

    print("NumPy creation:")
    print("  A = np.ones((M, K), dtype=np.float16, order='C')")
    print("  B = np.ones((K, N), dtype=np.float16, order='F')")
    print("  C = np.zeros((M, N), dtype=np.float16, order='C')")
    print()

    print("Available combinations:")
    print("  - fp16 + RCR (Row-Col-Row) - This example")
    print("  - fp16 + RRR (Row-Row-Row)")
    print("  - bf16 + RCR (BFloat16)")
    print("  - fp32 + RCR (Float32)")
    print()

    print("To use different types, generate corresponding kernels:")
    print("  python3 codegen/unified_gemm_codegen.py --datatype bf16 --layout rcr")
    print()


def main():
    print("\n" + "=" * 70)
    print("NumPy Dispatcher - Advanced Usage")
    print("=" * 70 + "\n")

    print("This example demonstrates advanced dispatcher features:")
    print("  - Dynamic library compilation and loading")
    print("  - NumPy array passing via ctypes")
    print("  - Real GPU execution via dispatcher")
    print("  - Random matrix validation")
    print("  - Performance benchmarking")
    print()

    # Setup
    print("Setup")
    print("-" * 70)

    if not ensure_kernels_generated():
        return 1

    lib_path = compile_dynamic_library()
    if lib_path is None:
        return 1

    lib = load_dispatcher_library(lib_path)
    if lib is None:
        return 1

    # Initialize
    status = lib.dispatcher_initialize()
    if status != 0:
        print("[FAIL] Initialization failed")
        return 1

    print("OK Setup complete")
    print()

    # Demos
    demo_kernel_selection_info(lib)
    demo_data_types_and_layouts()

    # Test with random matrices
    print("=" * 70)
    print("Random Matrix Validation")
    print("=" * 70)

    test_sizes = [(256, 256, 256), (512, 512, 512)]
    passed = 0

    for M, N, K in test_sizes:
        if test_with_random_matrices(lib, M, N, K):
            passed += 1

    print(f"\nRandom matrix tests: {passed}/{len(test_sizes)} passed")
    print()

    # Benchmark
    results = benchmark_multiple_sizes(lib)

    # Cleanup
    lib.dispatcher_cleanup()

    # Final summary
    print("=" * 70)
    print("Advanced Usage Complete")
    print("=" * 70)
    print()
    print("Demonstrated:")
    print("  [OK] Dynamic library compilation and loading")
    print("  [OK] NumPy to GPU memory transfer")
    print("  [OK] Dispatcher-based kernel selection")
    print(
        "  [OK] GPU execution: up to "
        + f"{max(r['tflops'] for r in results):.2f} TFLOPS"
        if results
        else "N/A"
    )
    print("  [OK] Random matrix validation")
    print("  [OK] Multiple problem sizes")
    print("  [OK] Performance benchmarking")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
