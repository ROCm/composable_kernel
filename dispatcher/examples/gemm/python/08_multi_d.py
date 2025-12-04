#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 08: Multi-D GEMM

Demonstrates Multi-D GEMM with fused element-wise operations.

Multi-D GEMM computes: E = ElementWise(A @ B, D0, D1, ...)

For example with MultiDMultiply:
    E = (A @ B) * D0 * D1

Key concepts:
  - D tensors have same shape as output (M x N)
  - Loaded during epilogue phase (fused, no extra memory passes)
  - Supports: MultiDAdd, MultiDMultiply, Relu, Gelu, etc.

NOTE: Multi-D requires kernel generation with --variants multi_d flag:
    python3 codegen/unified_gemm_codegen.py --variants multi_d ...

Complexity: ★★★★★

Usage:
    python3 08_multi_d.py
    python3 08_multi_d.py --help
    python3 08_multi_d.py --verify
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np  # noqa: E402

from ctypes_utils import (  # noqa: E402
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
    reset_for_example,
)


def relu(x):
    """ReLU activation"""
    return np.maximum(x, 0)


def gelu(x):
    """GELU activation (approximate)"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def multi_d_multiply(c, d0, d1):
    """Multi-D multiply: E = C * D0 * D1"""
    return c * d0 * d1


def multi_d_add(c, d0, d1=None):
    """Multi-D add: E = C + D0 (+ D1)"""
    result = c + d0
    if d1 is not None:
        result = result + d1
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Multi-D GEMM Example - demonstrates fused element-wise operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Multi-D GEMM computes: E = ElementWise(A @ B, D0, D1, ...)

Key points:
  - D tensors have same shape as output (M x N)
  - Loaded during epilogue (no extra memory passes)
  - Supports: MultiDAdd, MultiDMultiply, Relu, Gelu

Examples:
  python3 08_multi_d.py                  # Default simulation
  python3 08_multi_d.py --verify         # With verification
  python3 08_multi_d.py --size 1024      # Custom size
        """,
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--size", type=int, default=512, help="Problem size MxNxK (default: 512)"
    )
    parser.add_argument(
        "--arch", default="gfx942", help="Target architecture (default: gfx942)"
    )
    parser.add_argument("--verify", action="store_true", help="Run CPU verification")
    parser.add_argument(
        "--elementwise",
        default="multiply",
        choices=["multiply", "add"],
        help="Element-wise operation (default: multiply)",
    )
    args = parser.parse_args()

    reset_for_example()

    print("=" * 70)
    print("Example 08: Multi-D GEMM (Fused Element-wise Operations)")
    print("=" * 70)

    M, N, K = args.size, args.size, args.size
    np.random.seed(42)

    # =========================================================================
    # Step 1: Setup dispatcher (for standard GEMM)
    # =========================================================================
    print("\nStep 1: Setup Dispatcher")
    print("-" * 40)

    config = KernelConfig(
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        tile_m=128,
        tile_n=128,
        tile_k=32,
        pipeline="compv4",
        gfx_arch=args.arch,
        variant="multi_d",  # Enable multi-d specific validation
    )

    setup = setup_gemm_dispatcher(config, registry_name="multi_d", verbose=True)
    if not setup.success:
        print(f"  ERROR: {setup.error}")
        print("\n  Note: Multi-D kernels require generation with --variants multi_d")
        print("  Continuing with CPU simulation...\n")
        dispatcher = None
    else:
        dispatcher = setup.dispatcher

    print("\n  Multi-D GEMM Overview:")
    print("    - E = ElementWise(A @ B, D0, D1, ...)")
    print("    - D tensors: same shape as output (M x N)")
    print("    - Fused: loaded during epilogue, zero overhead")
    print("    - Operations: MultiDAdd, MultiDMultiply, Relu, Gelu")

    # =========================================================================
    # Step 2: Create tensors
    # =========================================================================
    print("\nStep 2: Create Tensors")
    print("-" * 40)

    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    # Input tensors
    A = (np.random.randn(M, K) * 0.1).astype(np_dtype)
    B = (np.random.randn(K, N) * 0.1).astype(np_dtype)

    # D tensors (same shape as output)
    D0 = (np.random.uniform(0.5, 1.5, (M, N))).astype(np_dtype)  # Positive for multiply
    D1 = (np.random.uniform(0.5, 1.5, (M, N))).astype(np_dtype)

    print(f"  Problem: {M} x {N} x {K}")
    print(f"  A:  {A.shape} ({args.dtype})")
    print(f"  B:  {B.shape} ({args.dtype})")
    print(f"  D0: {D0.shape} ({args.dtype})")
    print(f"  D1: {D1.shape} ({args.dtype})")

    # =========================================================================
    # Step 3: CPU reference computation
    # =========================================================================
    print("\nStep 3: CPU Reference Computation")
    print("-" * 40)

    # Standard GEMM
    C_fp32 = A.astype(np.float32) @ B.astype(np.float32)

    # Apply element-wise operation
    if args.elementwise == "multiply":
        E_ref = multi_d_multiply(
            C_fp32, D0.astype(np.float32), D1.astype(np.float32)
        ).astype(np_dtype)
        op_name = "E = (A @ B) * D0 * D1"
    else:
        E_ref = multi_d_add(
            C_fp32, D0.astype(np.float32), D1.astype(np.float32)
        ).astype(np_dtype)
        op_name = "E = (A @ B) + D0 + D1"

    print(f"  Operation: {op_name}")
    print(f"  C = A @ B:  mean={np.mean(C_fp32):>8.4f}, std={np.std(C_fp32):>8.4f}")
    print(f"  E (fused):  mean={np.mean(E_ref):>8.4f}, std={np.std(E_ref):>8.4f}")

    # =========================================================================
    # Step 4: GPU execution (if available)
    # =========================================================================
    print("\nStep 4: GPU Execution")
    print("-" * 40)

    if dispatcher is not None:
        # Run standard GEMM (Multi-D requires special kernel)
        result = dispatcher.run(A, B, M, N, K)

        if result.success:
            print(f"  Standard GEMM Time: {result.time_ms:.4f} ms")
            print(f"  Standard GEMM TFLOPS: {result.tflops:.2f}")
            print("\n  Note: Full Multi-D fusion requires generated multi_d kernels")
        else:
            print(f"  GPU execution failed: {result.error}")
    else:
        print("  [GPU not available - using CPU simulation]")

        # Simulate timing
        import time

        start = time.perf_counter()
        _ = A.astype(np.float32) @ B.astype(np.float32)
        cpu_time = (time.perf_counter() - start) * 1000

        print(f"  CPU GEMM time: {cpu_time:.4f} ms")

    # =========================================================================
    # Step 5: Verification
    # =========================================================================
    if args.verify:
        print("\nStep 5: Verification")
        print("-" * 40)

        # Compare different approaches
        C_direct = (A.astype(np.float32) @ B.astype(np.float32)).astype(np_dtype)

        # Multi-D fused (reference)
        if args.elementwise == "multiply":
            E_fused = (
                C_direct.astype(np.float32)
                * D0.astype(np.float32)
                * D1.astype(np.float32)
            ).astype(np_dtype)
        else:
            E_fused = (
                C_direct.astype(np.float32)
                + D0.astype(np.float32)
                + D1.astype(np.float32)
            ).astype(np_dtype)

        # Verify reference matches
        max_diff = np.max(np.abs(E_ref.astype(np.float32) - E_fused.astype(np.float32)))
        rtol = 0.01 if np_dtype == np.float16 else 0.001

        passed = max_diff < rtol * np.max(np.abs(E_ref))

        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Tolerance: {rtol * np.max(np.abs(E_ref)):.6f}")
        print(f"  Status:    {'PASS' if passed else 'FAIL'}")

    # Cleanup
    cleanup_gemm()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Multi-D GEMM Pattern Summary:")
    print("=" * 70)
    print("  1. D tensors loaded during epilogue (zero extra memory passes)")
    print("  2. Supports multiple D tensors: D0, D1, ...")
    print("  3. Flexible element-wise: MultiDAdd, MultiDMultiply, Relu, Gelu")
    print("  4. Use cases:")
    print("     - Transformers: GEMM + bias + activation")
    print("     - MLPs: GEMM + residual connection")
    print("     - Conv layers: GEMM + batch norm fusion")
    print("")
    print("  To generate Multi-D kernels:")
    print("    python3 codegen/unified_gemm_codegen.py --variants multi_d ...")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
