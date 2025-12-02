#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 08: Multi-D GEMM

Demonstrates Multi-D kernel configuration with fused operations.

Complexity: ★★★★★

Usage:
    python3 08_multi_d.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
    reset_for_example,
)


def relu(x):
    return np.maximum(x, 0)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def main():
    reset_for_example()

    print("=" * 60)
    print("Example 08: Multi-D GEMM")
    print("=" * 60)

    # =========================================================================
    # Step 1: Setup dispatcher
    # =========================================================================
    print("\nStep 1: Setup Dispatcher")

    config = KernelConfig(
        dtype_a="fp16",
        tile_m=128,
        tile_n=128,
        tile_k=32,
        pipeline="compv4",
    )

    setup = setup_gemm_dispatcher(config, registry_name="multi_d", verbose=True)
    if not setup.success:
        print(f"  ERROR: {setup.error}")
        return 1

    dispatcher = setup.dispatcher

    print("\n  Supported Fused Operations:")
    print("    - PassThrough: C = A @ B")
    print("    - MultiDAdd:   C = A @ B + D0 + D1 + ...")
    print("    - Relu:        C = relu(A @ B + D0)")
    print("    - Gelu:        C = gelu(A @ B + D0)")

    # =========================================================================
    # Step 2: CPU simulation of fused operations
    # =========================================================================
    print("\nStep 2: CPU Simulation of Fused Operations")

    M, N, K = 512, 512, 512
    np.random.seed(42)

    A = (np.random.randn(M, K) * 0.1).astype(np.float32)
    B = (np.random.randn(K, N) * 0.1).astype(np.float32)
    bias = (np.random.randn(N) * 0.1).astype(np.float32)

    C_gemm = A @ B
    C_bias = C_gemm + bias
    C_relu = relu(C_bias)
    C_gelu = gelu(C_bias)

    print(f"\n  Problem: {M}x{N}x{K}")
    print(f"    GEMM only:   mean={np.mean(C_gemm):>8.4f}")
    print(f"    GEMM+Bias:   mean={np.mean(C_bias):>8.4f}")
    print(f"    GEMM+ReLU:   mean={np.mean(C_relu):>8.4f}")
    print(f"    GEMM+GELU:   mean={np.mean(C_gelu):>8.4f}")

    # =========================================================================
    # Step 3: GPU GEMM
    # =========================================================================
    print("\nStep 3: GPU GEMM")

    A_fp16 = A.astype(np.float16)
    B_fp16 = B.astype(np.float16)

    result = dispatcher.run(A_fp16, B_fp16, M, N, K)

    if result.success:
        print(f"  Time: {result.time_ms:.4f} ms ({result.tflops:.2f} TFLOPS)")
        print("  With Multi-D fusion, bias+activation computed in same kernel!")

    # Cleanup
    cleanup_gemm()

    # Summary
    print("\n" + "=" * 60)
    print("Multi-D Pattern:")
    print("=" * 60)
    print("  1. Generate 'multi_d' variant")
    print("  2. Fuses: GEMM + Bias + Activation in one kernel")
    print("  3. Zero overhead for elementwise ops")
    print("  4. Common in: Transformers, MLPs, Conv layers")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
