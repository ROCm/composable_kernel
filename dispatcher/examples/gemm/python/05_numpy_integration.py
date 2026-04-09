#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 05: NumPy Integration

Shows how to create a GPU-accelerated matmul wrapper.


Usage:
    python3 05_numpy_integration.py
    python3 05_numpy_integration.py --help
    python3 05_numpy_integration.py --dtype bf16
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    Dispatcher,
    setup_gemm_dispatcher,
    cleanup_gemm,
    reset_for_example,
    detect_gpu_arch,
)


class GPUMatmul:
    """GPU-accelerated matrix multiplication wrapper."""

    def __init__(self, dispatcher: Dispatcher):
        self.dispatcher = dispatcher

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute C = A @ B on GPU with CPU fallback."""
        M, K = A.shape
        K2, N = B.shape

        if K != K2:
            raise ValueError(f"Dimension mismatch: {A.shape} @ {B.shape}")

        if not self.dispatcher.is_supported(M, N, K):
            return np.matmul(A, B)

        result = self.dispatcher.run(A, B, M, N, K)
        return result.output if result.success else np.matmul(A, B)


def main():
    parser = argparse.ArgumentParser(
        description="NumPy Integration Example - GPU-accelerated matmul wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 05_numpy_integration.py             # Default FP16
  python3 05_numpy_integration.py --dtype bf16  # BF16 mode
        """,
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--arch",
        default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo)",
    )
    args = parser.parse_args()

    reset_for_example()

    print("=" * 60)
    print("Example 05: NumPy Integration")
    print("=" * 60)

    # =========================================================================
    # Step 1: Setup dispatcher
    # =========================================================================
    print("\nStep 1: Setup Dispatcher")

    config = KernelConfig(
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        tile_m=128,
        tile_n=128,
        tile_k=32,
        gfx_arch=args.arch,
    )

    setup = setup_gemm_dispatcher(config, registry_name="numpy", verbose=True)
    if not setup.success:
        print(f"  ERROR: {setup.error}")
        return 1

    dispatcher = setup.dispatcher
    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    # =========================================================================
    # Step 2: Create GPU matmul wrapper
    # =========================================================================
    print("\nStep 2: Create GPUMatmul")

    gpu_matmul = GPUMatmul(dispatcher=dispatcher)
    print("  gpu_matmul ready")

    # =========================================================================
    # Step 3: Demo - Simple multiplication using gpu_matmul
    # =========================================================================
    print("\nStep 3: Demo - Simple Multiplication")

    A = np.random.randn(1024, 512).astype(np_dtype) * 0.1
    B = np.random.randn(512, 256).astype(np_dtype) * 0.1

    # Use the gpu_matmul wrapper
    C = gpu_matmul(A, B)
    print(f"  gpu_matmul result: {C.shape}, sum={C.sum():.4f}")

    M, K = A.shape
    _, N = B.shape
    result = dispatcher.run(A, B, M, N, K)

    print(f"  A: {A.shape}, B: {B.shape} -> C: {result.output.shape}")
    print(f"  GPU: {result.time_ms:.4f} ms, {result.tflops:.2f} TFLOPS")

    # =========================================================================
    # Step 4: Demo - FFN block
    # =========================================================================
    print("\nStep 4: Demo - FFN Block")

    batch, hidden, ffn = 128, 768, 3072
    X = np.random.randn(batch, hidden).astype(np_dtype) * 0.02
    W1 = np.random.randn(hidden, ffn).astype(np_dtype) * 0.02
    W2 = np.random.randn(ffn, hidden).astype(np_dtype) * 0.02

    result1 = dispatcher.run(X, W1, batch, ffn, hidden)
    H = result1.output
    result2 = dispatcher.run(H, W2, batch, hidden, ffn)

    print(f"  X: {X.shape} -> H: {H.shape} -> Y: {result2.output.shape}")
    print(f"  Total: {result1.time_ms + result2.time_ms:.4f} ms")

    # Cleanup
    cleanup_gemm()

    # Summary
    print("\n" + "=" * 60)
    print("NumPy Integration Pattern:")
    print("=" * 60)
    print("  1. setup_gemm_dispatcher(config)")
    print("  2. GPUMatmul(dispatcher)")
    print("  3. C = gpu_matmul(A, B)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
