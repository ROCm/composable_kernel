#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 05: NumPy Integration

Shows how to create a GPU-accelerated matmul using explicit API.

Complexity: ★★☆☆☆

Usage:
    python3 05_numpy_integration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    DispatcherLib,
    Registry,
    Dispatcher,
)


class GPUMatmul:
    """GPU-accelerated matrix multiplication with explicit dispatcher."""

    def __init__(self, config: KernelConfig, dispatcher: Dispatcher):
        self.config = config
        self.dispatcher = dispatcher

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute C = A @ B on GPU."""
        M, K = A.shape
        K2, N = B.shape

        if K != K2:
            raise ValueError(f"Dimension mismatch: {A.shape} @ {B.shape}")

        if not self.dispatcher.is_supported(M, N, K):
            # Fallback to CPU for unsupported sizes
            return np.matmul(A, B)

        result = self.dispatcher.run(A, B, M, N, K)
        return result.output if result.status == 0 else np.matmul(A, B)


def main():
    print("=" * 60)
    print("Example 05: NumPy Integration")
    print("=" * 60)

    # =========================================================================
    # Step 1: Define kernel config
    # =========================================================================
    print("\nStep 1: Define KernelConfig")

    # Note: The pre-built library uses 128x128x32 tiles without padding.
    # Sizes should be multiples of tile dimensions for best performance.
    config = KernelConfig(
        tile_m=128,
        tile_n=128,
        tile_k=32,
    )
    print(f"  Tile: {config.tile_str}")

    # =========================================================================
    # Step 2: Setup registry and dispatcher
    # =========================================================================
    print("\nStep 2: Setup")

    lib = DispatcherLib.auto()
    if lib is None:
        print("  ERROR: Could not load library")
        print("  Build with: cmake .. -DBUILD_DISPATCHER_EXAMPLES=ON && make")
        return 1

    registry = Registry(name="numpy", lib=lib)
    registry.register_kernel(config)

    dispatcher = Dispatcher(registry=registry, lib=lib)
    print(f"  {dispatcher}")

    # =========================================================================
    # Step 3: Create GPU matmul function
    # =========================================================================
    print("\nStep 3: Create GPUMatmul")

    gpu_matmul = GPUMatmul(config=config, dispatcher=dispatcher)
    print(f"  gpu_matmul ready (tile={config.tile_str})")

    # =========================================================================
    # Step 4: Demo - Simple multiplication
    # =========================================================================
    print("\nStep 4: Demo - Simple Multiplication")

    A = np.random.randn(1024, 512).astype(np.float16) * 0.1
    B = np.random.randn(512, 256).astype(np.float16) * 0.1

    print(f"  A: {A.shape}")
    print(f"  B: {B.shape}")

    # Run with timing to show GPU execution
    M, K = A.shape
    _, N = B.shape
    result = dispatcher.run(A, B, M, N, K)
    C = result.output

    print(f"  C: {C.shape}")
    print(f"  C.sum(): {np.sum(C):.4f}")
    print(f"  *** GPU: {result.time_ms:.4f} ms, {result.tflops:.2f} TFLOPS ***")

    # =========================================================================
    # Step 5: Demo - Neural network layer (FFN block)
    # =========================================================================
    print("\nStep 5: Demo - Neural Network Layer (FFN)")

    # Use batch size that's a multiple of tile_m (128) for the non-padded kernel
    batch, hidden, ffn = 128, 768, 3072

    X = np.random.randn(batch, hidden).astype(np.float16) * 0.02
    W1 = np.random.randn(hidden, ffn).astype(np.float16) * 0.02
    W2 = np.random.randn(ffn, hidden).astype(np.float16) * 0.02

    print(f"  Input:   {X.shape}")
    print(f"  W1:      {W1.shape}")
    print(f"  W2:      {W2.shape}")

    # FFN forward pass with timing
    # X @ W1: (128, 768) @ (768, 3072) -> (128, 3072)
    result1 = dispatcher.run(X, W1, batch, ffn, hidden)  # M=128, N=3072, K=768
    H = result1.output  # Up projection

    # H @ W2: (128, 3072) @ (3072, 768) -> (128, 768)
    result2 = dispatcher.run(H, W2, batch, hidden, ffn)  # M=128, N=768, K=3072
    Y = result2.output  # Down projection

    print(f"  Output:  {Y.shape}")
    print(f"  Y.mean(): {np.mean(Y):.6f}")

    total_time = result1.time_ms + result2.time_ms
    total_tflops = result1.tflops + result2.tflops
    print(f"  *** GPU: {total_time:.4f} ms total ***")
    print(
        f"  *** {result1.tflops:.1f} + {result2.tflops:.1f} = {total_tflops:.1f} TFLOPS ***"
    )

    # =========================================================================
    # Step 6: Demo - Using GPUMatmul class with automatic fallback
    # =========================================================================
    print("\nStep 6: Demo - GPUMatmul with Auto-Fallback")

    # This uses the wrapper class that automatically falls back to CPU
    # for sizes not supported by the GPU kernel
    A_small = np.random.randn(64, 256).astype(np.float16)  # M=64 < tile_m=128
    B_small = np.random.randn(256, 128).astype(np.float16)

    print(f"  A: {A_small.shape} (M=64 < tile_m=128)")
    print(f"  B: {B_small.shape}")

    C_small = gpu_matmul(A_small, B_small)
    print(f"  C: {C_small.shape}")
    print("  (Falls back to CPU for sizes smaller than tile)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("NumPy Integration Pattern:")
    print("=" * 60)
    print("  1. Define KernelConfig")
    print("  2. Create Registry and Dispatcher")
    print("  3. Wrap in GPUMatmul class")
    print("  4. Use like np.matmul: C = gpu_matmul(A, B)")
    print("")
    print("Note: Default kernel uses 128x128 tiles without padding.")
    print("      Sizes must be multiples of tile dims for GPU execution.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
