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
    CodegenRunner,
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
            # Fallback to CPU
            return np.matmul(A, B)

        result = self.dispatcher.run(A, B, M, N, K)
        return result.output if result.success else np.matmul(A, B)


def main():
    print("=" * 60)
    print("Example 05: NumPy Integration")
    print("=" * 60)

    # =========================================================================
    # Step 1: Define kernel config
    # =========================================================================
    print("\nStep 1: Define KernelConfig")

    config = KernelConfig(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        pad_m=True,
        pad_n=True,
        pad_k=True,
    )
    print(f"  Tile: {config.tile_str}")

    # =========================================================================
    # Step 2: Setup registry and dispatcher
    # =========================================================================
    print("\nStep 2: Setup")

    codegen = CodegenRunner()
    codegen.generate_from_config(config)

    lib = DispatcherLib.auto()
    if lib is None:
        print("  ERROR: Could not load library")
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

    C = gpu_matmul(A, B)
    print(f"  C: {C.shape}")
    print(f"  C.sum(): {np.sum(C):.4f}")

    # =========================================================================
    # Step 5: Demo - Neural network layer
    # =========================================================================
    print("\nStep 5: Demo - Neural Network Layer")

    batch, hidden, ffn = 64, 768, 3072

    X = np.random.randn(batch, hidden).astype(np.float16) * 0.02
    W1 = np.random.randn(hidden, ffn).astype(np.float16) * 0.02
    W2 = np.random.randn(ffn, hidden).astype(np.float16) * 0.02

    print(f"  Input:   {X.shape}")
    print(f"  W1:      {W1.shape}")
    print(f"  W2:      {W2.shape}")

    # FFN forward pass
    H = gpu_matmul(X, W1)  # Up projection
    Y = gpu_matmul(H, W2)  # Down projection

    print(f"  Output:  {Y.shape}")
    print(f"  Y.mean(): {np.mean(Y):.6f}")

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
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
