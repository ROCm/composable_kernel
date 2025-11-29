#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 02: Batch GEMM

Runs multiple GEMM operations with different sizes using explicit
Registry and Dispatcher API.

Complexity: ★★☆☆☆

Usage:
    python3 02_batch_gemm.py
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


def main():
    print("=" * 60)
    print("Example 02: Batch GEMM")
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
        pad_k=True,  # Enable padding for all sizes
    )
    print(f"  Tile: {config.tile_str}")
    print("  Padding: enabled (supports any size)")

    # =========================================================================
    # Step 2: Generate and load
    # =========================================================================
    print("\nStep 2: Setup")

    codegen = CodegenRunner()
    codegen.generate_from_config(config)

    lib = DispatcherLib.auto()
    if lib is None:
        print("  ERROR: Could not load library")
        return 1

    # =========================================================================
    # Step 3: Create registry and dispatcher
    # =========================================================================
    print("\nStep 3: Create Registry and Dispatcher")

    registry = Registry(name="batch_gemm", lib=lib)
    registry.register_kernel(config)
    print(f"  {registry}")

    dispatcher = Dispatcher(registry=registry, lib=lib)
    print(f"  {dispatcher}")

    # =========================================================================
    # Step 4: Run batch of different sizes
    # =========================================================================
    print("\nStep 4: Run Batch")

    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    print(f"\n  {'Size':<20} | {'Time (ms)':>12} | {'TFLOPS':>10} | {'Status':>8}")
    print("  " + "-" * 60)

    total_ops = 0
    total_time = 0

    for M, N, K in sizes:
        # Check support
        if not dispatcher.is_supported(M, N, K):
            print(f"  {M:>4}x{N:>4}x{K:<4} | {'N/A':>12} | {'N/A':>10} | Skipped")
            continue

        # Create inputs
        A = np.random.randn(M, K).astype(np.float16) * 0.1
        B = np.random.randn(K, N).astype(np.float16) * 0.1

        # Run via dispatcher
        result = dispatcher.run(A, B, M, N, K)

        if result.success:
            total_ops += 2 * M * N * K
            total_time += result.time_ms
            print(
                f"  {M:>4}x{N:>4}x{K:<4} | {result.time_ms:>12.4f} | "
                f"{result.tflops:>10.2f} | OK"
            )
        else:
            print(f"  {M:>4}x{N:>4}x{K:<4} | {'N/A':>12} | {'N/A':>10} | Error")

    print("  " + "-" * 60)

    if total_time > 0:
        avg_tflops = (total_ops / 1e12) / (total_time / 1000)
        print(f"\n  Total: {total_time:.2f} ms, Average: {avg_tflops:.2f} TFLOPS")

    print("\n" + "=" * 60)
    print("Batch GEMM complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
