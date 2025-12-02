#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 07: PreShuffle Pipeline

Demonstrates PreShuffle kernel configuration for large matrices.

Complexity: ★★★★☆

Usage:
    python3 07_preshuffle.py
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


def main():
    reset_for_example()

    print("=" * 60)
    print("Example 07: PreShuffle Pipeline")
    print("=" * 60)

    # =========================================================================
    # Step 1: Setup dispatcher with large tiles
    # =========================================================================
    print("\nStep 1: Setup Dispatcher")

    # PreShuffle works best with larger tiles
    config = KernelConfig(
        dtype_a="fp16",
        tile_m=256,
        tile_n=256,
        tile_k=64,
        wave_m=4,
        wave_n=4,
        pipeline="compv4",
    )

    setup = setup_gemm_dispatcher(config, registry_name="preshuffle", verbose=True)
    if not setup.success:
        print(f"  ERROR: {setup.error}")
        return 1

    dispatcher = setup.dispatcher

    print("\n  PreShuffle Benefits:")
    print("    - Pre-shuffles data in LDS before computation")
    print("    - Reduces bank conflicts")
    print("    - Best for large matrices (2048+)")

    # =========================================================================
    # Step 2: Run GEMM with large matrices
    # =========================================================================
    print("\nStep 2: Run GEMM (large matrices)")

    sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    print(f"\n  {'Size':<20} {'Time (ms)':>12} {'TFLOPS':>10}")
    print("  " + "-" * 45)

    for M, N, K in sizes:
        if not dispatcher.is_supported(M, N, K):
            continue

        A = np.random.randn(M, K).astype(np.float16) * 0.1
        B = np.random.randn(K, N).astype(np.float16) * 0.1

        result = dispatcher.run(A, B, M, N, K)

        if result.success:
            print(f"  {M}x{N}x{K:<10} {result.time_ms:>12.4f} {result.tflops:>10.2f}")

    # Cleanup
    cleanup_gemm()

    # Summary
    print("\n" + "=" * 60)
    print("PreShuffle Pattern:")
    print("=" * 60)
    print("  1. Use larger tiles (256x256x64)")
    print("  2. Generate 'preshuffle' variant")
    print("  3. Best for large matrices (M,N >= 2048)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
