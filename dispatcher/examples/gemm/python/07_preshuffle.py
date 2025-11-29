#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 07: PreShuffle Pipeline

Demonstrates PreShuffle kernel configuration using explicit API.

Complexity: ★★★★☆

Usage:
    python3 07_preshuffle.py
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
    print("Example 07: PreShuffle Pipeline")
    print("=" * 60)

    # =========================================================================
    # Step 1: Define PreShuffle kernel config
    # =========================================================================
    print("\nStep 1: Define PreShuffle KernelConfig")

    # PreShuffle works best with larger tiles
    preshuffle_config = KernelConfig(
        tile_m=256,
        tile_n=256,
        tile_k=64,
        wave_m=4,
        wave_n=4,
        wave_k=1,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        block_size=256,
        pipeline="compv4",
        scheduler="intrawave",
        pad_m=True,
        pad_n=True,
        pad_k=True,
    )

    print("  PreShuffle Configuration:")
    print(f"    Tile: {preshuffle_config.tile_str}")
    print(
        f"    Waves: {preshuffle_config.wave_m}x{preshuffle_config.wave_n}x{preshuffle_config.wave_k}"
    )
    print(f"    Pipeline: {preshuffle_config.pipeline}")
    print("\n  PreShuffle Benefits:")
    print("    - Pre-shuffles data in LDS before computation")
    print("    - Reduces bank conflicts")
    print("    - Best for large matrices (2048+)")

    # =========================================================================
    # Step 2: Setup registry and dispatcher
    # =========================================================================
    print("\nStep 2: Setup")

    codegen = CodegenRunner()

    # Generate preshuffle variant
    result = codegen.generate("preshuffle")
    print(f"  Generated preshuffle kernels: {result.kernel_count}")

    lib = DispatcherLib.auto()
    if lib is None:
        print("  ERROR: Could not load library")
        return 1

    registry = Registry(name="preshuffle", lib=lib)
    registry.register_kernel(preshuffle_config)

    dispatcher = Dispatcher(registry=registry, lib=lib)
    print(f"  {dispatcher}")

    # =========================================================================
    # Step 3: Run GEMM with large matrices
    # =========================================================================
    print("\nStep 3: Run GEMM (large matrices)")

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

    # =========================================================================
    # Summary
    # =========================================================================
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
