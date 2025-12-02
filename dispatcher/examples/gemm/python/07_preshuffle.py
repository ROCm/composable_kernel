#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 07: PreShuffle Pipeline

Demonstrates PreShuffle kernel configuration for large matrices.

Complexity: ★★★★☆

Usage:
    python3 07_preshuffle.py
    python3 07_preshuffle.py --help
    python3 07_preshuffle.py --dtype bf16
"""

import sys
import argparse
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
    parser = argparse.ArgumentParser(
        description="PreShuffle Pipeline Example - optimized for large matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 07_preshuffle.py                    # Default FP16
  python3 07_preshuffle.py --dtype bf16       # BF16 mode
  python3 07_preshuffle.py --max-size 8192    # Test larger sizes
        """,
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=4096,
        help="Maximum problem size (default: 4096)",
    )
    parser.add_argument(
        "--arch", default="gfx942", help="Target architecture (default: gfx942)"
    )
    args = parser.parse_args()

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
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        tile_m=256,
        tile_n=256,
        tile_k=64,
        wave_m=4,
        wave_n=4,
        pipeline="compv4",
        gfx_arch=args.arch,
    )

    setup = setup_gemm_dispatcher(config, registry_name="preshuffle", verbose=True)
    if not setup.success:
        print(f"  ERROR: {setup.error}")
        return 1

    dispatcher = setup.dispatcher
    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    print("\n  PreShuffle Benefits:")
    print("    - Pre-shuffles data in LDS before computation")
    print("    - Reduces bank conflicts")
    print("    - Best for large matrices (2048+)")

    # =========================================================================
    # Step 2: Run GEMM with large matrices
    # =========================================================================
    print("\nStep 2: Run GEMM (large matrices)")

    all_sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]
    sizes = [(m, n, k) for m, n, k in all_sizes if max(m, n, k) <= args.max_size]

    print(f"\n  {'Size':<20} {'Time (ms)':>12} {'TFLOPS':>10}")
    print("  " + "-" * 45)

    for M, N, K in sizes:
        if not dispatcher.is_supported(M, N, K):
            continue

        A = np.random.randn(M, K).astype(np_dtype) * 0.1
        B = np.random.randn(K, N).astype(np_dtype) * 0.1

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
