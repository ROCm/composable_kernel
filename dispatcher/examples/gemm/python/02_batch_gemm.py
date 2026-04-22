#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 02: Batch GEMM

Runs multiple GEMM operations with different sizes using JIT compilation.

Usage:
    python3 02_batch_gemm.py
    python3 02_batch_gemm.py --help
    python3 02_batch_gemm.py --dtype bf16
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    Registry,
    detect_gpu_arch,
)


def main():
    parser = argparse.ArgumentParser(
        description="Batch GEMM Example - runs multiple sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 02_batch_gemm.py                    # Default FP16
  python3 02_batch_gemm.py --dtype bf16       # BF16 GEMM
  python3 02_batch_gemm.py --max-size 2048    # Limit max size
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
        "--arch",
        default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Example 02: Batch GEMM")
    print("=" * 60)

    # =========================================================================
    # Step 1: JIT build dispatcher
    # =========================================================================
    print("\nStep 1: JIT Build Dispatcher")

    config = KernelConfig(
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        tile_m=128,
        tile_n=128,
        tile_k=32,
        gfx_arch=args.arch,
    )

    reg = Registry(name="batch_gemm")
    reg.register_kernel(config)

    setups = reg.build(verbose=True)
    if not setups or not setups[0].success:
        error = setups[0].error if setups else "No kernels built"
        print(f"  ERROR: {error}")
        return 1

    dispatcher = setups[0].dispatcher

    # =========================================================================
    # Step 2: Run batch of different sizes
    # =========================================================================
    print("\nStep 2: Run Batch")

    all_sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    sizes = [(m, n, k) for m, n, k in all_sizes if max(m, n, k) <= args.max_size]

    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    print(f"\n  {'Size':<20} | {'Time (ms)':>12} | {'TFLOPS':>10} | {'Status':>8}")
    print("  " + "-" * 60)

    total_ops = 0
    total_time = 0

    for M, N, K in sizes:
        if not dispatcher.is_supported(M, N, K):
            print(f"  {M:>4}x{N:>4}x{K:<4} | {'N/A':>12} | {'N/A':>10} | Skipped")
            continue

        A = np.random.randn(M, K).astype(np_dtype) * 0.1
        B = np.random.randn(K, N).astype(np_dtype) * 0.1

        result = dispatcher.run(A, B, M, N, K)

        if result.success:
            total_ops += 2 * M * N * K
            total_time += result.time_ms
            print(
                f"  {M:>4}x{N:>4}x{K:<4} | {result.time_ms:>12.4f} | {result.tflops:>10.2f} | OK"
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
