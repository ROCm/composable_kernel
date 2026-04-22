#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 03: Benchmark

Performance benchmarking with compute-optimized kernel configuration
using JIT compilation.

Usage:
    python3 03_benchmark.py
    python3 03_benchmark.py --help
    python3 03_benchmark.py --size 4096
    python3 03_benchmark.py --dtype bf16 --iterations 20
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
        description="GEMM Benchmark Example - performance testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 03_benchmark.py                     # Default benchmark suite
  python3 03_benchmark.py --size 4096         # Single size benchmark
  python3 03_benchmark.py --dtype bf16        # BF16 benchmark
  python3 03_benchmark.py --iterations 20     # More iterations
        """,
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: bf16)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=0,
        help="Single problem size MxNxK (default: run all sizes)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Benchmark iterations (default: 10)"
    )
    parser.add_argument(
        "--arch",
        default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Example 03: Benchmark")
    print("=" * 60)

    # =========================================================================
    # Step 1: JIT build dispatcher with compute-optimized config
    # =========================================================================
    print("\nStep 1: JIT Build Dispatcher")

    config = KernelConfig(
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        tile_m=128,
        tile_n=128,
        tile_k=32,
        pipeline="compv4",
        scheduler="intrawave",
        gfx_arch=args.arch,
    )

    reg = Registry(name="benchmark")
    reg.register_kernel(config)

    setups = reg.build(verbose=True)
    if not setups or not setups[0].success:
        error = setups[0].error if setups else "No kernels built"
        print(f"  ERROR: {error}")
        return 1

    dispatcher = setups[0].dispatcher

    # =========================================================================
    # Step 2: Benchmark
    # =========================================================================
    print("\nStep 2: Benchmark")

    if args.size > 0:
        sizes = [(args.size, args.size, args.size)]
    else:
        sizes = [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (1024, 2048, 512),
            (2048, 1024, 2048),
        ]

    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    print(f"  Warmup: {args.warmup}, Iterations: {args.iterations}\n")

    print(f"  {'Size':<20} | {'Min (ms)':>10} | {'Avg (ms)':>10} | {'TFLOPS':>10}")
    print("  " + "-" * 60)

    all_tflops = []

    for M, N, K in sizes:
        if not dispatcher.is_supported(M, N, K):
            continue

        A = np.random.randn(M, K).astype(np_dtype) * 0.1
        B = np.random.randn(K, N).astype(np_dtype) * 0.1

        for _ in range(args.warmup):
            dispatcher.run(A, B, M, N, K)

        times = []
        for _ in range(args.iterations):
            result = dispatcher.run(A, B, M, N, K)
            if result.success:
                times.append(result.time_ms)

        if times:
            min_time = min(times)
            avg_time = sum(times) / len(times)
            tflops = (2.0 * M * N * K / (avg_time * 1e-3)) / 1e12
            all_tflops.append(tflops)
            print(
                f"  {M:>4}x{N:>4}x{K:<4} | {min_time:>10.4f} | {avg_time:>10.4f} | {tflops:>10.2f}"
            )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if all_tflops:
        print(f"  Average: {sum(all_tflops) / len(all_tflops):.2f} TFLOPS")
        print(f"  Peak:    {max(all_tflops):.2f} TFLOPS")

    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
