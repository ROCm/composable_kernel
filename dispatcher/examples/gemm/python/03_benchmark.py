#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 03: Benchmark

Performance benchmarking with explicit Registry and Dispatcher.
Shows compute-optimized kernel configuration.

Complexity: ★★★☆☆

Usage:
    python3 03_benchmark.py [M] [N] [K]
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
    print("Example 03: Benchmark")
    print("=" * 60)

    # Parse args
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    # =========================================================================
    # Step 1: Define compute-optimized kernel config
    # =========================================================================
    print("\nStep 1: Define KernelConfig (compute-optimized)")

    config = KernelConfig(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        wave_m=2,
        wave_n=2,
        wave_k=1,
        block_size=256,
        pipeline="compv4",
        scheduler="intrawave",
        pad_m=True,
        pad_n=True,
        pad_k=True,
    )
    print(f"  Tile: {config.tile_str}")
    print(f"  Pipeline: {config.pipeline}/{config.scheduler}")

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

    registry = Registry(name="benchmark", lib=lib)
    registry.register_kernel(config)

    dispatcher = Dispatcher(registry=registry, lib=lib)
    print(f"  {dispatcher}")

    # =========================================================================
    # Step 3: Define benchmark sizes
    # =========================================================================
    print("\nStep 3: Benchmark")

    if M > 0 and N > 0 and K > 0:
        sizes = [(M, N, K)]
    else:
        sizes = [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (1024, 2048, 512),
            (2048, 1024, 2048),
        ]

    warmup = 3
    iterations = 10
    print(f"  Warmup: {warmup}, Iterations: {iterations}\n")

    print(f"  {'Size':<20} | {'Min (ms)':>10} | {'Avg (ms)':>10} | {'TFLOPS':>10}")
    print("  " + "-" * 60)

    all_tflops = []

    for M, N, K in sizes:
        if not dispatcher.is_supported(M, N, K):
            continue

        A = np.random.randn(M, K).astype(np.float16) * 0.1
        B = np.random.randn(K, N).astype(np.float16) * 0.1

        # Warmup
        for _ in range(warmup):
            dispatcher.run(A, B, M, N, K)

        # Benchmark
        times = []
        for _ in range(iterations):
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

    # =========================================================================
    # Summary
    # =========================================================================
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
