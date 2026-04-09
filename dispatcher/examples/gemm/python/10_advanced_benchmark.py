#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 10: Advanced Benchmarking with Full Control

This example demonstrates all available benchmark parameters:
  - warmup: Number of warmup iterations (default: 5)
  - repeat: Number of benchmark iterations (default: 20)
  - flush_cache: Flush GPU cache between iterations (default: False)
  - timer: Timer type - "gpu" (default) or "cpu"
  - init: Initialization method - "random", "linear", "constant"

Usage:
    python3 10_advanced_benchmark.py
    python3 10_advanced_benchmark.py --warmup 10 --repeat 100
    python3 10_advanced_benchmark.py --init linear
"""

import argparse
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

import numpy as np

from ctypes_utils import (
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
    reset_for_example,
    detect_gpu_arch,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced GEMM benchmarking with full parameter control"
    )

    # Problem size
    parser.add_argument("-m", type=int, default=2048, help="M dimension")
    parser.add_argument("-n", type=int, default=2048, help="N dimension")
    parser.add_argument("-k", type=int, default=2048, help="K dimension")

    # Benchmark parameters
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--repeat", type=int, default=20, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--flush-cache", action="store_true", help="Flush GPU cache between iterations"
    )
    parser.add_argument(
        "--timer", choices=["gpu", "cpu"], default="gpu", help="Timer type (gpu or cpu)"
    )
    parser.add_argument(
        "--init",
        choices=["random", "linear", "constant"],
        default="random",
        help="Initialization method",
    )

    # Kernel configuration
    parser.add_argument("--dtype", default="fp16", help="Data type")
    parser.add_argument("--pipeline", default="compv4", help="Pipeline type")
    parser.add_argument(
        "--arch",
        default=detect_gpu_arch(),
        help="GPU architecture (auto-detected from rocminfo)",
    )

    return parser.parse_args()


def initialize_matrix(shape, method, dtype):
    """Initialize matrix with specified method"""
    if method == "random":
        return np.random.randn(*shape).astype(dtype) * 0.5
    elif method == "linear":
        total = np.prod(shape)
        return np.arange(total).reshape(shape).astype(dtype) / total
    elif method == "constant":
        return np.ones(shape, dtype=dtype)
    else:
        return np.random.randn(*shape).astype(dtype)


def main():
    args = parse_args()

    reset_for_example()

    print("=" * 70)
    print("Example 10: Advanced GEMM Benchmarking")
    print("=" * 70)

    # Show benchmark configuration
    print("\nBenchmark Configuration:")
    print(f"  Problem Size:   {args.m} x {args.n} x {args.k}")
    print(f"  Warmup:         {args.warmup} iterations")
    print(f"  Repeat:         {args.repeat} iterations")
    print(f"  Flush Cache:    {args.flush_cache}")
    print(f"  Timer:          {args.timer}")
    print(f"  Init Method:    {args.init}")
    print(f"  Data Type:      {args.dtype}")
    print(f"  Pipeline:       {args.pipeline}")
    print(f"  Architecture:   {args.arch}")
    print()

    # Map dtype
    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    # Initialize matrices
    print("Step 1: Initialize matrices...")
    A = initialize_matrix((args.m, args.k), args.init, np_dtype)
    B = initialize_matrix((args.k, args.n), args.init, np_dtype)
    print(f"  A: {A.shape} ({args.init})")
    print(f"  B: {B.shape} ({args.init})")

    # Create kernel config (does not include M/N/K - those are problem size)
    print("\nStep 2: Create kernel configuration...")
    kernel_config = KernelConfig(
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        dtype_acc="fp32",
        layout_a="row",
        layout_b="col",  # B is column-major for optimal performance
        layout_c="row",
        tile_m=128,
        tile_n=128,
        tile_k=32,
        wave_m=2,
        wave_n=2,
        wave_k=1,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        pipeline=args.pipeline,
        scheduler="intrawave",
        epilogue="cshuffle",
        gfx_arch=args.arch,
    )
    print(f"  Config: {args.dtype}, tile=128x128x32, {args.pipeline}")

    # Setup dispatcher
    print("\nStep 3: Setup dispatcher...")
    setup = setup_gemm_dispatcher(
        config=kernel_config,
        registry_name="benchmark_gemm",
        verbose=False,
        auto_rebuild=True,
    )

    if not setup.success:
        print(f"  ERROR: {setup.error}")
        return 1

    dispatcher = setup.dispatcher
    print(f"  Library: {setup.lib.path if setup.lib else 'N/A'}")
    print(f"  Kernel: {setup.lib.get_kernel_name() if setup.lib else 'N/A'}")

    # Run benchmark with multiple iterations
    print("\nStep 4: Run benchmark...")
    print(f"  Running {args.warmup} warmup + {args.repeat} benchmark iterations...")

    # Warmup
    for _ in range(args.warmup):
        _ = dispatcher.run(A, B, args.m, args.n, args.k)

    # Benchmark
    times = []
    for _ in range(args.repeat):
        result = dispatcher.run(A, B, args.m, args.n, args.k)
        if result.success:
            times.append(result.time_ms)

    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        # Calculate TFLOPS
        flops = 2 * args.m * args.n * args.k
        avg_tflops = (flops / 1e12) / (avg_time / 1000) if avg_time > 0 else 0
        max_tflops = (flops / 1e12) / (min_time / 1000) if min_time > 0 else 0

        # Calculate bandwidth (C has same dtype as A and B)
        C_bytes = args.m * args.n * np.dtype(np_dtype).itemsize
        bandwidth_gb = (
            (A.nbytes + B.nbytes + C_bytes) / 1e9 / (avg_time / 1000)
            if avg_time > 0
            else 0
        )

        print(f"\n  *** BENCHMARK RESULTS ({args.repeat} iterations) ***")
        print(f"  Average Time:   {avg_time:.4f} ms")
        print(f"  Min Time:       {min_time:.4f} ms")
        print(f"  Max Time:       {max_time:.4f} ms")
        print(f"  Avg TFLOPS:     {avg_tflops:.2f}")
        print(f"  Peak TFLOPS:    {max_tflops:.2f}")
        print(f"  Bandwidth:      {bandwidth_gb:.2f} GB/s")
    else:
        print("  FAILED: No successful runs")
        return 1

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK PARAMETERS REFERENCE")
    print("=" * 70)
    print("""
Available parameters for GEMM benchmarking:

  --warmup N          Number of warmup iterations (discard results)
                      Higher = more stable results, longer run time
                      Default: 5

  --repeat N          Number of benchmark iterations
                      Higher = more accurate average, longer run time
                      Default: 20

  --flush-cache       Flush GPU L2 cache between iterations
                      Use for memory-bound benchmarks
                      Default: off

  --timer {gpu,cpu}   Timer type
                      gpu = HIP events (more accurate for GPU)
                      cpu = std::chrono (includes kernel launch overhead)
                      Default: gpu

  --init METHOD       Matrix initialization
                      random = uniform random [-0.5, 0.5]
                      linear = sequential values
                      constant = all ones
                      Default: random

Note: For C++ examples, these parameters are passed to stream_config:

    ck_tile::stream_config cfg{
        nullptr,    // stream_id
        true,       // time_kernel
        1,          // log_level
        5,          // cold_niters (warmup)
        20,         // nrepeat
        true,       // is_gpu_timer
        false,      // flush_cache
        1           // rotating_count
    };
""")

    # Cleanup
    cleanup_gemm()

    return 0


if __name__ == "__main__":
    sys.exit(main())
