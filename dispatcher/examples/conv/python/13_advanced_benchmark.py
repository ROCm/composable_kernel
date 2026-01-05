#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 13: Advanced Conv Benchmarking with Full Control

This example demonstrates all available benchmark parameters:
  - warmup: Number of warmup iterations (default: 5)
  - repeat: Number of benchmark iterations (default: 20)
  - flush_cache: Flush GPU cache between iterations (default: False)
  - rotating_count: Number of rotating buffers for cache simulation (default: 1)
  - timer: Timer type - "gpu" (default) or "cpu"
  - init: Initialization method - "random", "linear", "constant"

Usage:
    python3 13_advanced_benchmark.py
    python3 13_advanced_benchmark.py --warmup 10 --repeat 100 --flush-cache
    python3 13_advanced_benchmark.py --timer cpu --init linear
"""

import argparse
import sys
from pathlib import Path

# Add path for imports - conv_utils.py is in dispatcher/python/
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

import numpy as np  # noqa: E402
from conv_utils import (  # noqa: E402
    ConvProblem,
    GpuConvRunner,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced Conv benchmarking with full parameter control"
    )

    # Problem size
    parser.add_argument("-n", type=int, default=1, help="Batch size")
    parser.add_argument("-c", type=int, default=64, help="Input channels")
    parser.add_argument("-k", type=int, default=128, help="Output channels")
    parser.add_argument("-hi", type=int, default=28, help="Input height")
    parser.add_argument("-wi", type=int, default=28, help="Input width")
    parser.add_argument("-y", type=int, default=3, help="Filter height")
    parser.add_argument("-x", type=int, default=3, help="Filter width")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--pad", type=int, default=1, help="Padding")

    # Direction
    parser.add_argument(
        "--direction",
        choices=["forward", "bwd_data", "bwd_weight"],
        default="forward",
        help="Convolution direction",
    )

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
        "--rotating-count",
        type=int,
        default=1,
        help="Number of rotating buffers for cache simulation",
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

    return parser.parse_args()


def initialize_tensor(shape, method, dtype):
    """Initialize tensor with specified method"""
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

    print("=" * 70)
    print("Example 13: Advanced Conv Benchmarking")
    print("=" * 70)

    # Calculate output size
    Ho = (args.hi + 2 * args.pad - args.y) // args.stride + 1
    Wo = (args.wi + 2 * args.pad - args.x) // args.stride + 1

    # Show benchmark configuration
    print("\nBenchmark Configuration:")
    print(f"  Direction:      {args.direction}")
    print(f"  Problem:        N={args.n}, C={args.c}, K={args.k}")
    print(f"  Input Size:     {args.hi}x{args.wi}")
    print(f"  Filter Size:    {args.y}x{args.x}")
    print(f"  Output Size:    {Ho}x{Wo}")
    print(f"  Stride/Pad:     {args.stride}/{args.pad}")
    print(f"  Warmup:         {args.warmup} iterations")
    print(f"  Repeat:         {args.repeat} iterations")
    print(f"  Flush Cache:    {args.flush_cache}")
    print(f"  Rotating Count: {args.rotating_count}")
    print(f"  Timer:          {args.timer}")
    print(f"  Init Method:    {args.init}")
    print(f"  Data Type:      {args.dtype}")
    print()

    # Map dtype
    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    # Initialize tensors (NHWGC layout)
    print("Step 1: Initialize tensors...")
    G = 1  # Groups
    input_data = initialize_tensor(
        (args.n, args.hi, args.wi, G, args.c), args.init, np_dtype
    )
    weight_data = initialize_tensor(
        (G, args.k, args.y, args.x, args.c), args.init, np_dtype
    )
    output_data = np.zeros((args.n, Ho, Wo, G, args.k), dtype=np_dtype)

    print(f"  Input:  {input_data.shape} ({args.init})")
    print(f"  Weight: {weight_data.shape} ({args.init})")
    print(f"  Output: {output_data.shape}")

    # Create problem
    print("\nStep 2: Create problem...")
    problem = ConvProblem(
        N=args.n,
        C=args.c,
        K=args.k,
        G=G,
        Hi=args.hi,
        Wi=args.wi,
        Y=args.y,
        X=args.x,
        stride_h=args.stride,
        stride_w=args.stride,
        pad_h=args.pad,
        pad_w=args.pad,
        direction=args.direction,
    )
    print(f"  Problem: {args.direction} {args.dtype}")

    # Create runner with benchmark settings
    print("\nStep 3: Create GPU runner with benchmark settings...")
    runner = GpuConvRunner(
        warmup=args.warmup,
        repeat=args.repeat,
        flush_cache=args.flush_cache,
        rotating_count=args.rotating_count,
        timer=args.timer,
    )

    if not runner.is_available():
        print("  ERROR: GPU not available")
        return 1

    print(f"  Library: {runner.library_path}")
    print(f"  Warmup: {args.warmup}, Repeat: {args.repeat}")
    print(f"  Flush Cache: {args.flush_cache}, Timer: {args.timer}")

    # Run benchmark
    print("\nStep 4: Run benchmark...")
    result = runner.run(input_data, weight_data, problem, output_data)

    if result.get("success"):
        time_ms = result.get("time_ms", 0)
        tflops = result.get("tflops", 0)

        # Calculate statistics
        flops = 2 * args.n * args.k * args.c * Ho * Wo * args.y * args.x
        bandwidth_gb = (
            (input_data.nbytes + weight_data.nbytes + output_data.nbytes)
            / 1e9
            / (time_ms / 1000)
            if time_ms > 0
            else 0
        )

        print("\n  *** BENCHMARK RESULTS ***")
        print(f"  Average Time:   {time_ms:.4f} ms")
        print(f"  TFLOPS:         {tflops:.2f}")
        print(f"  Bandwidth:      {bandwidth_gb:.2f} GB/s")
        print(f"  FLOPs:          {flops:.2e}")
    else:
        print(f"  FAILED: {result.get('error', 'Unknown error')}")
        return 1

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK PARAMETERS REFERENCE")
    print("=" * 70)
    print("""
Available parameters for convolution benchmarking:

  --warmup N          Number of warmup iterations (discard results)
                      Higher = more stable results, longer run time
                      Default: 5

  --repeat N          Number of benchmark iterations
                      Higher = more accurate average, longer run time
                      Default: 20

  --flush-cache       Flush GPU L2 cache between iterations
                      Use for memory-bound benchmarks
                      Default: off

  --rotating-count N  Number of rotating buffers (requires --flush-cache)
                      Simulates real workload cache behavior
                      Default: 1

  --timer {gpu,cpu}   Timer type
                      gpu = HIP events (more accurate for GPU)
                      cpu = std::chrono (includes kernel launch overhead)
                      Default: gpu

  --init METHOD       Tensor initialization
                      random = uniform random [-0.5, 0.5]
                      linear = sequential values
                      constant = all ones
                      Default: random

  --direction DIR     Convolution direction
                      forward = Input x Weight -> Output
                      bwd_data = dOutput x Weight -> dInput
                      bwd_weight = Input x dOutput -> dWeight
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
