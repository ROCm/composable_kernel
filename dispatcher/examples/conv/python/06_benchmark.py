#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 06: Convolution Benchmarking

Demonstrates benchmarking convolution kernels across multiple problem sizes
with validation and cleanup.

Usage:
    python3 06_benchmark.py
    python3 06_benchmark.py --cpu  # Include slow CPU reference
    python3 06_benchmark.py --dtype bf16
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from conv_utils import (
    ConvSignature,
    ConvAlgorithm,
    ArchInfo,
    ConvKernelSet,
    ConvProblem,
    GpuConvRunner,
    validate_conv_config,
    reset_for_conv_example,
    cleanup_conv,
    print_conv_kernel_config,
)


def main():
    parser = argparse.ArgumentParser(description="Convolution Benchmarking")
    parser.add_argument(
        "--cpu", action="store_true", help="Include CPU reference (slow)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="compv4",
        choices=["compv3", "compv4", "mem"],
        help="Pipeline version (default: compv4)",
    )
    parser.add_argument(
        "--arch", type=str, default="gfx942", help="Target architecture"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Example 06: Convolution Benchmarking")
    print("=" * 60)
    print()

    # =========================================================================
    # Step 0: Reset state for clean example run
    # =========================================================================
    reset_for_conv_example(verbose=True)

    # =========================================================================
    # Step 1: Define benchmark problems (small for quick runs)
    # =========================================================================
    print("\nBENCHMARK PROBLEMS")
    print("=" * 60)

    problems = [
        # Small problems for quick benchmarking
        ConvProblem(N=1, C=64, K=64, Hi=14, Wi=14, Y=3, X=3, pad_h=1, pad_w=1),
        ConvProblem(N=1, C=128, K=128, Hi=14, Wi=14, Y=3, X=3, pad_h=1, pad_w=1),
        # Pointwise (fast)
        ConvProblem(N=1, C=64, K=128, Hi=14, Wi=14, Y=1, X=1),
        # Larger problem
        ConvProblem(N=1, C=256, K=256, Hi=28, Wi=28, Y=3, X=3, pad_h=1, pad_w=1),
    ]

    for p in problems:
        print(f"  {p}")
    print()

    # =========================================================================
    # Step 2: Define kernel configurations
    # =========================================================================
    print("KERNEL CONFIGURATIONS")
    print("=" * 60)

    kernel_set = ConvKernelSet("benchmark_kernels")
    arch = ArchInfo(name=args.arch)

    for tile_k, tile_c in [(64, 64), (128, 128)]:
        sig = ConvSignature()
        sig.dtype(args.dtype)
        sig.layout = "nhwgc"
        sig.direction = "forward"

        algo = ConvAlgorithm()
        algo.tile(1, tile_k, tile_c)
        algo.wave(2, 2, 1)
        algo.warp(32, 32, 16)
        algo.pipeline = args.pipeline
        algo.scheduler = "intrawave"

        # Validate configuration
        validation = validate_conv_config(
            pipeline=algo.pipeline,
            scheduler=algo.scheduler,
            epilogue=algo.epilogue,
            wave_m=algo.wave_m,
            wave_n=algo.wave_n,
            wave_k=algo.wave_k,
            warp_m=algo.warp_m,
            warp_n=algo.warp_n,
            warp_k=algo.warp_k,
            dtype=sig.dtype_in,
            arch=arch.name,
        )

        if validation.is_valid:
            kernel_set.add(sig, algo, arch)
        else:
            print(f"  [SKIPPED] tile={tile_k}x{tile_c}: invalid for {args.arch}")

    kernel_set.print()

    # Print one config for reference
    if kernel_set.configs:
        cfg = kernel_set.configs[0]
        print_conv_kernel_config(cfg.signature, cfg.algorithm, cfg.arch)
    print()

    # =========================================================================
    # Step 3: GPU Benchmark
    # =========================================================================
    print("GPU BENCHMARKS")
    print("=" * 60)

    runner = GpuConvRunner()
    if runner.is_available():
        print(f"  Library: {runner.library_path}")
        print()

        # Determine numpy dtype
        np_dtype = {
            "fp16": np.float16,
            "bf16": np.float16,
            "fp32": np.float32,
        }[args.dtype]

        print(f"{'Problem':<40} | {'Time (ms)':>10} | {'TFLOPS':>8}")
        print("-" * 65)

        for prob in problems:
            # Create data with correct dtype
            input_host = np.random.randn(
                prob.N, prob.Hi, prob.Wi, prob.G, prob.C
            ).astype(np_dtype)
            weight_host = np.random.randn(
                prob.G, prob.K, prob.Y, prob.X, prob.C // prob.G
            ).astype(np_dtype)

            # Run
            result = runner.run_forward(input_host, weight_host, prob)

            prob_str = f"C={prob.C} K={prob.K} {prob.Hi}x{prob.Wi} {prob.Y}x{prob.X}"
            if result.get("success"):
                time_ms = result["time_ms"]
                tflops = result["tflops"]
                print(f"{prob_str:<40} | {time_ms:>10.4f} | {tflops:>8.2f}")
            else:
                print(f"{prob_str:<40} | {'N/A':>10} | {'N/A':>8}")

        print()
        print("*** GPU BENCHMARK COMPLETE ***")
        runner.cleanup()
    else:
        print("  Library not available")
        print(
            "  Build with: cd dispatcher/build && cmake .. && make dispatcher_conv_lib"
        )

    # =========================================================================
    # Optional: CPU Reference (slow, use --cpu flag)
    # =========================================================================
    if args.cpu:
        print()
        print("CPU REFERENCE (slow)")
        print("=" * 60)

        import time

        # Only test smallest problem
        prob = problems[0]
        np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

        input_data = np.random.randn(prob.N, prob.Hi, prob.Wi, prob.C).astype(np_dtype)
        weight = np.random.randn(prob.K, prob.Y, prob.X, prob.C // prob.G).astype(
            np_dtype
        )

        start = time.perf_counter()
        # Naive convolution (just one iteration)
        padded = np.pad(
            input_data,
            ((0, 0), (prob.pad_h, prob.pad_h), (prob.pad_w, prob.pad_w), (0, 0)),
        )
        output = np.zeros((prob.N, prob.Ho, prob.Wo, prob.K), dtype=np_dtype)

        for n in range(prob.N):
            for ho in range(prob.Ho):
                for wo in range(prob.Wo):
                    for k in range(prob.K):
                        acc = 0.0
                        for y in range(prob.Y):
                            for x in range(prob.X):
                                for c in range(prob.C):
                                    hi = ho * prob.stride_h + y
                                    wi = wo * prob.stride_w + x
                                    acc += float(padded[n, hi, wi, c]) * float(
                                        weight[k, y, x, c]
                                    )
                        output[n, ho, wo, k] = acc

        elapsed_ms = (time.perf_counter() - start) * 1000
        gflops = (prob.flops / (elapsed_ms * 1e-3)) / 1e9
        print(f"  Problem: C={prob.C} K={prob.K} {prob.Hi}x{prob.Wi}")
        print(f"  Time: {elapsed_ms:.2f} ms, GFLOPS: {gflops:.2f}")

    # =========================================================================
    # Cleanup and Summary
    # =========================================================================
    cleanup_conv()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Data Type: {args.dtype}")
    print(f"  Pipeline:  {args.pipeline}")
    print(f"  Arch:      {args.arch}")
    print(f"  Problems:  {len(problems)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
