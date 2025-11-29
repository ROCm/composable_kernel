#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 06: Convolution Benchmarking

Demonstrates benchmarking convolution kernels across multiple problem sizes.

Usage:
    python3 06_benchmark.py
    python3 06_benchmark.py --cpu  # Include slow CPU reference
"""

import argparse
import numpy as np
from conv_utils import (
    ConvSignature,
    ConvAlgorithm,
    ArchInfo,
    ConvKernelSet,
    ConvProblem,
)


def main():
    parser = argparse.ArgumentParser(description="Convolution Benchmarking")
    parser.add_argument(
        "--cpu", action="store_true", help="Include CPU reference (slow)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Example 06: Convolution Benchmarking")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # Step 1: Define benchmark problems (small for quick runs)
    # -------------------------------------------------------------------------
    print("BENCHMARK PROBLEMS")
    print("=" * 40)

    problems = [
        # Small problems for quick benchmarking
        ConvProblem(N=1, C=64, K=64, Hi=14, Wi=14, Y=3, X=3, pad_h=1, pad_w=1),
        ConvProblem(N=1, C=128, K=128, Hi=14, Wi=14, Y=3, X=3, pad_h=1, pad_w=1),
        # Pointwise (fast)
        ConvProblem(N=1, C=64, K=128, Hi=14, Wi=14, Y=1, X=1),
    ]

    for p in problems:
        print(f"  {p}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Define kernel configurations
    # -------------------------------------------------------------------------
    print("KERNEL CONFIGURATIONS")
    print("=" * 40)

    kernel_set = ConvKernelSet("benchmark_kernels")

    for tile_k, tile_c in [(64, 64), (128, 128)]:
        sig = ConvSignature()
        sig.dtype("fp16")
        sig.layout = "nhwc"
        sig.direction = "forward"

        algo = ConvAlgorithm()
        algo.tile(1, tile_k, tile_c)
        algo.wave(2, 2, 1)
        algo.pipeline = "compv4"

        kernel_set.add(sig, algo, ArchInfo(name="gfx942"))

    kernel_set.print()
    print()

    # -------------------------------------------------------------------------
    # Step 3: GPU Benchmark
    # -------------------------------------------------------------------------
    print("GPU BENCHMARKS")
    print("=" * 40)

    try:
        from conv_utils import ConvDispatcherLib
        import ctypes

        lib = ConvDispatcherLib.auto()
        if lib:
            print(f"  Library: {lib.path}")

            # Load HIP
            hip = ctypes.CDLL("libamdhip64.so")
            hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            hip.hipMalloc.restype = ctypes.c_int
            hip.hipFree.argtypes = [ctypes.c_void_p]
            hip.hipFree.restype = ctypes.c_int
            hip.hipMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            hip.hipMemcpy.restype = ctypes.c_int

            print()
            print(f"{'Problem':<35} | {'Time (ms)':>10} | {'TFLOPS':>8}")
            print("-" * 60)

            for prob in problems:
                # Create data
                input_host = np.random.randn(prob.N, prob.Hi, prob.Wi, prob.C).astype(
                    np.float16
                )
                weight_host = np.random.randn(
                    prob.K, prob.Y, prob.X, prob.C // prob.G
                ).astype(np.float16)

                # Allocate GPU
                input_dev = ctypes.c_void_p()
                weight_dev = ctypes.c_void_p()
                output_dev = ctypes.c_void_p()

                hip.hipMalloc(ctypes.byref(input_dev), input_host.nbytes)
                hip.hipMalloc(ctypes.byref(weight_dev), weight_host.nbytes)
                hip.hipMalloc(
                    ctypes.byref(output_dev), prob.N * prob.Ho * prob.Wo * prob.K * 2
                )

                # Copy to device
                hip.hipMemcpy(input_dev, input_host.ctypes.data, input_host.nbytes, 1)
                hip.hipMemcpy(
                    weight_dev, weight_host.ctypes.data, weight_host.nbytes, 1
                )

                # Run
                time_ms = lib.run(
                    input_dev.value, weight_dev.value, output_dev.value, prob
                )

                # Free
                hip.hipFree(input_dev)
                hip.hipFree(weight_dev)
                hip.hipFree(output_dev)

                if time_ms > 0:
                    tflops = prob.flops / (time_ms * 1e9)
                    prob_str = (
                        f"C={prob.C} K={prob.K} {prob.Hi}x{prob.Wi} {prob.Y}x{prob.X}"
                    )
                    print(f"{prob_str:<35} | {time_ms:>10.4f} | {tflops:>8.2f}")
                else:
                    prob_str = (
                        f"C={prob.C} K={prob.K} {prob.Hi}x{prob.Wi} {prob.Y}x{prob.X}"
                    )
                    print(f"{prob_str:<35} | {'N/A':>10} | {'N/A':>8}")

            print()
            print("*** GPU BENCHMARK COMPLETE ***")
        else:
            print("  Library not available")
    except Exception as e:
        print(f"  Error: {e}")

    # -------------------------------------------------------------------------
    # Optional: CPU Reference (slow, use --cpu flag)
    # -------------------------------------------------------------------------
    if args.cpu:
        print()
        print("CPU REFERENCE (slow)")
        print("=" * 40)

        import time

        # Only test smallest problem
        prob = problems[0]
        input_data = np.random.randn(prob.N, prob.Hi, prob.Wi, prob.C).astype(
            np.float16
        )
        weight = np.random.randn(prob.K, prob.Y, prob.X, prob.C // prob.G).astype(
            np.float16
        )

        start = time.perf_counter()
        # Naive convolution (just one iteration)
        padded = np.pad(
            input_data,
            ((0, 0), (prob.pad_h, prob.pad_h), (prob.pad_w, prob.pad_w), (0, 0)),
        )
        output = np.zeros((prob.N, prob.Ho, prob.Wo, prob.K), dtype=np.float16)

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

    print()
    print("=" * 60)
    print("Benchmark completed!")


if __name__ == "__main__":
    main()
