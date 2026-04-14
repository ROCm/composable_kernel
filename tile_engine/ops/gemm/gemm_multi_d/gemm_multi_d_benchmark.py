#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import sys
import argparse
import time
import importlib.util


def _import_gemm_benchmark():
    """Import gemm benchmark from parent directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(
        "gemm_benchmark",
        os.path.join(parent_dir, "gemm_benchmark.py"),
    )
    gemm_benchmark_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_benchmark_module)

    return gemm_benchmark_module.GemmBenchmark


def _import_benchmark_utils():
    """Import benchmark utilities from commons directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(
        "benchmark_utils",
        os.path.join(parent_dir, "common", "benchmark_utils.py"),
    )
    benchmark_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark_utils)

    return benchmark_utils


GemmBenchmark = _import_gemm_benchmark()
benchmark_utils = _import_benchmark_utils()


class GemmMultiDBenchmark(GemmBenchmark):
    def __init__(self, build_dir: str, verbose: bool = False):
        super().__init__(build_dir, verbose, name="benchmark_gemm_multi_d_")


def main():
    parser = argparse.ArgumentParser(
        description="GEMM Multi D Kernel Benchmarking Tool"
    )
    parser.add_argument(
        "build_dir", help="Build directory containing kernel executables"
    )
    parser.add_argument(
        "--problem-sizes",
        nargs="+",
        default=["1024,1024,1024", "2048,2048,2048", "4096,4096,4096"],
        help="Problem sizes as M,N,K tuples",
    )
    parser.add_argument(
        "--split-k", nargs="+", type=int, default=[1], help="Split-K values to test"
    )
    parser.add_argument("--verify", action="store_true", help="Enable verification")
    parser.add_argument(
        "--csv",
        default="gemm_multi_d_benchmark_results.csv",
        help="CSV output filename",
    )
    parser.add_argument(
        "--best", default="best_kernels.txt", help="Best kernels output filename"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        default=True,
        help="Enable cache flushing (default: True)",
    )
    parser.add_argument(
        "--rotating-count",
        type=int,
        default=1000,
        help="Number of iterations to rotate cache (default: 1000)",
    )
    parser.add_argument("--json", help="JSON output filename (optional)")

    args = parser.parse_args()

    # Parse problem sizes
    problem_sizes = []
    for size_str in args.problem_sizes:
        try:
            m, n, k = map(int, size_str.split(","))
            problem_sizes.append((m, n, k))
        except ValueError:
            print(f"Invalid problem size: {size_str}")
            return 1

    # Create benchmark instance
    benchmark = GemmMultiDBenchmark(args.build_dir, verbose=args.verbose)

    # Run benchmark sweep
    print("Starting GEMM Multi D kernel benchmark sweep...")
    start_time = time.time()

    best_kernels = benchmark.benchmark_sweep(
        problem_sizes=problem_sizes,
        split_k_values=args.split_k,
        verify=args.verify,
        warmup=args.warmup,
        repeat=args.repeat,
        flush_cache=args.flush_cache,
        rotating_count=args.rotating_count,
    )

    elapsed_time = time.time() - start_time
    print(f"\nBenchmark completed in {elapsed_time:.2f} seconds")

    # Export results
    benchmark_utils.export_csv(benchmark.results, args.csv)
    benchmark_utils.export_best_kernels(best_kernels, args.best)

    # Export JSON if requested
    if args.json:
        benchmark_utils.export_json(benchmark.results, args.json, best_kernels)

    return 0


if __name__ == "__main__":
    sys.exit(main())
