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

    spec = importlib.util.spec_from_file_location(
        "benchmark_utils",
        os.path.join(parent_dir, "common", "benchmark_utils.py"),
    )
    benchmark_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark_utils)

    return benchmark_utils


GemmBenchmark = _import_gemm_benchmark()
benchmark_utils = _import_benchmark_utils()


class MxGemmBenchmark(GemmBenchmark):
    def __init__(self, build_dir: str, verbose: bool = False):
        super().__init__(build_dir, verbose, name="benchmark_mx_gemm_")

    def extract_kernel_info(self, kernel_path):
        """Extract MX GEMM kernel metadata from benchmark executable names."""
        name = kernel_path.stem
        if name.startswith(self.name):
            args = name[len(self.name) :]
        else:
            args = name

        info = {
            "executable": str(kernel_path),
            "name": name,
            "data_type": "unknown",
            "layout": "unknown",
            "pipeline": "unknown",
            "scheduler": "unknown",
            "epilogue": "unknown",
        }

        parts = args.split("_")
        if len(parts) >= 6 and parts[2] == "comp" and parts[3] == "async":
            info["data_type"] = parts[0]
            info["layout"] = parts[1]
            info["pipeline"] = "comp_async"
            info["epilogue"] = parts[4]
            info["scheduler"] = parts[5]
        elif len(parts) >= 5:
            info["data_type"] = parts[0]
            info["layout"] = parts[1]
            info["pipeline"] = parts[2]
            info["epilogue"] = parts[3]
            info["scheduler"] = parts[4]

        config_info = self.parse_detailed_config(name)
        info.update(config_info)
        info["config_id"] = self.generate_config_id(info)

        return info


def main():
    parser = argparse.ArgumentParser(description="MX GEMM Kernel Benchmarking Tool")
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
        default="mx_gemm_benchmark_results.csv",
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
    flush_cache_group = parser.add_mutually_exclusive_group()
    flush_cache_group.add_argument(
        "--flush-cache",
        dest="flush_cache",
        action="store_true",
        help="Enable cache flushing (default: True)",
    )
    flush_cache_group.add_argument(
        "--no-flush-cache",
        dest="flush_cache",
        action="store_false",
        help="Disable cache flushing",
    )
    parser.set_defaults(flush_cache=True)
    parser.add_argument(
        "--rotating-count",
        type=int,
        default=1000,
        help="Number of iterations to rotate cache (default: 1000)",
    )
    parser.add_argument("--json", help="JSON output filename (optional)")

    args = parser.parse_args()

    problem_sizes = []
    for size_str in args.problem_sizes:
        try:
            m, n, k = map(int, size_str.split(","))
            problem_sizes.append((m, n, k))
        except ValueError:
            print(f"Invalid problem size: {size_str}")
            return 1

        if k % 32 != 0:
            print(f"Invalid problem size: {size_str}; MX GEMM requires K divisible by 32")
            return 1

    benchmark = MxGemmBenchmark(args.build_dir, verbose=args.verbose)

    print("Starting MX GEMM kernel benchmark sweep...")
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

    benchmark_utils.export_csv(benchmark.results, args.csv)
    benchmark_utils.export_best_kernels(best_kernels, args.best)

    if args.json:
        benchmark_utils.export_json(benchmark.results, args.json, best_kernels)

    return 0


if __name__ == "__main__":
    sys.exit(main())
