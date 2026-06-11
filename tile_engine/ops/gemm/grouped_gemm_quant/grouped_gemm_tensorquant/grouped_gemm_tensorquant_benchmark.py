# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import sys
import time
import argparse
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple


def _import_gemm_benchmark():
    """Import gemm benchmark from gemm directory."""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    gemm_dir = os.path.dirname(os.path.dirname(current_dir))

    spec = importlib.util.spec_from_file_location(
        "gemm_benchmark", os.path.join(gemm_dir, "gemm_benchmark.py")
    )

    gemm_benchmark_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_benchmark_module)

    return gemm_benchmark_module.GemmBenchmark


def _import_benchmark_utils():
    """Import benchmark utils from common directory."""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    ops_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

    spec = importlib.util.spec_from_file_location(
        "benchmark_utils", os.path.join(ops_dir, "common", "benchmark_utils.py")
    )

    benchmark_utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark_utils_module)

    return benchmark_utils_module


GemmBenchmark = _import_gemm_benchmark()
benchmark_utils = _import_benchmark_utils()


class GroupedTensorQuantGemmBenchmark(GemmBenchmark):
    def __init__(self, build_dir, verbose=False):
        super().__init__(build_dir, verbose, name="benchmark_grouped_gemm_tensorquant")

    def benchmark_problem_size(
        self,
        kernels: List[Path],
        group_count: int = 8,
        m: int = 3840,
        n: int = 4096,
        k: int = 2048,
        kbatch: int = 1,
        verify: int = 0,
        warmup: int = 50,
        repeat: int = 100,
        flush_cache: bool = True,
        rotating_count: int = 1000,
    ) -> List[Dict]:
        """Benchmark all kernels for a specific grouped tensorquant problem size."""
        results = []

        params = {
            "group_count": group_count,
            "m": m,
            "n": n,
            "k": k,
            "kbatch": kbatch,
            "verify": verify,
            "warmup": warmup,
            "repeat": repeat,
            "flush_cache": str(flush_cache).lower(),
            "rotating_count": rotating_count,
        }

        print(
            f"\nBenchmarking group_count={group_count}, M={m}, N={n}, K={k}, kbatch={kbatch}"
        )

        for kernel_path in kernels:
            kernel_info = self.extract_kernel_info(kernel_path)
            result = benchmark_utils.run_kernel(
                self.build_dir, kernel_path, params, verbose=self.verbose
            )

            if result:
                structured_result = {
                    "name": kernel_info["name"],
                    "config_id": kernel_info["config_id"],
                    "problem": result.get("problem", {}),
                    "perf_result": result.get("perf_result", {}),
                    "config": {
                        "data_type": kernel_info["data_type"],
                        "layout": kernel_info["layout"],
                        "pipeline": kernel_info["pipeline"],
                        "scheduler": kernel_info["scheduler"],
                        "epilogue": kernel_info["epilogue"],
                        "tile_sizes": kernel_info.get("tile_sizes", {}),
                        "warp_config": kernel_info.get("warp_config", {}),
                        "warp_tile": kernel_info.get("warp_tile", {}),
                        "optimization_flags": kernel_info.get("optimization_flags", {}),
                    },
                    "executable": kernel_info["executable"],
                    "time_ms": result.get("time_ms", 0),
                    "tflops": result.get("tflops", 0),
                    "bandwidth_gb_s": result.get("bandwidth_gb_s", 0),
                }

                results.append(structured_result)

                if self.verbose:
                    print(
                        f"  {kernel_info['config_id']}: {structured_result['tflops']:.2f} TFLOPS, "
                        f"{structured_result['bandwidth_gb_s']:.2f} GB/s, {structured_result['time_ms']:.2f}ms"
                    )

        return results

    def benchmark_sweep(
        self,
        problem_sizes: List[Tuple[int, int, int]],
        group_counts: List[int] = [8],
        kbatch_values: List[int] = [1],
        verify: bool = False,
        warmup: int = 50,
        repeat: int = 100,
        flush_cache: bool = True,
        rotating_count: int = 1000,
    ) -> Dict:
        """Run comprehensive benchmark sweep for grouped tensor quant GEMM."""
        kernels = self.discover_kernels()
        if not kernels:
            print("No kernels found!")
            return {}

        all_results = []
        best_kernels = {}

        for m, n, k in problem_sizes:
            for group_count in group_counts:
                for kbatch in kbatch_values:
                    results = self.benchmark_problem_size(
                        kernels,
                        group_count=group_count,
                        m=m,
                        n=n,
                        k=k,
                        kbatch=kbatch,
                        verify=1 if verify else 0,
                        warmup=warmup,
                        repeat=repeat,
                        flush_cache=flush_cache,
                        rotating_count=rotating_count,
                    )

                    all_results.extend(results)

                    # Find best kernel for this configuration
                    best = benchmark_utils.find_best_kernel(results)
                    if best:
                        key = f"g{group_count}_m{m}_n{n}_k{k}_kbatch{kbatch}"
                        best_kernels[key] = best
                        print(
                            f"Best for {key}: {best['name']} ({best['tflops']:.2f} TFLOPS, "
                            f"{best['bandwidth_gb_s']:.2f} GB/s, {best['time_ms']:.2f}ms)"
                        )

        self.results = all_results
        return best_kernels


def main():
    parser = argparse.ArgumentParser(
        description="Grouped TensorQuant GEMM Kernel Benchmarking Tool"
    )
    parser.add_argument(
        "build_dir", help="Build directory containing kernel executables"
    )
    parser.add_argument(
        "--problem-sizes",
        nargs="+",
        default=["1024,1024,1024"],
        help="Default problem sizes as M,N,K tuples (used for all groups)",
    )
    parser.add_argument(
        "--group-counts",
        nargs="+",
        type=int,
        default=[8],
        help="Group count values to test (default: 8)",
    )
    parser.add_argument(
        "--kbatch",
        nargs="+",
        type=int,
        default=[1],
        help="K-batch (SplitK) values to test",
    )
    parser.add_argument("--verify", action="store_true", help="Enable verification")
    parser.add_argument(
        "--csv",
        default="grouped_gemm_tensorquant_benchmark_results.csv",
        help="CSV output filename",
    )
    parser.add_argument(
        "--best",
        default="best_grouped_gemm_tensorquant_kernels.txt",
        help="Best kernels output filename",
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
    benchmark = GroupedTensorQuantGemmBenchmark(args.build_dir, verbose=args.verbose)

    # Run benchmark sweep
    print("Starting Grouped TensorQuant GEMM kernel benchmark sweep...")
    print(f"Problem sizes: {problem_sizes}")
    print(f"Group counts: {args.group_counts}")
    print(f"K-batch values: {args.kbatch}")
    start_time = time.time()

    best_kernels = benchmark.benchmark_sweep(
        problem_sizes,
        group_counts=args.group_counts,
        kbatch_values=args.kbatch,
        verify=args.verify,
        warmup=args.warmup,
        repeat=args.repeat,
        flush_cache=args.flush_cache,
        rotating_count=args.rotating_count,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal benchmark time: {elapsed_time:.2f} seconds")

    # Export results
    if benchmark.results:
        benchmark_utils.export_csv(benchmark.results, args.csv)
        benchmark_utils.export_best_kernels(best_kernels, args.best)
        if args.json:
            benchmark_utils.export_json(benchmark.results, args.json, best_kernels)

    print("\nBenchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
