# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import sys
import json
import csv
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


class GroupedRowColQuantGemmBenchmark(GemmBenchmark):
    def __init__(self, build_dir, verbose=False):
        super().__init__(build_dir, verbose, name="benchmark_grouped_gemm_rowcolquant")

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
        """Benchmark all kernels for a specific grouped rowcolquant problem size."""
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
        """Run comprehensive benchmark sweep for grouped row/col quant GEMM."""
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

    def export_csv(self, filename: str):
        """Export all results to CSV."""
        if not self.results:
            print("No results to export")
            return

        # Get all unique keys from results
        all_keys = set()
        for result in self.results:
            all_keys.update(result.keys())

        # Sort keys for consistent output
        fieldnames = sorted(all_keys)

        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Results exported to {filename}")

    def export_best_kernels(self, best_kernels: Dict, filename: str):
        """Export best kernel selections to file."""
        with open(filename, "w") as f:
            f.write("# Best kernel selections for grouped rowcolquant GEMM\n")
            f.write(
                "# Format: problem_size -> kernel_name (TFLOPS, bandwidth, latency)\n\n"
            )

            for key, kernel in sorted(best_kernels.items()):
                f.write(
                    f"{key}: {kernel['name']} ({kernel['tflops']:.2f} TFLOPS, "
                    f"{kernel['bandwidth_gb_s']:.2f} GB/s, {kernel['time_ms']:.2f}ms)\n"
                )

        print(f"Best kernels exported to {filename}")

    def export_json(self, filename: str, best_kernels: Dict = None):
        """Export all results and best kernels to JSON with comprehensive metadata."""
        from datetime import datetime

        # Calculate comprehensive summary statistics for all metrics
        successful_results = [r for r in self.results if r.get("tflops", 0) > 0]

        tflops_values = [r.get("tflops", 0) for r in successful_results]
        bandwidth_values = [r.get("bandwidth_gb_s", 0) for r in successful_results]
        latency_values = [
            r.get("time_ms", 0) for r in successful_results if r.get("time_ms", 0) > 0
        ]

        # Performance breakdown by kernel type
        pipeline_stats = {}
        scheduler_stats = {}
        data_type_stats = {}

        for result in successful_results:
            config = result.get("config", {})

            # Pipeline statistics
            pipeline = config.get("pipeline", "unknown")
            if pipeline not in pipeline_stats:
                pipeline_stats[pipeline] = {
                    "count": 0,
                    "avg_tflops": 0,
                    "best_tflops": 0,
                }
            pipeline_stats[pipeline]["count"] += 1
            pipeline_stats[pipeline]["best_tflops"] = max(
                pipeline_stats[pipeline]["best_tflops"], result.get("tflops", 0)
            )

            # Scheduler statistics
            scheduler = config.get("scheduler", "unknown")
            if scheduler not in scheduler_stats:
                scheduler_stats[scheduler] = {
                    "count": 0,
                    "avg_tflops": 0,
                    "best_tflops": 0,
                }
            scheduler_stats[scheduler]["count"] += 1
            scheduler_stats[scheduler]["best_tflops"] = max(
                scheduler_stats[scheduler]["best_tflops"], result.get("tflops", 0)
            )

            # Data type statistics
            data_type = config.get("data_type", "unknown")
            if data_type not in data_type_stats:
                data_type_stats[data_type] = {
                    "count": 0,
                    "avg_tflops": 0,
                    "best_tflops": 0,
                }
            data_type_stats[data_type]["count"] += 1
            data_type_stats[data_type]["best_tflops"] = max(
                data_type_stats[data_type]["best_tflops"], result.get("tflops", 0)
            )

        # Calculate averages for breakdown stats
        for stats_dict, field_name in [
            (pipeline_stats, "pipeline"),
            (scheduler_stats, "scheduler"),
            (data_type_stats, "data_type"),
        ]:
            for key in stats_dict:
                relevant_results = [
                    r
                    for r in successful_results
                    if r.get("config", {}).get(field_name, "unknown") == key
                ]
                if relevant_results:
                    stats_dict[key]["avg_tflops"] = sum(
                        r.get("tflops", 0) for r in relevant_results
                    ) / len(relevant_results)

        output_data = {
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
                "operation": "grouped_gemm_rowcolquant",
                "total_kernels_tested": len(self.results),
                "unique_kernels": len(
                    set(r.get("name", "unknown") for r in self.results)
                ),
                "successful_runs": len(successful_results),
                "failed_runs": len(self.results) - len(successful_results),
            },
            "performance_summary": {
                "tflops_stats": {
                    "best": max(tflops_values, default=0),
                    "average": (
                        sum(tflops_values) / len(tflops_values) if tflops_values else 0
                    ),
                    "min": min(tflops_values, default=0),
                    "median": (
                        sorted(tflops_values)[len(tflops_values) // 2]
                        if tflops_values
                        else 0
                    ),
                },
                "bandwidth_stats": {
                    "best_gb_s": max(bandwidth_values, default=0),
                    "average_gb_s": (
                        sum(bandwidth_values) / len(bandwidth_values)
                        if bandwidth_values
                        else 0
                    ),
                    "min_gb_s": min(bandwidth_values, default=0),
                    "median_gb_s": (
                        sorted(bandwidth_values)[len(bandwidth_values) // 2]
                        if bandwidth_values
                        else 0
                    ),
                },
                "latency_stats": {
                    "best_ms": min(latency_values, default=0),
                    "average_ms": (
                        sum(latency_values) / len(latency_values)
                        if latency_values
                        else 0
                    ),
                    "max_ms": max(latency_values, default=0),
                    "median_ms": (
                        sorted(latency_values)[len(latency_values) // 2]
                        if latency_values
                        else 0
                    ),
                },
                "kernel_type_breakdown": {
                    "by_pipeline": pipeline_stats,
                    "by_scheduler": scheduler_stats,
                    "by_data_type": data_type_stats,
                },
                "total_problem_configurations": (
                    len(best_kernels) if best_kernels else 0
                ),
            },
            "kernel_results": self.results,
            "best_kernels_by_problem": best_kernels or {},
        }

        with open(filename, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"JSON results exported to {filename}")
        print(f"  - Total kernels: {len(self.results)}")
        print(f"  - Successful runs: {len(successful_results)}")
        print(f"  - Best TFLOPS: {max(tflops_values, default=0):.2f}")
        print(f"  - Best bandwidth: {max(bandwidth_values, default=0):.2f} GB/s")
        print(f"  - Best latency: {min(latency_values, default=0):.2f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Grouped RowColQuant GEMM Kernel Benchmarking Tool"
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
        default="grouped_gemm_rowcolquant_benchmark_results.csv",
        help="CSV output filename",
    )
    parser.add_argument(
        "--best",
        default="best_grouped_gemm_rowcolquant_kernels.txt",
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
    benchmark = GroupedRowColQuantGemmBenchmark(args.build_dir, verbose=args.verbose)

    # Run benchmark sweep
    print("Starting Grouped RowColQuant GEMM kernel benchmark sweep...")
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
