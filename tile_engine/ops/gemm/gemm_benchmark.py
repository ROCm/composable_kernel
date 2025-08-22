#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import sys
import json
import subprocess
import argparse
import csv
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class GemmBenchmark:
    def __init__(self, build_dir: str, verbose: bool = False):
        self.build_dir = Path(build_dir)
        self.verbose = verbose
        self.results = []

    def discover_kernels(self) -> List[Path]:
        """Find all benchmark_gemm_* executables in the build directory"""
        bin_dir = self.build_dir / "bin"
        if not bin_dir.exists():
            print(f"Error: Binary directory {bin_dir} does not exist")
            return []

        kernels = list(bin_dir.glob("benchmark_gemm_*"))
        if self.verbose:
            print(f"Found {len(kernels)} kernel executables")
            for k in kernels:
                print(f"  - {k.name}")
        return kernels

    def extract_kernel_info(self, kernel_path: Path) -> Dict[str, str]:
        """Extract kernel information from filename"""
        name = kernel_path.stem

        info = {
            "executable": str(kernel_path),
            "name": name,
            "data_type": "unknown",
            "layout": "unknown",
            "pipeline": "unknown",
            "scheduler": "unknown",
            "epilogue": "unknown",
        }

        # Extract data type
        for dtype in ["fp16", "fp32", "fp64", "bf16", "fp8", "bf8", "int8", "int32"]:
            if dtype in name:
                info["data_type"] = dtype
                break

        # Extract layout
        for layout in ["rrr", "rcr", "rrc", "rcc", "crr", "crc", "ccr", "ccc"]:
            if layout in name:
                info["layout"] = layout
                break

        # Extract pipeline
        for pipeline in ["compv3", "compv4", "mem"]:
            if pipeline in name:
                info["pipeline"] = pipeline
                break

        # Extract scheduler
        if "interwave" in name:
            info["scheduler"] = "interwave"
        else:
            info["scheduler"] = "intrawave"

        # Extract epilogue
        if "default" in name and "default_" not in name:
            info["epilogue"] = "default"
        else:
            info["epilogue"] = "cshuffle"

        return info

    def run_kernel(self, kernel_path: Path, params: Dict[str, str]) -> Optional[Dict]:
        """Run a single kernel with given parameters and save output to individual JSON file"""
        # Create results directory
        results_dir = self.build_dir / "results"
        results_dir.mkdir(exist_ok=True)

        # Generate unique JSON filename for this kernel
        json_file = results_dir / f"{kernel_path.stem}.json"

        cmd = [str(kernel_path)]

        # Add parameters
        for key, value in params.items():
            cmd.append(f"-{key}={value}")

        # Add JSON output flag for clean JSON output
        cmd.append("-json_output=true")

        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"Error running {kernel_path.name}: {result.stderr}")
                return None

            # Save raw output to individual JSON file
            output = result.stdout.strip()
            if output:
                with open(json_file, "w") as f:
                    f.write(output)

                # Parse the JSON file
                return self.parse_json_file(json_file)
            else:
                print(f"No output from {kernel_path.name}")
                return None

        except subprocess.TimeoutExpired:
            print(f"Timeout running {kernel_path.name}")
            return None
        except Exception as e:
            print(f"Error running {kernel_path.name}: {e}")
            return None

    def parse_json_file(self, json_file: Path) -> Optional[Dict]:
        """Parse JSON data from individual kernel output file"""
        try:
            with open(json_file, "r") as f:
                content = f.read().strip()

            # Parse the JSON directly since executables produce clean JSON
            data = json.loads(content)

            # Return the complete JSON data as-is, just add some convenience fields
            result = data.copy()
            if "perf_result" in data:
                perf = data["perf_result"]
                # Add convenience fields for backward compatibility
                result["time_ms"] = perf.get("latency(ms)", 0)
                result["tflops"] = perf.get("tflops(TFlops)", 0)
                result["bandwidth_gb_s"] = perf.get("bandwidth(GB/s)", 0)

            return result

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"Failed to parse JSON from {json_file}: {e}")
            return None
        except Exception as e:
            if self.verbose:
                print(f"Error reading JSON file {json_file}: {e}")
            return None

    def parse_benchmark_output(self, output: str) -> Optional[Dict]:
        """Parse the benchmark output format - extract JSON directly"""
        try:
            # Find JSON block between asterisk markers
            lines = output.split("\n")
            json_start = -1
            json_end = -1

            for i, line in enumerate(lines):
                if line.strip().startswith("{"):
                    json_start = i
                elif line.strip().endswith("}") and json_start != -1:
                    json_end = i
                    break

            if json_start != -1 and json_end != -1:
                json_text = "\n".join(lines[json_start : json_end + 1])
                data = json.loads(json_text)

                # Return the complete JSON data as-is, just add some convenience fields
                result = data.copy()
                if "perf_result" in data:
                    perf = data["perf_result"]
                    # Add convenience fields for backward compatibility
                    result["time_ms"] = perf.get("latency(ms)", 0)
                    result["tflops"] = perf.get("tflops(TFlops)", 0)
                    result["bandwidth_gb_s"] = perf.get("bandwidth(GB/s)", 0)

                return result

            return None

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"Failed to parse JSON: {e}")
                print(f"Output was: {output[:200]}...")
            return None
        except Exception as e:
            if self.verbose:
                print(f"Error parsing output: {e}")
            return None

    def benchmark_problem_size(
        self,
        kernels: List[Path],
        m: int,
        n: int,
        k: int,
        split_k: int = 1,
        verify: int = 0,
        warmup: int = 50,
        repeat: int = 100,
        flush_cache: bool = True,
        rotating_count: int = 1000,
    ) -> List[Dict]:
        """Benchmark all kernels for a specific problem size"""
        results = []

        params = {
            "m": m,
            "n": n,
            "k": k,
            "split_k": split_k,
            "verify": verify,
            "warmup": warmup,
            "repeat": repeat,
            "flush_cache": str(flush_cache).lower(),
            "rotating_count": rotating_count,
        }

        print(f"\nBenchmarking M={m}, N={n}, K={k}, split_k={split_k}")

        for kernel_path in kernels:
            kernel_info = self.extract_kernel_info(kernel_path)
            result = self.run_kernel(kernel_path, params)

            if result:
                # Merge kernel info and benchmark result
                result.update(kernel_info)
                results.append(result)

                if self.verbose:
                    print(
                        f"  {kernel_info['name']}: {result['tflops']:.2f} TFLOPS, {result['bandwidth_gb_s']:.2f} GB/s, {result['time_ms']:.2f}ms"
                    )

        return results

    def find_best_kernel(
        self, results: List[Dict], metric: str = "tflops"
    ) -> Optional[Dict]:
        """Find the best performing kernel based on metric"""
        if not results:
            return None

        if metric == "tflops":
            return max(results, key=lambda x: x.get("tflops", 0))
        elif metric == "time_ms":
            return min(results, key=lambda x: x.get("time_ms", float("inf")))
        elif metric == "bandwidth_gb_s":
            return max(results, key=lambda x: x.get("bandwidth_gb_s", 0))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def benchmark_sweep(
        self,
        problem_sizes: List[Tuple[int, int, int]],
        split_k_values: List[int] = [1],
        verify: bool = False,
        warmup: int = 50,
        repeat: int = 100,
        flush_cache: bool = True,
        rotating_count: int = 1000,
    ) -> Dict:
        """Run comprehensive benchmark sweep"""
        kernels = self.discover_kernels()
        if not kernels:
            print("No kernels found!")
            return {}

        all_results = []
        best_kernels = {}

        for m, n, k in problem_sizes:
            for split_k in split_k_values:
                results = self.benchmark_problem_size(
                    kernels,
                    m,
                    n,
                    k,
                    split_k,
                    verify=2 if verify else 0,
                    warmup=warmup,
                    repeat=repeat,
                    flush_cache=flush_cache,
                    rotating_count=rotating_count,
                )

                all_results.extend(results)

                # Find best kernel for this configuration
                best = self.find_best_kernel(results)
                if best:
                    key = f"m{m}_n{n}_k{k}_splitk{split_k}"
                    best_kernels[key] = best
                    print(
                        f"Best for {key}: {best['name']} ({best['tflops']:.2f} TFLOPS, {best['bandwidth_gb_s']:.2f} GB/s, {best['time_ms']:.2f}ms)"
                    )

        self.results = all_results
        return best_kernels

    def export_csv(self, filename: str):
        """Export all results to CSV"""
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
        """Export best kernel selections to file"""
        with open(filename, "w") as f:
            f.write("# Best kernel selections\n")
            f.write(
                "# Format: problem_size -> kernel_name (TFLOPS, bandwidth, latency)\n\n"
            )

            for key, kernel in sorted(best_kernels.items()):
                f.write(
                    f"{key}: {kernel['name']} ({kernel['tflops']:.2f} TFLOPS, {kernel['bandwidth_gb_s']:.2f} GB/s, {kernel['time_ms']:.2f}ms)\n"
                )

        print(f"Best kernels exported to {filename}")

    def export_json(self, filename: str, best_kernels: Dict = None):
        """Export all results and best kernels to JSON with comprehensive metadata"""
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
            # Pipeline statistics
            pipeline = result.get("pipeline", "unknown")
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
            scheduler = result.get("scheduler", "unknown")
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
            data_type = result.get("data_type", "unknown")
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
        for stats in [pipeline_stats, scheduler_stats, data_type_stats]:
            for key in stats:
                relevant_results = [
                    r
                    for r in successful_results
                    if r.get(key.split("_")[0] if "_" in key else "pipeline", "unknown")
                    == key
                ]
                if relevant_results:
                    stats[key]["avg_tflops"] = sum(
                        r.get("tflops", 0) for r in relevant_results
                    ) / len(relevant_results)

        output_data = {
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
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
                    "average": sum(tflops_values) / len(tflops_values)
                    if tflops_values
                    else 0,
                    "min": min(tflops_values, default=0),
                    "median": sorted(tflops_values)[len(tflops_values) // 2]
                    if tflops_values
                    else 0,
                },
                "bandwidth_stats": {
                    "best_gb_s": max(bandwidth_values, default=0),
                    "average_gb_s": sum(bandwidth_values) / len(bandwidth_values)
                    if bandwidth_values
                    else 0,
                    "min_gb_s": min(bandwidth_values, default=0),
                    "median_gb_s": sorted(bandwidth_values)[len(bandwidth_values) // 2]
                    if bandwidth_values
                    else 0,
                },
                "latency_stats": {
                    "best_ms": min(latency_values, default=0),
                    "average_ms": sum(latency_values) / len(latency_values)
                    if latency_values
                    else 0,
                    "max_ms": max(latency_values, default=0),
                    "median_ms": sorted(latency_values)[len(latency_values) // 2]
                    if latency_values
                    else 0,
                },
                "kernel_type_breakdown": {
                    "by_pipeline": pipeline_stats,
                    "by_scheduler": scheduler_stats,
                    "by_data_type": data_type_stats,
                },
                "total_problem_configurations": len(best_kernels)
                if best_kernels
                else 0,
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
    parser = argparse.ArgumentParser(description="GEMM Kernel Benchmarking Tool")
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
        "--csv", default="gemm_benchmark_results.csv", help="CSV output filename"
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
    benchmark = GemmBenchmark(args.build_dir, verbose=args.verbose)

    # Run benchmark sweep
    print("Starting GEMM kernel benchmark sweep...")
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
    benchmark.export_csv(args.csv)
    benchmark.export_best_kernels(best_kernels, args.best)

    # Export JSON if requested
    if args.json:
        benchmark.export_json(args.json, best_kernels)

    return 0


if __name__ == "__main__":
    sys.exit(main())
