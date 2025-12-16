#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
import json
import subprocess
import argparse
import csv
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class PoolBenchmark:
    def __init__(self, build_dir: str, verbose: bool = False):
        self.build_dir = Path(build_dir)
        self.verbose = verbose
        self.results = []

    def discover_kernels(self) -> List[Path]:
        """Find all benchmark_pool_* executables in the build directory"""
        bin_dir = self.build_dir / "bin"
        if not bin_dir.exists():
            print(f"Error: Binary directory {bin_dir} does not exist")
            return []

        kernels = list(bin_dir.glob("benchmark_pool*"))
        if self.verbose:
            print(f"Found {len(kernels)} kernel executables")
            for k in kernels:
                print(f"  - {k.name}")
        return kernels

    def extract_kernel_info(self, kernel_path: Path) -> Dict[str, str]:
        """Extract comprehensive kernel information from filename"""
        name = kernel_path.stem

        # Initialize with basic info
        info = {
            "executable": str(kernel_path),
            "name": name,
            "data_type": "unknown",
            "reduce_op": "unknown",
            "pool_dim": 0,
            "output_index": False,
            "propagate_nan": False,
        }

        # Parse the kernel name pattern:
        # benchmark_pool3d_fp16_max_True_False_128x1_1x1_2x1
        parts = name.split("_")

        if len(parts) >= 3:
            # Extract pool dimension (e.g., pool3d -> 3)
            if "pool2d" in parts[1]:
                info["pool_dim"] = 2
            elif "pool3d" in parts[1]:
                info["pool_dim"] = 3

            # Extract data type
            info["data_type"] = parts[2] if len(parts) > 2 else "unknown"

            # Extract reduce op
            info["reduce_op"] = parts[3] if len(parts) > 3 else "unknown"

            # Extract flags
            if len(parts) > 4:
                info["output_index"] = parts[4] == "True"
            if len(parts) > 5:
                info["propagate_nan"] = parts[5] == "True"

        # Extract block configuration
        config_info = self.parse_block_config(name)
        info.update(config_info)

        # Generate config ID
        info["config_id"] = self.generate_config_id(info)

        return info

    def parse_block_config(self, kernel_name: str) -> Dict:
        """Parse block configuration from kernel name"""
        config = {
            "block_sizes": {"block_m": 0, "block_n": 0},
            "warp_config": {"warp_m": 0, "warp_n": 0},
            "thread_tile": {"thread_tile_m": 0, "thread_tile_n": 0},
        }

        parts = kernel_name.split("_")

        # Look for dimension patterns (e.g., 128x1)
        dimension_groups = []
        for part in parts:
            if "x" in part and len(part.split("x")) == 2:
                try:
                    dims = [int(x) for x in part.split("x")]
                    if all(d >= 0 for d in dims):
                        dimension_groups.append(dims)
                except ValueError:
                    continue

        # Assign dimensions based on order
        if len(dimension_groups) >= 3:
            config["block_sizes"]["block_m"] = dimension_groups[0][0]
            config["block_sizes"]["block_n"] = dimension_groups[0][1]
            config["warp_config"]["warp_m"] = dimension_groups[1][0]
            config["warp_config"]["warp_n"] = dimension_groups[1][1]
            config["thread_tile"]["thread_tile_m"] = dimension_groups[2][0]
            config["thread_tile"]["thread_tile_n"] = dimension_groups[2][1]
        elif len(dimension_groups) == 2:
            config["block_sizes"]["block_m"] = dimension_groups[0][0]
            config["block_sizes"]["block_n"] = dimension_groups[0][1]
            config["warp_config"]["warp_m"] = dimension_groups[1][0]
            config["warp_config"]["warp_n"] = dimension_groups[1][1]
        elif len(dimension_groups) == 1:
            config["block_sizes"]["block_m"] = dimension_groups[0][0]
            config["block_sizes"]["block_n"] = dimension_groups[0][1]

        return config

    def generate_config_id(self, info: Dict) -> str:
        """Generate a compact config ID from kernel info"""
        parts = [
            f"pool{info.get('pool_dim', 0)}d",
            info.get("data_type", "unk"),
            info.get("reduce_op", "unk"),
        ]

        block_sizes = info.get("block_sizes", {})
        if block_sizes.get("block_m", 0) > 0:
            block_str = f"{block_sizes['block_m']}x{block_sizes['block_n']}"
            parts.append(block_str)

        return "_".join(parts)

    def run_kernel(self, kernel_path: Path, params: Dict[str, str]) -> Optional[Dict]:
        """Run a single kernel with given parameters"""
        results_dir = self.build_dir / "results"
        results_dir.mkdir(exist_ok=True)

        json_file = results_dir / f"{kernel_path.stem}.json"

        cmd = [str(kernel_path)]

        for key, value in params.items():
            cmd.append(f"-{key}={value}")

        cmd.append("-json_output=true")

        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                print(f"Error running {kernel_path.name}: {result.stderr}")
                return None

            output = result.stdout.strip()
            if output:
                with open(json_file, "w") as f:
                    f.write(output)

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

            data = json.loads(content)

            result = data.copy()
            if "perf_result" in data:
                perf = data["perf_result"]
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

    def benchmark_problem_size(
        self,
        kernels: List[Path],
        N: int,
        D: int,
        H: int,
        W: int,
        C: int,
        window_z: int = 2,
        window_y: int = 2,
        window_x: int = 2,
        stride_z: int = 2,
        stride_y: int = 2,
        stride_x: int = 2,
        pool_dim: int = 3,
        verify: int = 0,
        warmup: int = 20,
        repeat: int = 100,
        flush_cache: bool = True,
        rotating_count: int = 1000,
    ) -> List[Dict]:
        """Benchmark all kernels for a specific problem size"""
        results = []

        params = {
            "N": N,
            "D": D,
            "H": H,
            "W": W,
            "C": C,
            "Z": window_z,
            "Y": window_y,
            "X": window_x,
            "Sz": stride_z,
            "Sy": stride_y,
            "Sx": stride_x,
            "pool_dim": pool_dim,
            "verify": verify,
            "warmup": warmup,
            "repeat": repeat,
            "flush_cache": str(flush_cache).lower(),
            "rotating_count": rotating_count,
        }

        print(f"\nBenchmarking N={N}, D={D}, H={H}, W={W}, C={C}")
        print(
            f"  Window: {window_z}x{window_y}x{window_x}, Stride: {stride_z}x{stride_y}x{stride_x}"
        )

        for kernel_path in kernels:
            kernel_info = self.extract_kernel_info(kernel_path)
            result = self.run_kernel(kernel_path, params)

            if result:
                structured_result = {
                    "name": kernel_info["name"],
                    "config_id": kernel_info["config_id"],
                    "problem": result.get("problem", {}),
                    "perf_result": result.get("perf_result", {}),
                    "config": {
                        "data_type": kernel_info["data_type"],
                        "reduce_op": kernel_info["reduce_op"],
                        "pool_dim": kernel_info["pool_dim"],
                        "output_index": kernel_info["output_index"],
                        "propagate_nan": kernel_info["propagate_nan"],
                        "block_sizes": kernel_info.get("block_sizes", {}),
                        "warp_config": kernel_info.get("warp_config", {}),
                        "thread_tile": kernel_info.get("thread_tile", {}),
                    },
                    "executable": kernel_info["executable"],
                    "time_ms": result.get("time_ms", 0),
                    "tflops": result.get("tflops", 0),
                    "bandwidth_gb_s": result.get("bandwidth_gb_s", 0),
                }

                results.append(structured_result)

                if self.verbose:
                    print(
                        f"  {kernel_info['config_id']}: {structured_result['bandwidth_gb_s']:.2f} GB/s, {structured_result['time_ms']:.2f}ms"
                    )

        return results

    def find_best_kernel(
        self, results: List[Dict], metric: str = "bandwidth_gb_s"
    ) -> Optional[Dict]:
        """Find the best performing kernel based on metric"""
        if not results:
            return None

        if metric == "bandwidth_gb_s":
            return max(results, key=lambda x: x.get("bandwidth_gb_s", 0))
        elif metric == "time_ms":
            return min(results, key=lambda x: x.get("time_ms", float("inf")))
        elif metric == "tflops":
            return max(results, key=lambda x: x.get("tflops", 0))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def benchmark_sweep(
        self,
        problem_sizes: List[Tuple[int, int, int, int, int]],  # N, D, H, W, C
        window_sizes: List[Tuple[int, int, int]] = [(2, 2, 2)],
        stride_sizes: List[Tuple[int, int, int]] = [(2, 2, 2)],
        pool_dim: int = 3,
        verify: bool = False,
        warmup: int = 20,
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

        for N, D, H, W, C in problem_sizes:
            for wz, wy, wx in window_sizes:
                for sz, sy, sx in stride_sizes:
                    results = self.benchmark_problem_size(
                        kernels,
                        N,
                        D,
                        H,
                        W,
                        C,
                        window_z=wz,
                        window_y=wy,
                        window_x=wx,
                        stride_z=sz,
                        stride_y=sy,
                        stride_x=sx,
                        pool_dim=pool_dim,
                        verify=1 if verify else 0,
                        warmup=warmup,
                        repeat=repeat,
                        flush_cache=flush_cache,
                        rotating_count=rotating_count,
                    )

                    all_results.extend(results)

                    best = self.find_best_kernel(results)
                    if best:
                        key = (
                            f"N{N}_D{D}_H{H}_W{W}_C{C}_w{wz}x{wy}x{wx}_s{sz}x{sy}x{sx}"
                        )
                        best_kernels[key] = best
                        print(
                            f"Best for {key}: {best['name']} ({best['bandwidth_gb_s']:.2f} GB/s, {best['time_ms']:.2f}ms)"
                        )

        self.results = all_results
        return best_kernels

    def export_csv(self, filename: str):
        """Export all results to CSV"""
        if not self.results:
            print("No results to export")
            return

        all_keys = set()
        for result in self.results:
            all_keys.update(result.keys())

        fieldnames = sorted(all_keys)

        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Results exported to {filename}")

    def export_best_kernels(self, best_kernels: Dict, filename: str):
        """Export best kernel selections to file"""
        with open(filename, "w") as f:
            f.write("# Best kernel selections for pooling\n")
            f.write("# Format: problem_size -> kernel_name (bandwidth, latency)\n\n")

            for key, kernel in sorted(best_kernels.items()):
                f.write(
                    f"{key}: {kernel['name']} ({kernel['bandwidth_gb_s']:.2f} GB/s, {kernel['time_ms']:.2f}ms)\n"
                )

        print(f"Best kernels exported to {filename}")

    def export_json(self, filename: str, best_kernels: Dict = None):
        """Export all results and best kernels to JSON"""
        from datetime import datetime

        successful_results = [r for r in self.results if r.get("bandwidth_gb_s", 0) > 0]

        bandwidth_values = [r.get("bandwidth_gb_s", 0) for r in successful_results]
        latency_values = [
            r.get("time_ms", 0) for r in successful_results if r.get("time_ms", 0) > 0
        ]

        # Performance breakdown by kernel type
        reduce_op_stats = {}
        data_type_stats = {}

        for result in successful_results:
            config = result.get("config", {})

            reduce_op = config.get("reduce_op", "unknown")
            if reduce_op not in reduce_op_stats:
                reduce_op_stats[reduce_op] = {
                    "count": 0,
                    "avg_bandwidth": 0,
                    "best_bandwidth": 0,
                }
            reduce_op_stats[reduce_op]["count"] += 1
            reduce_op_stats[reduce_op]["best_bandwidth"] = max(
                reduce_op_stats[reduce_op]["best_bandwidth"],
                result.get("bandwidth_gb_s", 0),
            )

            data_type = config.get("data_type", "unknown")
            if data_type not in data_type_stats:
                data_type_stats[data_type] = {
                    "count": 0,
                    "avg_bandwidth": 0,
                    "best_bandwidth": 0,
                }
            data_type_stats[data_type]["count"] += 1
            data_type_stats[data_type]["best_bandwidth"] = max(
                data_type_stats[data_type]["best_bandwidth"],
                result.get("bandwidth_gb_s", 0),
            )

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
                "bandwidth_stats": {
                    "best_gb_s": max(bandwidth_values, default=0),
                    "average_gb_s": sum(bandwidth_values) / len(bandwidth_values)
                    if bandwidth_values
                    else 0,
                    "min_gb_s": min(bandwidth_values, default=0),
                },
                "latency_stats": {
                    "best_ms": min(latency_values, default=0),
                    "average_ms": sum(latency_values) / len(latency_values)
                    if latency_values
                    else 0,
                    "max_ms": max(latency_values, default=0),
                },
                "kernel_type_breakdown": {
                    "by_reduce_op": reduce_op_stats,
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
        print(f"  - Best bandwidth: {max(bandwidth_values, default=0):.2f} GB/s")
        print(f"  - Best latency: {min(latency_values, default=0):.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Pool Kernel Benchmarking Tool")
    parser.add_argument(
        "build_dir", help="Build directory containing kernel executables"
    )
    parser.add_argument(
        "--problem-sizes",
        nargs="+",
        default=["2,30,30,30,32", "4,64,64,64,64", "8,128,128,128,128"],
        help="Problem sizes as N,D,H,W,C tuples",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        default=["2,2,2", "3,3,3"],
        help="Window sizes as Z,Y,X tuples",
    )
    parser.add_argument(
        "--stride-sizes",
        nargs="+",
        default=["2,2,2"],
        help="Stride sizes as Z,Y,X tuples",
    )
    parser.add_argument(
        "--pool-dim", type=int, default=3, help="Pooling dimension (2 or 3)"
    )
    parser.add_argument("--verify", action="store_true", help="Enable verification")
    parser.add_argument(
        "--csv", default="pool_benchmark_results.csv", help="CSV output filename"
    )
    parser.add_argument(
        "--best", default="best_pool_kernels.txt", help="Best kernels output filename"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations (default: 20)",
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
            parts = list(map(int, size_str.split(",")))
            if len(parts) == 5:
                problem_sizes.append(tuple(parts))
            else:
                print(f"Invalid problem size: {size_str} (expected N,D,H,W,C)")
                return 1
        except ValueError:
            print(f"Invalid problem size: {size_str}")
            return 1

    # Parse window sizes
    window_sizes = []
    for size_str in args.window_sizes:
        try:
            parts = list(map(int, size_str.split(",")))
            if len(parts) == 3:
                window_sizes.append(tuple(parts))
            else:
                print(f"Invalid window size: {size_str} (expected Z,Y,X)")
                return 1
        except ValueError:
            print(f"Invalid window size: {size_str}")
            return 1

    # Parse stride sizes
    stride_sizes = []
    for size_str in args.stride_sizes:
        try:
            parts = list(map(int, size_str.split(",")))
            if len(parts) == 3:
                stride_sizes.append(tuple(parts))
            else:
                print(f"Invalid stride size: {size_str} (expected Z,Y,X)")
                return 1
        except ValueError:
            print(f"Invalid stride size: {size_str}")
            return 1

    # Create benchmark instance
    benchmark = PoolBenchmark(args.build_dir, verbose=args.verbose)

    # Run benchmark sweep
    print("Starting Pool kernel benchmark sweep...")
    start_time = time.time()

    best_kernels = benchmark.benchmark_sweep(
        problem_sizes=problem_sizes,
        window_sizes=window_sizes,
        stride_sizes=stride_sizes,
        pool_dim=args.pool_dim,
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

    if args.json:
        benchmark.export_json(args.json, best_kernels)

    return 0


if __name__ == "__main__":
    sys.exit(main())
