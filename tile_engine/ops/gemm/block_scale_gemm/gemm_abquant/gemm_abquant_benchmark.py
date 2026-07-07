# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
import json
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class ABQuantGemmBenchmark:
    def __init__(self, build_dir: str, verbose: bool = False):
        self.build_dir = Path(build_dir)
        self.verbose = verbose
        self.results = []

    def discover_kernels(self) -> List[Path]:
        """Find all benchmark_gemm_abquant_* executables in the build directory."""
        bin_dir = self.build_dir / "bin"
        if not bin_dir.exists():
            print(f"Error: Binary directory {bin_dir} does not exist")
            return []

        kernels = list(bin_dir.glob("benchmark_gemm_abquant_*"))
        if self.verbose:
            print(f"Found {len(kernels)} ABQuant kernel executables")
            for k in kernels:
                print(f"  - {k.name}")
        return kernels

    def extract_kernel_info(self, kernel_path: Path) -> Dict[str, str]:
        """Extract kernel information from filename."""
        name = kernel_path.stem

        info = {
            "executable": str(kernel_path),
            "name": name,
            "data_type": "unknown",
            "layout": "unknown",
            "pipeline": "unknown",
            "scheduler": "unknown",
            "epilogue": "unknown",
            "a_preshuffle_quant": False,
            "b_preshuffle_quant": False,
            "group_size_n": 1,
        }

        # Parse: benchmark_gemm_abquant_fp8_rcr_compv3_default_intrawave_False_False_False_False_False_gsn1_128x128x128_...
        parts = name.split("_")

        if len(parts) >= 3:
            # Skip "benchmark_gemm_abquant" prefix (3 parts)
            idx = 3
            if idx < len(parts):
                info["data_type"] = parts[idx]
            idx += 1
            if idx < len(parts):
                info["layout"] = parts[idx]
            idx += 1
            if idx < len(parts):
                info["pipeline"] = parts[idx]
            idx += 1
            if idx < len(parts):
                info["epilogue"] = parts[idx]
            idx += 1
            if idx < len(parts):
                info["scheduler"] = parts[idx]
            idx += 1
            # Skip pad_m, pad_n, pad_k
            idx += 3
            if idx < len(parts):
                info["a_preshuffle_quant"] = parts[idx] == "True"
            idx += 1
            if idx < len(parts):
                info["b_preshuffle_quant"] = parts[idx] == "True"
            idx += 1
            # Parse gsn value (e.g. gsn1, gsn128)
            if idx < len(parts) and parts[idx].startswith("gsn"):
                info["group_size_n"] = int(parts[idx][3:])

        config_info = self.parse_detailed_config(name)
        info.update(config_info)

        # Warn on fields that failed to parse — silent "unknown"/zero values hide name format drift.
        unknown_fields = [k for k in ("data_type", "layout", "pipeline", "scheduler", "epilogue")
                          if info.get(k) == "unknown"]
        if unknown_fields:
            print(f"Warning: could not parse {unknown_fields} from kernel name: {name}")
        tile_sizes = info.get("tile_sizes", {})
        if any(v == 0 for v in tile_sizes.values()):
            print(f"Warning: zero tile dimension parsed from kernel name: {name} -> {tile_sizes}")

        info["config_id"] = self.generate_config_id(info)
        return info

    def parse_detailed_config(self, kernel_name: str) -> Dict:
        """Parse tile dimensions and boolean flags from kernel name."""
        config = {
            "tile_sizes": {"tile_m": 0, "tile_n": 0, "tile_k": 0},
            "warp_config": {"warp_m": 0, "warp_n": 0, "warp_k": 0},
            "warp_tile": {"warp_tile_m": 0, "warp_tile_n": 0, "warp_tile_k": 0},
            "optimization_flags": {
                "pad_m": False,
                "pad_n": False,
                "pad_k": False,
                "a_preshuffle_quant": False,
                "b_preshuffle_quant": False,
            },
        }

        parts = kernel_name.split("_")

        # Extract boolean flags (5 consecutive True/False values)
        bool_sequence = []
        for i, part in enumerate(parts):
            if part in ["True", "False"]:
                bool_sequence.append(part == "True")
                j = i + 1
                while j < len(parts) and parts[j] in ["True", "False"]:
                    bool_sequence.append(parts[j] == "True")
                    j += 1
                break

        if len(bool_sequence) >= 5:
            config["optimization_flags"]["pad_m"] = bool_sequence[0]
            config["optimization_flags"]["pad_n"] = bool_sequence[1]
            config["optimization_flags"]["pad_k"] = bool_sequence[2]
            config["optimization_flags"]["a_preshuffle_quant"] = bool_sequence[3]
            config["optimization_flags"]["b_preshuffle_quant"] = bool_sequence[4]

        # Extract dimension groups (e.g., 128x128x128) in positional order.
        # Kernel names encode groups as: tile_MxNxK_warp_MxNxK_warp_tile_MxNxK
        dimension_groups = []
        for part in parts:
            if "x" in part and len(part.split("x")) == 3:
                try:
                    dims = [int(x) for x in part.split("x")]
                    if all(d > 0 for d in dims):
                        dimension_groups.append(dims)
                except ValueError:
                    continue

        if len(dimension_groups) >= 3:
            config["tile_sizes"]["tile_m"] = dimension_groups[0][0]
            config["tile_sizes"]["tile_n"] = dimension_groups[0][1]
            config["tile_sizes"]["tile_k"] = dimension_groups[0][2]
            config["warp_config"]["warp_m"] = dimension_groups[1][0]
            config["warp_config"]["warp_n"] = dimension_groups[1][1]
            config["warp_config"]["warp_k"] = dimension_groups[1][2]
            config["warp_tile"]["warp_tile_m"] = dimension_groups[2][0]
            config["warp_tile"]["warp_tile_n"] = dimension_groups[2][1]
            config["warp_tile"]["warp_tile_k"] = dimension_groups[2][2]

        return config

    def generate_config_id(self, info: Dict) -> str:
        """Generate a compact config ID from kernel info."""
        parts = [
            info.get("data_type", "unk"),
            info.get("layout", "unk"),
            info.get("pipeline", "unk"),
            info.get("scheduler", "unk"),
        ]

        tile_sizes = info.get("tile_sizes", {})
        if tile_sizes.get("tile_m", 0) > 0:
            parts.append(
                f"{tile_sizes['tile_m']}x{tile_sizes['tile_n']}x{tile_sizes['tile_k']}"
            )

        gsn = info.get("group_size_n", 1)
        parts.append(f"gsn{gsn}")

        return "_".join(parts)

    def run_kernel(self, kernel_path: Path, params: Dict[str, str]) -> Optional[Dict]:
        """Run a single kernel with given parameters."""
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

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
        """Parse JSON data from kernel output file."""
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

        except (json.JSONDecodeError, Exception) as e:
            if self.verbose:
                print(f"Failed to parse JSON from {json_file}: {e}")
            return None

    def benchmark_problem_size(
        self,
        kernels: List[Path],
        m: int,
        n: int,
        k: int,
        group_size_k: int = 128,
        verify: int = 0,
        warmup: int = 50,
        repeat: int = 100,
        flush_cache: bool = True,
        rotating_count: int = 1000,
    ) -> List[Dict]:
        """Benchmark all kernels for a specific problem size."""
        results = []

        print(f"\nBenchmarking M={m}, N={n}, K={k}, group_size_k={group_size_k}")

        for kernel_path in kernels:
            kernel_info = self.extract_kernel_info(kernel_path)
            group_size_n = kernel_info.get("group_size_n", 1)

            params = {
                "m": m,
                "n": n,
                "k": k,
                "group_size_k": group_size_k,
                "group_size_n": group_size_n,
                "verify": verify,
                "warmup": warmup,
                "repeat": repeat,
                "flush_cache": str(flush_cache).lower(),
                "rotating_count": rotating_count,
            }

            result = self.run_kernel(kernel_path, params)

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
                        f"  {kernel_info['config_id']}: "
                        f"{structured_result['tflops']:.2f} TFLOPS, "
                        f"{structured_result['bandwidth_gb_s']:.2f} GB/s, "
                        f"{structured_result['time_ms']:.2f}ms"
                    )

        return results

    def find_best_kernel(
        self, results: List[Dict], metric: str = "tflops"
    ) -> Optional[Dict]:
        """Find the best performing kernel based on metric."""
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
        group_size_k: int = 128,
        verify: bool = False,
        warmup: int = 50,
        repeat: int = 100,
        flush_cache: bool = True,
        rotating_count: int = 1000,
    ) -> Dict:
        """Run comprehensive benchmark sweep."""
        kernels = self.discover_kernels()
        if not kernels:
            print("No kernels found!")
            return {}

        all_results = []
        best_kernels = {}

        for m, n, k in problem_sizes:
            results = self.benchmark_problem_size(
                kernels,
                m,
                n,
                k,
                group_size_k=group_size_k,
                verify=1 if verify else 0,
                warmup=warmup,
                repeat=repeat,
                flush_cache=flush_cache,
                rotating_count=rotating_count,
            )

            all_results.extend(results)

            best = self.find_best_kernel(results)
            if best:
                key = f"m{m}_n{n}_k{k}"
                best_kernels[key] = best
                print(
                    f"Best for {key}: {best['name']} "
                    f"({best['tflops']:.2f} TFLOPS, "
                    f"{best['bandwidth_gb_s']:.2f} GB/s, "
                    f"{best['time_ms']:.2f}ms)"
                )

        self.results = all_results
        return best_kernels

    def export_json(self, filename: str, best_kernels: Dict = None):
        """Export results to JSON."""
        from datetime import datetime

        output_data = {
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
                "operator": "gemm_abquant",
                "total_kernels_tested": len(self.results),
                "successful_runs": len(
                    [r for r in self.results if r.get("tflops", 0) > 0]
                ),
            },
            "kernel_results": self.results,
            "best_kernels_by_problem": best_kernels or {},
        }

        with open(filename, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"JSON results exported to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="ABQuant GEMM Kernel Benchmarking Tool"
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
        "--group-size-k",
        type=int,
        default=128,
        help="Quantization group size along K (default: 128)",
    )
    parser.add_argument("--verify", action="store_true", help="Enable verification")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--no-flush-cache",
        action="store_true",
        help="Disable cache flushing between iterations",
    )
    parser.add_argument(
        "--rotating-count",
        type=int,
        default=1000,
        help="Number of iterations to rotate the cache (default: 1000)",
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

    benchmark = ABQuantGemmBenchmark(args.build_dir, verbose=args.verbose)

    print("Starting ABQuant GEMM kernel benchmark sweep...")
    start_time = time.time()

    best_kernels = benchmark.benchmark_sweep(
        problem_sizes=problem_sizes,
        group_size_k=args.group_size_k,
        verify=args.verify,
        warmup=args.warmup,
        repeat=args.repeat,
        flush_cache=not args.no_flush_cache,
        rotating_count=args.rotating_count,
    )

    elapsed_time = time.time() - start_time
    print(f"\nBenchmark completed in {elapsed_time:.2f} seconds")

    if args.json:
        benchmark.export_json(args.json, best_kernels)

    return 0


if __name__ == "__main__":
    sys.exit(main())
