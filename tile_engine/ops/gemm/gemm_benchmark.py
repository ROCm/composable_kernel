#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple


# TODO: explore modularizing tile engine to avoid accessing imports like this
def _import_benchmark_utils():
    """Import benchmark utilities from commons directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(
        "benchmark_utils",
        os.path.join(parent_dir, "common", "benchmark_utils.py"),
    )
    benchmark_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark_utils)

    return benchmark_utils


benchmark_utils = _import_benchmark_utils()


class GemmBenchmark:
    def __init__(
        self, build_dir: str, verbose: bool = False, name: str = "benchmark_gemm_"
    ):
        self.build_dir = Path(build_dir)
        self.verbose = verbose
        self.results = []
        self.name = name

    def discover_kernels(self) -> List[Path]:
        """Find all benchmark_gemm_* executables in the build directory"""
        bin_dir = self.build_dir / "bin"
        if not bin_dir.exists():
            print(f"Error: Binary directory {bin_dir} does not exist")
            return []

        glob_name = f"{self.name}*"
        kernels = list(bin_dir.glob(glob_name))
        if self.verbose:
            print(f"Found {len(kernels)} kernel executables")
            for k in kernels:
                print(f"  - {k.name}")
        return kernels

    def extract_kernel_info(self, kernel_path: Path) -> Dict[str, str]:
        """Extract comprehensive kernel information from filename"""
        name = kernel_path.stem
        if name.startswith(self.name):
            args = name[len(self.name) :]
        else:
            args = name

        # Initialize with basic info
        info = {
            "executable": str(kernel_path),
            "name": name,
            "data_type": "unknown",
            "layout": "unknown",
            "pipeline": "unknown",
            "scheduler": "unknown",
            "epilogue": "unknown",
        }

        # Parse the kernel name pattern:
        # benchmark_gemm_fp16_rcr_mem_default_intrawave_False_False_False_False_False_256x256x32_2x2x1_4x64x16
        parts = args.split("_")

        if len(parts) >= 5:
            info["data_type"] = parts[0]
            info["layout"] = parts[1]
            info["pipeline"] = parts[2]
            info["epilogue"] = parts[3]
            info["scheduler"] = parts[4]

        # Extract detailed configuration from the end of the name
        config_info = self.parse_detailed_config(name)
        info.update(config_info)

        # Generate config ID
        info["config_id"] = self.generate_config_id(info)

        return info

    def parse_detailed_config(self, kernel_name: str) -> Dict:
        """Parse detailed configuration from kernel name"""
        config = {
            "tile_sizes": {"tile_m": 0, "tile_n": 0, "tile_k": 0},
            "warp_config": {"warp_m": 0, "warp_n": 0, "warp_k": 0},
            "warp_tile": {"warp_tile_m": 0, "warp_tile_n": 0, "warp_tile_k": 0},
            "optimization_flags": {
                "pad_m": False,
                "pad_n": False,
                "pad_k": False,
                "persistent": False,
            },
        }

        # Split by underscore and look for patterns
        parts = kernel_name.split("_")

        # Look for boolean flags (sequence of True/False values)
        bool_sequence = []
        for i, part in enumerate(parts):
            if part in ["True", "False"]:
                bool_sequence.append(part == "True")
                # Continue collecting consecutive boolean values
                j = i + 1
                while j < len(parts) and parts[j] in ["True", "False"]:
                    bool_sequence.append(parts[j] == "True")
                    j += 1
                break

        # Assign boolean flags if we found them
        # Order: pad_m, pad_n, pad_k, persistent (4 flags total)
        if len(bool_sequence) >= 4:
            config["optimization_flags"]["pad_m"] = bool_sequence[0]
            config["optimization_flags"]["pad_n"] = bool_sequence[1]
            config["optimization_flags"]["pad_k"] = bool_sequence[2]
            config["optimization_flags"]["persistent"] = bool_sequence[3]

        # Look for tile size patterns (e.g., 256x256x32_2x2x1_4x64x16)
        # The pattern is: tile_sizes_warp_config_warp_tile
        dimension_groups = []
        for part in parts:
            if "x" in part and len(part.split("x")) == 3:
                try:
                    dims = [int(x) for x in part.split("x")]
                    if all(d > 0 for d in dims):
                        dimension_groups.append(dims)
                except ValueError:
                    continue

        # Assign dimensions based on order and magnitude
        if len(dimension_groups) >= 3:
            # Sort by magnitude to identify: largest=tile_sizes, smallest=warp_config, middle=warp_tile
            sorted_groups = sorted(dimension_groups, key=lambda x: max(x), reverse=True)

            # Largest dimensions = tile sizes
            config["tile_sizes"]["tile_m"] = sorted_groups[0][0]
            config["tile_sizes"]["tile_n"] = sorted_groups[0][1]
            config["tile_sizes"]["tile_k"] = sorted_groups[0][2]

            # Smallest dimensions = warp config
            config["warp_config"]["warp_m"] = sorted_groups[2][0]
            config["warp_config"]["warp_n"] = sorted_groups[2][1]
            config["warp_config"]["warp_k"] = sorted_groups[2][2]

            # Middle dimensions = warp tile
            config["warp_tile"]["warp_tile_m"] = sorted_groups[1][0]
            config["warp_tile"]["warp_tile_n"] = sorted_groups[1][1]
            config["warp_tile"]["warp_tile_k"] = sorted_groups[1][2]
        elif len(dimension_groups) == 2:
            # If only 2 groups, assign based on magnitude
            sorted_groups = sorted(dimension_groups, key=lambda x: max(x), reverse=True)

            # Larger = tile sizes
            config["tile_sizes"]["tile_m"] = sorted_groups[0][0]
            config["tile_sizes"]["tile_n"] = sorted_groups[0][1]
            config["tile_sizes"]["tile_k"] = sorted_groups[0][2]

            # Smaller = warp config
            config["warp_config"]["warp_m"] = sorted_groups[1][0]
            config["warp_config"]["warp_n"] = sorted_groups[1][1]
            config["warp_config"]["warp_k"] = sorted_groups[1][2]
        elif len(dimension_groups) == 1:
            # Only one group - assume it's tile sizes
            config["tile_sizes"]["tile_m"] = dimension_groups[0][0]
            config["tile_sizes"]["tile_n"] = dimension_groups[0][1]
            config["tile_sizes"]["tile_k"] = dimension_groups[0][2]

        return config

    def generate_config_id(self, info: Dict) -> str:
        """Generate a compact config ID from kernel info"""
        # Create a compact identifier
        parts = [
            info.get("data_type", "unk"),
            info.get("layout", "unk"),
            info.get("pipeline", "unk"),
            info.get("scheduler", "unk"),
        ]

        # Add tile configuration if available
        tile_sizes = info.get("tile_sizes", {})
        if tile_sizes.get("tile_m", 0) > 0:
            tile_str = (
                f"{tile_sizes['tile_m']}x{tile_sizes['tile_n']}x{tile_sizes['tile_k']}"
            )
            parts.append(tile_str)

        # Add warp config if available
        warp_config = info.get("warp_config", {})
        if warp_config.get("warp_m", 0) > 0:
            warp_str = f"w{warp_config['warp_m']}x{warp_config['warp_n']}x{warp_config['warp_k']}"
            parts.append(warp_str)

        # Add warp tile if available
        warp_tile = info.get("warp_tile", {})
        if warp_tile.get("warp_tile_m", 0) > 0:
            warp_tile_str = f"wt{warp_tile['warp_tile_m']}x{warp_tile['warp_tile_n']}x{warp_tile['warp_tile_k']}"
            parts.append(warp_tile_str)

        return "_".join(parts)

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
            result = benchmark_utils.run_kernel(
                self.build_dir, kernel_path, params, verbose=self.verbose
            )
            if result:
                # Create new structured result format
                structured_result = {
                    "name": kernel_info["name"],  # Add name field for compatibility
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
                    # Keep backward compatibility fields
                    "time_ms": result.get("time_ms", 0),
                    "tflops": result.get("tflops", 0),
                    "bandwidth_gb_s": result.get("bandwidth_gb_s", 0),
                }

                results.append(structured_result)

                if self.verbose:
                    print(
                        f"  {kernel_info['config_id']}: {structured_result['tflops']:.2f} TFLOPS, {structured_result['bandwidth_gb_s']:.2f} GB/s, {structured_result['time_ms']:.2f}ms"
                    )

        return results

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
                best = benchmark_utils.find_best_kernel(results)
                if best:
                    key = f"m{m}_n{n}_k{k}_splitk{split_k}"
                    best_kernels[key] = best
                    print(
                        f"Best for {key}: {best['name']} ({best['tflops']:.2f} TFLOPS, {best['bandwidth_gb_s']:.2f} GB/s, {best['time_ms']:.2f}ms)"
                    )

        self.results = all_results
        return best_kernels
