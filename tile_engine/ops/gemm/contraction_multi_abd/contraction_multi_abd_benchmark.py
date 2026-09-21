# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchmark driver for the contraction_multi_abd tile_engine operator.

Modelled on batched_contraction_benchmark.py -- same CLI shape
(``--problem-configs g=..;m=..;n=..;k=..``), same ``bin/`` discovery, same
exit-code contract. Two things differ, both forced by the executable:

1. ``contraction_multi_abd_benchmark_single.cpp`` has NO ``-json_output``
   option (its ``create_args`` inserts only g/m/n/k dims, verify, warmup,
   repeat, timer, log). So results are parsed from the single stdout line
   emitted by contraction_multi_abd_benchmark.hpp instead of from JSON.
2. It has no ``split_k``, ``flush_cache`` or ``rotating_count`` argument;
   passing them would be rejected, so they are absent here too.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# CMakeLists names each target benchmark_contraction_multi_abd_<kernel_name>,
# and <kernel_name> as emitted by unified_contraction_multi_abd_codegen.py
# --list-name ALREADY starts with contraction_multi_abd_:
#   contraction_multi_abd_fp16_rcr_compv3_cshuffle_intrawave_
#   False_False_False_False_256x256x64_2x2x1_32x32x16_
#   na1_nb1_nd1_g1_m2_n2_k1_ewPassThrough_PassThrough_MultiDAdd
# so the executable stem carries the op name twice. Strip it repeatedly rather
# than once, or every field below shifts by three tokens and the report reads
# data_type="contraction", layout="multi".
EXE_GLOB = "benchmark_contraction_multi_abd_*"
_OP_TOKEN = "contraction_multi_abd_"


def _strip_exe_prefix(stem: str) -> str:
    """Reduce an executable stem to just the kernel configuration tokens."""
    name = stem[len("benchmark_"):] if stem.startswith("benchmark_") else stem
    while name.startswith(_OP_TOKEN):
        name = name[len(_OP_TOKEN):]
    return name

# The only machine-readable output the executable produces
# (contraction_multi_abd_benchmark.hpp):
#   kernel: NAME  latency(ms): 0.1234  tflops: 12.3456  bandwidth(GB/s): 123.4567
PERF_RE = re.compile(
    r"latency\(ms\):\s*([0-9.eE+-]+)\s+"
    r"tflops:\s*([0-9.eE+-]+)\s+"
    r"bandwidth\(GB/s\):\s*([0-9.eE+-]+)"
)

# g1_m2_n2_k1 in the kernel name records the dim COUNTS the kernel was compiled
# for. The executable hard-fails (EXIT_FAILURE) when the supplied dim list has a
# different length, so checking here turns a wall of opaque per-kernel failures
# into one actionable message.
DIMCOUNT_RE = re.compile(r"_g(\d+)_m(\d+)_n(\d+)_k(\d+)_")

TENSORCOUNT_RE = re.compile(r"_na(\d+)_nb(\d+)_nd(\d+)_")

# A 3-token AxBxC group; they appear in codegen order: tile, warp, warp_tile.
XDIM_RE = re.compile(r"^(\d+)x(\d+)x(\d+)$")


class ContractionMultiAbdBenchmark:
    def __init__(self, build_dir: str, verbose: bool = False):
        self.build_dir = Path(build_dir)
        self.verbose = verbose
        self.results = []
        self.launch_attempted = 0
        self.launch_failed = 0
        # Kernels never launched because their compiled dim arity did not match
        # the requested config. Counted separately so an all-skipped run reports
        # the real cause instead of "no kernels discovered".
        self.dim_mismatch_skipped = 0

    def discover_kernels(self) -> List[Path]:
        """Find all benchmark_contraction_multi_abd_* executables"""
        bin_dir = self.build_dir / "bin"
        if not bin_dir.exists():
            print(f"Error: Binary directory {bin_dir} does not exist")
            return []

        kernels = sorted(bin_dir.glob(EXE_GLOB))
        if self.verbose:
            print(f"Found {len(kernels)} kernel executables")
            for k in kernels:
                print(f"  - {k.name}")
        return kernels

    def extract_kernel_info(self, kernel_path: Path) -> Dict:
        """Extract kernel configuration from the executable filename.

        Everything here is best-effort metadata for the report; a name that does
        not parse still gets benchmarked, it just carries "unknown" fields.
        """
        name = kernel_path.stem
        stripped = _strip_exe_prefix(name)
        parts = stripped.split("_")

        info = {
            "executable": str(kernel_path),
            "name": name,
            "data_type": parts[0] if len(parts) > 0 else "unknown",
            "layout": parts[1] if len(parts) > 1 else "unknown",
            "pipeline": parts[2] if len(parts) > 2 else "unknown",
            "epilogue": parts[3] if len(parts) > 3 else "unknown",
            "scheduler": parts[4] if len(parts) > 4 else "unknown",
        }
        info.update(self.parse_detailed_config(stripped))
        info["config_id"] = self.generate_config_id(info)
        return info

    def parse_detailed_config(self, stripped_name: str) -> Dict:
        """Parse tile shape, flags, tensor counts and dim counts from the name."""
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
            "tensor_counts": {"num_a": 0, "num_b": 0, "num_d": 0},
            "dim_counts": {},
        }

        parts = stripped_name.split("_")

        # The four booleans are emitted as one contiguous run: pad_m, pad_n,
        # pad_k, persistent.
        bool_sequence = []
        for i, part in enumerate(parts):
            if part in ("True", "False"):
                j = i
                while j < len(parts) and parts[j] in ("True", "False"):
                    bool_sequence.append(parts[j] == "True")
                    j += 1
                break
        if len(bool_sequence) >= 4:
            config["optimization_flags"]["pad_m"] = bool_sequence[0]
            config["optimization_flags"]["pad_n"] = bool_sequence[1]
            config["optimization_flags"]["pad_k"] = bool_sequence[2]
            config["optimization_flags"]["persistent"] = bool_sequence[3]

        # Positional, not sorted-by-magnitude: codegen emits tile, then warp,
        # then warp_tile, and a small tile (e.g. 16x16x32) would defeat any
        # size-based ordering.
        groups = []
        for part in parts:
            m = XDIM_RE.match(part)
            if m:
                groups.append([int(x) for x in m.groups()])
        if len(groups) >= 3:
            config["tile_sizes"] = dict(
                zip(("tile_m", "tile_n", "tile_k"), groups[0])
            )
            config["warp_config"] = dict(
                zip(("warp_m", "warp_n", "warp_k"), groups[1])
            )
            config["warp_tile"] = dict(
                zip(("warp_tile_m", "warp_tile_n", "warp_tile_k"), groups[2])
            )

        # Bracket with underscores so the regexes cannot straddle a token
        # boundary at either end of the name.
        padded = f"_{stripped_name}_"

        tc = TENSORCOUNT_RE.search(padded)
        if tc:
            config["tensor_counts"] = {
                "num_a": int(tc.group(1)),
                "num_b": int(tc.group(2)),
                "num_d": int(tc.group(3)),
            }

        dc = DIMCOUNT_RE.search(padded)
        if dc:
            config["dim_counts"] = {
                "g": int(dc.group(1)),
                "m": int(dc.group(2)),
                "n": int(dc.group(3)),
                "k": int(dc.group(4)),
            }

        return config

    def generate_config_id(self, info: Dict) -> str:
        """Generate a compact config ID from kernel info"""
        parts = [
            info.get("data_type", "unk"),
            info.get("layout", "unk"),
            info.get("pipeline", "unk"),
            info.get("scheduler", "unk"),
        ]

        tile = info.get("tile_sizes", {})
        if tile.get("tile_m", 0) > 0:
            parts.append(f"{tile['tile_m']}x{tile['tile_n']}x{tile['tile_k']}")

        warp = info.get("warp_config", {})
        if warp.get("warp_m", 0) > 0:
            parts.append(f"w{warp['warp_m']}x{warp['warp_n']}x{warp['warp_k']}")

        wt = info.get("warp_tile", {})
        if wt.get("warp_tile_m", 0) > 0:
            parts.append(
                f"wt{wt['warp_tile_m']}x{wt['warp_tile_n']}x{wt['warp_tile_k']}"
            )

        dims = info.get("dim_counts", {})
        if dims:
            parts.append(f"g{dims['g']}m{dims['m']}n{dims['n']}k{dims['k']}")

        return "_".join(parts)

    def run_kernel(self, kernel_path: Path, params: Dict) -> Optional[Dict]:
        """Run one kernel and parse its perf line out of stdout."""
        cmd = [str(kernel_path)]
        for key, value in params.items():
            cmd.append(f"-{key}={value}")

        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            print(f"Timeout running {kernel_path.name}")
            return None
        except Exception as e:  # noqa: BLE001
            print(f"Error running {kernel_path.name}: {e}")
            return None

        if result.returncode != 0:
            # The executable exits 1 both for a dim-count/extent mismatch and for
            # the kernel's own unsupported-arguments signal; stderr says which.
            print(
                f"Error running {kernel_path.name} "
                f"(exit {result.returncode}): {result.stderr.strip()}"
            )
            return None

        match = PERF_RE.search(result.stdout)
        if not match:
            print(f"No parseable perf line from {kernel_path.name}")
            if self.verbose:
                print(f"  stdout was: {result.stdout.strip()!r}")
            return None

        return {
            "time_ms": float(match.group(1)),
            "tflops": float(match.group(2)),
            "bandwidth_gb_s": float(match.group(3)),
        }

    def benchmark_problem_size(
        self,
        kernels: List[Path],
        g_dims: str,
        m_dims: str,
        n_dims: str,
        k_dims: str,
        verify: int = 0,
        warmup: int = 50,
        repeat: int = 100,
    ) -> List[Dict]:
        """Benchmark all kernels for one problem configuration"""
        results = []

        params = {
            "g_dims": g_dims,
            "m_dims": m_dims,
            "n_dims": n_dims,
            "k_dims": k_dims,
            "verify": verify,
            "warmup": warmup,
            "repeat": repeat,
        }

        requested_counts = {
            "g": len(g_dims.split(",")),
            "m": len(m_dims.split(",")),
            "n": len(n_dims.split(",")),
            "k": len(k_dims.split(",")),
        }

        print(f"\nBenchmarking G={g_dims}, M={m_dims}, N={n_dims}, K={k_dims}")

        for kernel_path in kernels:
            kernel_info = self.extract_kernel_info(kernel_path)

            # A kernel is compiled for fixed NumDimsG/M/N/K. Supplying a
            # different-length dim list is a guaranteed EXIT_FAILURE, so say so
            # once, clearly, rather than letting every kernel fail opaquely.
            compiled = kernel_info.get("dim_counts") or {}
            if compiled and compiled != requested_counts:
                print(
                    f"  SKIP {kernel_info['name']}: compiled for "
                    f"g/m/n/k dim counts "
                    f"{compiled['g']}/{compiled['m']}/{compiled['n']}/{compiled['k']}, "
                    f"but this config supplies "
                    f"{requested_counts['g']}/{requested_counts['m']}/"
                    f"{requested_counts['n']}/{requested_counts['k']} "
                    f"-- adjust --problem-configs or CONTRACTION_MULTI_ABD_NUM_DIM_*"
                )
                self.dim_mismatch_skipped += 1
                continue

            self.launch_attempted += 1
            perf = self.run_kernel(kernel_path, params)

            if not perf:
                self.launch_failed += 1
                continue

            structured_result = {
                "name": kernel_info["name"],
                "config_id": kernel_info["config_id"],
                "problem": {
                    "g_dims": g_dims,
                    "m_dims": m_dims,
                    "n_dims": n_dims,
                    "k_dims": k_dims,
                },
                "perf_result": {
                    "latency(ms)": perf["time_ms"],
                    "tflops(TFlops)": perf["tflops"],
                    "bandwidth(GB/s)": perf["bandwidth_gb_s"],
                },
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
                    "tensor_counts": kernel_info.get("tensor_counts", {}),
                    "dim_counts": kernel_info.get("dim_counts", {}),
                },
                "executable": kernel_info["executable"],
                "time_ms": perf["time_ms"],
                "tflops": perf["tflops"],
                "bandwidth_gb_s": perf["bandwidth_gb_s"],
            }
            results.append(structured_result)

            if self.verbose:
                print(
                    f"  {kernel_info['config_id']}: {perf['tflops']:.2f} TFLOPS, "
                    f"{perf['bandwidth_gb_s']:.2f} GB/s, {perf['time_ms']:.2f}ms"
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
        problem_configs: List[Dict],
        verify: bool = False,
        warmup: int = 50,
        repeat: int = 100,
    ) -> Dict:
        """Run a benchmark sweep over problem configurations"""
        kernels = self.discover_kernels()
        if not kernels:
            print("No kernels found!")
            return {}

        all_results = []
        best_kernels = {}

        for cfg in problem_configs:
            g_dims = cfg.get("g_dims", "2")
            m_dims = cfg.get("m_dims", "4,256")
            n_dims = cfg.get("n_dims", "16,128")
            k_dims = cfg.get("k_dims", "64")

            results = self.benchmark_problem_size(
                kernels,
                g_dims,
                m_dims,
                n_dims,
                k_dims,
                verify=1 if verify else 0,
                warmup=warmup,
                repeat=repeat,
            )

            all_results.extend(results)

            best = self.find_best_kernel(results)
            if best:
                key = f"g{g_dims}_m{m_dims}_n{n_dims}_k{k_dims}"
                best_kernels[key] = best
                print(
                    f"Best for {key}: {best['name']} ({best['tflops']:.2f} TFLOPS, "
                    f"{best['bandwidth_gb_s']:.2f} GB/s, {best['time_ms']:.2f}ms)"
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

        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Results exported to {filename}")

    def export_best_kernels(self, best_kernels: Dict, filename: str):
        """Export best kernel selections to file"""
        with open(filename, "w") as f:
            f.write("# Best kernel selections for contraction multi ABD\n")
            f.write(
                "# Format: problem_config -> kernel_name (TFLOPS, bandwidth, latency)\n\n"
            )
            for key, kernel in sorted(best_kernels.items()):
                f.write(
                    f"{key}: {kernel['name']} ({kernel['tflops']:.2f} TFLOPS, "
                    f"{kernel['bandwidth_gb_s']:.2f} GB/s, {kernel['time_ms']:.2f}ms)\n"
                )

        print(f"Best kernels exported to {filename}")

    def export_json(self, filename: str, best_kernels: Dict = None):
        """Export all results and best kernels to JSON"""
        from datetime import datetime

        successful_results = [r for r in self.results if r.get("tflops", 0) > 0]

        tflops_values = [r.get("tflops", 0) for r in successful_results]
        bandwidth_values = [r.get("bandwidth_gb_s", 0) for r in successful_results]
        latency_values = [
            r.get("time_ms", 0) for r in successful_results if r.get("time_ms", 0) > 0
        ]

        pipeline_stats = {}
        scheduler_stats = {}
        data_type_stats = {}

        for result in successful_results:
            config = result.get("config", {})
            for stats_dict, field_name in (
                (pipeline_stats, "pipeline"),
                (scheduler_stats, "scheduler"),
                (data_type_stats, "data_type"),
            ):
                key = config.get(field_name, "unknown")
                entry = stats_dict.setdefault(
                    key, {"count": 0, "avg_tflops": 0, "best_tflops": 0}
                )
                entry["count"] += 1
                entry["best_tflops"] = max(
                    entry["best_tflops"], result.get("tflops", 0)
                )

        for stats_dict, field_name in (
            (pipeline_stats, "pipeline"),
            (scheduler_stats, "scheduler"),
            (data_type_stats, "data_type"),
        ):
            for key in stats_dict:
                relevant = [
                    r
                    for r in successful_results
                    if r.get("config", {}).get(field_name, "unknown") == key
                ]
                if relevant:
                    stats_dict[key]["avg_tflops"] = sum(
                        r.get("tflops", 0) for r in relevant
                    ) / len(relevant)

        output_data = {
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
                "operation": "contraction_multi_abd",
                "total_kernels_tested": len(self.results),
                "unique_kernels": len(
                    set(r.get("name", "unknown") for r in self.results)
                ),
                "successful_runs": len(successful_results),
                "failed_runs": len(self.results) - len(successful_results),
                "launches_attempted": self.launch_attempted,
                "launches_succeeded": self.launch_attempted - self.launch_failed,
                "launches_failed": self.launch_failed,
            },
            "performance_summary": {
                "tflops_stats": {
                    "best": max(tflops_values, default=0),
                    "average": sum(tflops_values) / len(tflops_values)
                    if tflops_values
                    else 0,
                    "min": min(tflops_values, default=0),
                },
                "bandwidth_stats": {
                    "best_gb_s": max(bandwidth_values, default=0),
                    "average_gb_s": sum(bandwidth_values) / len(bandwidth_values)
                    if bandwidth_values
                    else 0,
                },
                "latency_stats": {
                    "best_ms": min(latency_values, default=0),
                    "average_ms": sum(latency_values) / len(latency_values)
                    if latency_values
                    else 0,
                },
                "kernel_type_breakdown": {
                    "by_pipeline": pipeline_stats,
                    "by_scheduler": scheduler_stats,
                    "by_data_type": data_type_stats,
                },
            },
            "kernel_results": self.results,
            "best_kernels_by_problem": best_kernels or {},
        }

        with open(filename, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"JSON results exported to {filename}")
        print(f"  - Total kernels: {len(self.results)}")
        print(f"  - Successful runs: {len(successful_results)}")
        if tflops_values:
            print(f"  - Best TFLOPS: {max(tflops_values):.2f}")
        if bandwidth_values:
            print(f"  - Best bandwidth: {max(bandwidth_values):.2f} GB/s")
        if latency_values:
            print(f"  - Best latency: {min(latency_values):.2f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Contraction Multi-ABD Kernel Benchmarking Tool"
    )
    parser.add_argument(
        "build_dir", help="Build directory containing kernel executables"
    )
    parser.add_argument(
        "--problem-configs",
        nargs="+",
        # Two dims for M and N, one for G and K: matches the CMake defaults
        # (NUM_DIM_G=1, NUM_DIM_M=2, NUM_DIM_N=2, NUM_DIM_K=1) that the kernels
        # are compiled with. The dim COUNT here is not free -- see the guard in
        # benchmark_problem_size.
        default=["g=2;m=4,256;n=16,128;k=64"],
        help="Problem configs as g=<dims>;m=<dims>;n=<dims>;k=<dims> "
        "(dims comma-separated; the count of each must equal the compiled "
        "CONTRACTION_MULTI_ABD_NUM_DIM_*)",
    )
    parser.add_argument("--verify", action="store_true", help="Enable verification")
    parser.add_argument(
        "--csv",
        default="contraction_multi_abd_benchmark_results.csv",
        help="CSV output filename",
    )
    parser.add_argument(
        "--best", default="best_kernels.txt", help="Best kernels output filename"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--warmup", type=int, default=50, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--repeat", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument("--json", help="JSON output filename (optional)")

    args = parser.parse_args()

    dim_keys = {"g", "m", "n", "k"}
    problem_configs = []
    for config_str in args.problem_configs:
        cfg = {}
        for part in config_str.split(";"):
            if not part:
                continue
            if "=" not in part:
                print(f"Error: malformed problem config segment {part!r} "
                      f"in {config_str!r}; expected key=value")
                return 1
            key, val = part.split("=", 1)
            if key in dim_keys:
                cfg[f"{key}_dims"] = val
            else:
                cfg[key] = val
        problem_configs.append(cfg)

    benchmark = ContractionMultiAbdBenchmark(args.build_dir, verbose=args.verbose)

    print("Starting Contraction Multi-ABD kernel benchmark sweep...")
    start_time = time.time()

    best_kernels = benchmark.benchmark_sweep(
        problem_configs=problem_configs,
        verify=args.verify,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    elapsed_time = time.time() - start_time
    print(f"\nBenchmark completed in {elapsed_time:.2f} seconds")

    benchmark.export_csv(args.csv)
    benchmark.export_best_kernels(best_kernels, args.best)

    if args.json:
        benchmark.export_json(args.json, best_kernels)

    # Exiting 0 after every launch failed would leave the CI lane green with no signal
    attempted = benchmark.launch_attempted
    failed = benchmark.launch_failed
    succeeded = attempted - failed
    print(f"Launches: {attempted} attempted, {succeeded} succeeded, {failed} failed")

    if attempted == 0:
        if benchmark.dim_mismatch_skipped:
            print(
                f"No kernel launches were attempted - all "
                f"{benchmark.dim_mismatch_skipped} discovered kernel(s) were "
                f"compiled for a different g/m/n/k dim arity than "
                f"--problem-configs supplies"
            )
        else:
            print("No kernel launches were attempted - no kernels discovered")
        return 1
    if not benchmark.results:
        print("No benchmark results were collected")
        return 1
    if failed > 0:
        # A warning, not a failure: this sweep runs every generated config, and
        # individual configs legitimately fail to launch (unsupported tile shape
        # for the arch, workspace too large for the problem size). Making any
        # single failure red would trade a permanently-green lane for a
        # permanently-red one, which is no more informative. Total failure is
        # already caught by the empty-result check above, and the per-run counts
        # are in the JSON (launches_attempted/succeeded/failed) for anyone
        # tracking the trend.
        print(f"WARNING: {failed} of {attempted} kernel launch(es) failed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
