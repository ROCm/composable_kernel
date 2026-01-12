#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import subprocess
import csv
from pathlib import Path
from typing import List, Dict, Optional


def run_kernel(
    build_dir: Path, kernel_path: Path, params: Dict[str, str], verbose: bool = False
) -> Optional[Dict]:
    """Run a single kernel with given parameters and save output to individual JSON file"""
    # Create results directory
    results_dir = build_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate unique JSON filename for this kernel
    json_file = results_dir / f"{kernel_path.stem}.json"

    cmd = [str(kernel_path)]

    # Add parameters
    for key, value in params.items():
        cmd.append(f"-{key}={value}")

    # Add JSON output flag for clean JSON output
    cmd.append("-json_output=true")

    if verbose:
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
            return parse_json_file(json_file, verbose=verbose)
        else:
            print(f"No output from {kernel_path.name}")
            return None

    except subprocess.TimeoutExpired:
        print(f"Timeout running {kernel_path.name}")
        return None
    except Exception as e:
        print(f"Error running {kernel_path.name}: {e}")
        return None


def parse_json_file(json_file: Path, verbose: bool = False) -> Optional[Dict]:
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
        if verbose:
            print(f"Failed to parse JSON from {json_file}: {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"Error reading JSON file {json_file}: {e}")
        return None


def find_best_kernel(results: List[Dict], metric: str = "tflops") -> Optional[Dict]:
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


def export_csv(results: List[Dict], filename: str, verbose: bool = False):
    """Export all results to CSV"""
    if not results:
        print("No results to export")
        return

    # Get all unique keys from results
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())

    # Sort keys for consistent output
    fieldnames = sorted(all_keys)

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results exported to {filename}")


def export_best_kernels(best_kernels: Dict, filename: str, verbose: bool = False):
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


def export_json(
    results: List[Dict], filename: str, best_kernels: Dict = None, verbose: bool = False
):
    """Export all results and best kernels to JSON with comprehensive metadata"""
    from datetime import datetime

    # Calculate comprehensive summary statistics for all metrics
    successful_results = [r for r in results if r.get("tflops", 0) > 0]

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
        # Get config info from the new structure
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
            "total_kernels_tested": len(results),
            "unique_kernels": len(set(r.get("name", "unknown") for r in results)),
            "successful_runs": len(successful_results),
            "failed_runs": len(results) - len(successful_results),
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
            "total_problem_configurations": len(best_kernels) if best_kernels else 0,
        },
        "kernel_results": results,
        "best_kernels_by_problem": best_kernels or {},
    }

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"JSON results exported to {filename}")
    print(f"  - Total kernels: {len(results)}")
    print(f"  - Successful runs: {len(successful_results)}")
    print(f"  - Best TFLOPS: {max(tflops_values, default=0):.2f}")
    print(f"  - Best bandwidth: {max(bandwidth_values, default=0):.2f} GB/s")
    print(f"  - Best latency: {min(latency_values, default=0):.2f}ms")
