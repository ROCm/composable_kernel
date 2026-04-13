#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GEMM Universal Benchmark Data Generation Script

This script generates training data for ML-based kernel selection heuristics by:
1. Reading kernel configurations from the tile engine
2. Building benchmark executables (in parallel)
3. Running benchmarks across multiple problem sizes
4. Outputting performance data in JSON format

Usage:
    python generate_benchmark_data.py \
        --build_dir /tmp/build \
        --output_dir /tmp/benchmark_data \
        --dtype fp16 \
        --layout rcr \
        --num_build_jobs 4 \
        --num_benchmark_jobs 1

Requirements:
    - ROCm-capable GPU
    - CK tile engine built with CMake
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import re


@dataclass
class KernelConfig:
    """Represents a single kernel configuration."""

    name: str
    dtype: str
    layout: str
    pipeline: str
    epilogue: str
    scheduler: str
    pad_m: bool
    pad_n: bool
    pad_k: bool
    persistent: bool
    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    @classmethod
    def from_kernel_name(cls, name: str, dtype: str, layout: str) -> "KernelConfig":
        """Parse kernel name to extract configuration."""
        # Format: gemm_universal_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_{padM}_{padN}_{padK}_{persistent}_{tile_config}
        # tile_config: {tile_m}x{tile_n}x{tile_k}_{warp_m}x{warp_n}x{warp_k}_{warp_tile_m}x{warp_tile_n}x{warp_tile_k}

        parts = name.split("_")
        prefix = f"gemm_universal_{dtype}_{layout}_"
        trait_and_tile = name[len(prefix) :]
        trait_parts = trait_and_tile.split("_")

        pipeline = trait_parts[0]
        epilogue = trait_parts[1]
        scheduler = trait_parts[2]
        pad_m = trait_parts[3] == "True"
        pad_n = trait_parts[4] == "True"
        pad_k = trait_parts[5] == "True"
        persistent = trait_parts[6] == "True"

        # Parse tile config
        tile_dims = trait_parts[7].split("x")
        warp_dims = trait_parts[8].split("x")
        warp_tile_dims = trait_parts[9].split("x")

        return cls(
            name=name,
            dtype=dtype,
            layout=layout,
            pipeline=pipeline,
            epilogue=epilogue,
            scheduler=scheduler,
            pad_m=pad_m,
            pad_n=pad_n,
            pad_k=pad_k,
            persistent=persistent,
            tile_m=int(tile_dims[0]),
            tile_n=int(tile_dims[1]),
            tile_k=int(tile_dims[2]),
            warp_m=int(warp_dims[0]),
            warp_n=int(warp_dims[1]),
            warp_k=int(warp_dims[2]),
            warp_tile_m=int(warp_tile_dims[0]),
            warp_tile_n=int(warp_tile_dims[1]),
            warp_tile_k=int(warp_tile_dims[2]),
        )


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    kernel_name: str
    m: int
    n: int
    k: int
    avg_time_ms: float
    tflops: float
    is_valid: bool
    error: Optional[str] = None


@dataclass
class ProblemSize:
    """GEMM problem dimensions."""

    m: int
    n: int
    k: int


def get_problem_sizes() -> List[ProblemSize]:
    """
    Generate diverse problem sizes for benchmarking.

    Includes:
    - Square matrices (powers of 2)
    - Rectangular matrices (common in ML)
    - LLM-specific sizes (attention, MLP)
    - Edge cases (small, very large)
    """
    sizes = []

    # Powers of 2 (square)
    for p in [6, 7, 8, 9, 10, 11, 12, 13]:  # 64 to 8192
        dim = 2**p
        sizes.append(ProblemSize(dim, dim, dim))

    # Common ML sizes (batch x hidden)
    ml_sizes = [
        (1, 4096, 4096),  # Single token inference
        (8, 4096, 4096),  # Small batch
        (32, 4096, 4096),  # Medium batch
        (128, 4096, 4096),  # Large batch
        (1, 4096, 11008),  # LLaMA MLP up-projection
        (1, 11008, 4096),  # LLaMA MLP down-projection
        (32, 4096, 11008),
        (32, 11008, 4096),
        (1, 8192, 8192),  # Large model
        (32, 8192, 8192),
        (1, 8192, 28672),  # LLaMA-70B MLP
        (32, 8192, 28672),
    ]
    for m, n, k in ml_sizes:
        sizes.append(ProblemSize(m, n, k))

    # Rectangular matrices
    rect_sizes = [
        (1024, 4096, 1024),
        (4096, 1024, 4096),
        (2048, 8192, 2048),
        (256, 256, 8192),  # Tall K
        (8192, 8192, 256),  # Short K
    ]
    for m, n, k in rect_sizes:
        sizes.append(ProblemSize(m, n, k))

    # Remove duplicates while preserving order
    seen = set()
    unique_sizes = []
    for s in sizes:
        key = (s.m, s.n, s.k)
        if key not in seen:
            seen.add(key)
            unique_sizes.append(s)

    return unique_sizes


def load_kernel_list(build_dir: Path, dtype: str, layout: str) -> List[KernelConfig]:
    """Load kernel configurations from the tile engine build."""
    kernel_list_path = (
        build_dir
        / "tile_engine"
        / "ops"
        / "gemm"
        / "gemm_universal"
        / dtype
        / layout
        / "gemm_universal_kernel_list.txt"
    )

    if not kernel_list_path.exists():
        raise FileNotFoundError(f"Kernel list not found: {kernel_list_path}")

    kernels = []
    with open(kernel_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: kernel_name|tile_config|trait_combo
            parts = line.split("|")
            kernel_name = parts[0]
            kernels.append(KernelConfig.from_kernel_name(kernel_name, dtype, layout))

    return kernels


def build_kernel(build_dir: Path, kernel: KernelConfig) -> Tuple[str, bool, str]:
    """
    Build a single kernel benchmark executable.

    Returns: (kernel_name, success, error_message)
    """
    target_name = f"benchmark_{kernel.name}"

    try:
        result = subprocess.run(
            ["ninja", "-j1", target_name],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            return (kernel.name, False, result.stderr[:500])

        return (kernel.name, True, "")
    except subprocess.TimeoutExpired:
        return (kernel.name, False, "Build timeout")
    except Exception as e:
        return (kernel.name, False, str(e))


def run_benchmark(
    build_dir: Path,
    kernel: KernelConfig,
    problem: ProblemSize,
    warmup: int = 10,
    repeat: int = 50,
) -> BenchmarkResult:
    """
    Run benchmark for a single kernel and problem size.
    """
    exe_path = build_dir / "bin" / f"benchmark_{kernel.name}"

    if not exe_path.exists():
        return BenchmarkResult(
            kernel_name=kernel.name,
            m=problem.m,
            n=problem.n,
            k=problem.k,
            avg_time_ms=0,
            tflops=0,
            is_valid=False,
            error="Executable not found",
        )

    try:
        result = subprocess.run(
            [
                str(exe_path),
                f"-m={problem.m}",
                f"-n={problem.n}",
                f"-k={problem.k}",
                f"-warmup={warmup}",
                f"-repeat={repeat}",
                "-verify=0",
                "-json_output=true",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            # Try to parse error
            error = result.stderr[:200] if result.stderr else result.stdout[:200]
            return BenchmarkResult(
                kernel_name=kernel.name,
                m=problem.m,
                n=problem.n,
                k=problem.k,
                avg_time_ms=0,
                tflops=0,
                is_valid=False,
                error=error,
            )

        # Parse JSON output
        output = result.stdout.strip()

        # Try to find JSON in output
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            # Extract from nested perf_result object
            perf = data.get("perf_result", {})
            avg_time_ms = perf.get("latency(ms)", 0)
            tflops = perf.get("tflops(TFlops)", 0)

            return BenchmarkResult(
                kernel_name=kernel.name,
                m=problem.m,
                n=problem.n,
                k=problem.k,
                avg_time_ms=avg_time_ms,
                tflops=tflops,
                is_valid=True,
            )
        else:
            # Parse from text output
            # Look for patterns like "avg_time: X ms" or "tflops: Y"
            avg_time = 0.0
            tflops = 0.0

            time_match = re.search(
                r"(?:avg[_\s]?time|latency)[:\s]+(\d+\.?\d*)\s*(?:ms)?", output, re.I
            )
            if time_match:
                avg_time = float(time_match.group(1))

            tflops_match = re.search(r"tflops[:\s]+(\d+\.?\d*)", output, re.I)
            if tflops_match:
                tflops = float(tflops_match.group(1))

            # Calculate TFLOPs if not provided
            if tflops == 0 and avg_time > 0:
                flops = 2.0 * problem.m * problem.n * problem.k
                tflops = flops / (avg_time * 1e-3) / 1e12

            return BenchmarkResult(
                kernel_name=kernel.name,
                m=problem.m,
                n=problem.n,
                k=problem.k,
                avg_time_ms=avg_time,
                tflops=tflops,
                is_valid=avg_time > 0,
                error=None if avg_time > 0 else "Could not parse output",
            )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            kernel_name=kernel.name,
            m=problem.m,
            n=problem.n,
            k=problem.k,
            avg_time_ms=0,
            tflops=0,
            is_valid=False,
            error="Benchmark timeout",
        )
    except Exception as e:
        return BenchmarkResult(
            kernel_name=kernel.name,
            m=problem.m,
            n=problem.n,
            k=problem.k,
            avg_time_ms=0,
            tflops=0,
            is_valid=False,
            error=str(e),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate GEMM benchmark data for ML training"
    )
    parser.add_argument(
        "--build_dir", type=str, default="/tmp/build", help="CK build directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/benchmark_data",
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp8", "bf16", "bf8"],
        help="Data type to benchmark",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="rcr",
        choices=["rcr", "rrr", "crr", "ccr"],
        help="Matrix layout to benchmark",
    )
    parser.add_argument(
        "--num_build_jobs", type=int, default=4, help="Number of parallel build jobs"
    )
    parser.add_argument(
        "--num_benchmark_jobs",
        type=int,
        default=1,
        help="Number of parallel benchmark jobs (use 1 for accurate timing)",
    )
    parser.add_argument(
        "--max_kernels",
        type=int,
        default=None,
        help="Maximum number of kernels to benchmark (for testing)",
    )
    parser.add_argument(
        "--skip_build",
        action="store_true",
        help="Skip building and only run benchmarks",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--repeat", type=int, default=50, help="Number of benchmark iterations"
    )

    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load kernel configurations
    print(f"Loading kernel list for {args.dtype}/{args.layout}...")
    kernels = load_kernel_list(build_dir, args.dtype, args.layout)
    print(f"Found {len(kernels)} kernel configurations")

    if args.max_kernels:
        kernels = kernels[: args.max_kernels]
        print(f"Limiting to {len(kernels)} kernels")

    # Build kernels
    if not args.skip_build:
        print(
            f"\nBuilding {len(kernels)} kernels with {args.num_build_jobs} parallel jobs..."
        )
        build_results = {"success": 0, "failed": 0, "failed_kernels": []}

        with ProcessPoolExecutor(max_workers=args.num_build_jobs) as executor:
            futures = {executor.submit(build_kernel, build_dir, k): k for k in kernels}

            for i, future in enumerate(as_completed(futures)):
                kernel_name, success, error = future.result()
                if success:
                    build_results["success"] += 1
                else:
                    build_results["failed"] += 1
                    build_results["failed_kernels"].append(
                        {"name": kernel_name, "error": error}
                    )

                if (i + 1) % 10 == 0:
                    print(
                        f"  Built {i + 1}/{len(kernels)} ({build_results['success']} success, {build_results['failed']} failed)"
                    )

        print(
            f"\nBuild complete: {build_results['success']} success, {build_results['failed']} failed"
        )

        # Save build results
        with open(output_dir / "build_results.json", "w") as f:
            json.dump(build_results, f, indent=2)

    # Get problem sizes
    problem_sizes = get_problem_sizes()
    print(f"\nBenchmarking {len(problem_sizes)} problem sizes...")

    # Run benchmarks
    all_results = []
    total_benchmarks = len(kernels) * len(problem_sizes)
    completed = 0

    print(f"Total benchmarks to run: {total_benchmarks}")

    for kernel in kernels:
        kernel_results = {
            "kernel_config": asdict(kernel),
            "benchmarks": [],
        }

        for problem in problem_sizes:
            result = run_benchmark(
                build_dir,
                kernel,
                problem,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            kernel_results["benchmarks"].append(asdict(result))
            completed += 1

            if completed % 100 == 0:
                print(f"  Progress: {completed}/{total_benchmarks} benchmarks complete")

        all_results.append(kernel_results)

        # Save intermediate results
        intermediate_file = (
            output_dir / f"benchmark_results_{args.dtype}_{args.layout}_partial.json"
        )
        with open(intermediate_file, "w") as f:
            json.dump(all_results, f, indent=2)

    # Save final results
    final_file = output_dir / f"benchmark_results_{args.dtype}_{args.layout}.json"
    with open(final_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "dtype": args.dtype,
                    "layout": args.layout,
                    "num_kernels": len(kernels),
                    "num_problems": len(problem_sizes),
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "problem_sizes": [asdict(p) for p in problem_sizes],
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {final_file}")

    # Print summary
    valid_count = sum(
        1 for kr in all_results for br in kr["benchmarks"] if br["is_valid"]
    )
    print(f"Valid benchmarks: {valid_count}/{total_benchmarks}")


if __name__ == "__main__":
    main()
