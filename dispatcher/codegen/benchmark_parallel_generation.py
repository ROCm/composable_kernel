#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark parallel vs sequential kernel generation.

Times generation of ~128 kernel configurations for both GEMM and Conv,
comparing parallel and sequential modes.

Usage:
    python3 benchmark_parallel_generation.py
    python3 benchmark_parallel_generation.py --num-kernels 64
"""

import argparse
import time
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from unified_gemm_codegen import UnifiedGemmCodegen, GemmVariant
from unified_conv_codegen import UnifiedConvCodegen, ConvVariant


def benchmark_gemm_generation(num_kernels: int = 128, verbose: bool = True):
    """Benchmark GEMM kernel generation with and without parallelism."""

    results = {}

    # Note: num_kernels is used for reporting; actual kernel count depends on
    # UnifiedGemmCodegen's internal configuration (datatype, layout, variants)

    print(f"\n{'=' * 70}")
    print("GEMM Kernel Generation Benchmark")
    print(f"{'=' * 70}")
    print(f"Target kernels: ~{num_kernels}")

    for parallel in [True, False]:
        mode = "parallel" if parallel else "sequential"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            codegen = UnifiedGemmCodegen(
                output_dir=output_dir,
                datatype="fp16",
                layout="rcr",
                variants=[GemmVariant.STANDARD],
                gpu_target="gfx942",
            )

            start = time.perf_counter()
            result = codegen.generate_all(parallel=parallel)
            elapsed = time.perf_counter() - start

            num_generated = len(result.get("kernels", []))
            results[mode] = {
                "time_s": elapsed,
                "num_kernels": num_generated,
                "kernels_per_sec": num_generated / elapsed if elapsed > 0 else 0,
            }

            if verbose:
                print(f"\n  {mode.upper()}:")
                print(f"    Kernels generated: {num_generated}")
                print(f"    Time: {elapsed:.2f}s")
                print(f"    Rate: {results[mode]['kernels_per_sec']:.1f} kernels/s")

    # Summary
    if "parallel" in results and "sequential" in results:
        speedup = results["sequential"]["time_s"] / results["parallel"]["time_s"]
        print(f"\n  SPEEDUP: {speedup:.2f}x faster with parallel")

    return results


def benchmark_conv_generation(num_kernels: int = 128, verbose: bool = True):
    """Benchmark Conv kernel generation with and without parallelism."""

    results = {}

    print(f"\n{'=' * 70}")
    print("Conv Kernel Generation Benchmark")
    print(f"{'=' * 70}")
    print(f"Target kernels: ~{num_kernels}")

    for parallel in [True, False]:
        mode = "parallel" if parallel else "sequential"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            codegen = UnifiedConvCodegen(
                output_dir=output_dir,
                gpu_target="gfx942",
                enable_arch_filter=False,  # Disable for faster benchmark
            )

            # Create configs for ~num_kernels
            from unified_conv_codegen import ConvKernelConfig, TileConfig, TraitConfig

            configs = []
            tile_configs = [
                (16, 64, 64, 1, 4, 1, 16, 16, 32),
                (128, 128, 32, 2, 2, 1, 32, 32, 16),
                (256, 256, 64, 2, 2, 1, 32, 32, 16),
                (64, 64, 32, 2, 2, 1, 32, 32, 16),
            ]

            pipelines = ["mem", "compv3"]
            schedulers = ["intrawave", "interwave"]

            for tile_m, tile_n, tile_k, wm, wn, wk, wtm, wtn, wtk in tile_configs:
                for pipeline in pipelines:
                    for scheduler in schedulers:
                        # Skip invalid combinations
                        if pipeline == "compv3" and scheduler == "interwave":
                            continue

                        tile = TileConfig(
                            tile_m=tile_m,
                            tile_n=tile_n,
                            tile_k=tile_k,
                            warp_m=wm,
                            warp_n=wn,
                            warp_k=wk,
                            warp_tile_m=wtm,
                            warp_tile_n=wtn,
                            warp_tile_k=wtk,
                        )
                        trait = TraitConfig(
                            pipeline=pipeline, scheduler=scheduler, epilogue="cshuffle"
                        )
                        configs.append(
                            ConvKernelConfig(
                                tile=tile,
                                trait=trait,
                                variant=ConvVariant.FORWARD,
                                ndim_spatial=2,
                            )
                        )

                        if len(configs) >= num_kernels:
                            break
                    if len(configs) >= num_kernels:
                        break
                if len(configs) >= num_kernels:
                    break

            start = time.perf_counter()
            generated = codegen.generate_all(configs, ["fp16"], parallel=parallel)
            elapsed = time.perf_counter() - start

            num_generated = len(generated)
            results[mode] = {
                "time_s": elapsed,
                "num_kernels": num_generated,
                "kernels_per_sec": num_generated / elapsed if elapsed > 0 else 0,
            }

            if verbose:
                print(f"\n  {mode.upper()}:")
                print(f"    Kernels generated: {num_generated}")
                print(f"    Time: {elapsed:.2f}s")
                print(f"    Rate: {results[mode]['kernels_per_sec']:.1f} kernels/s")

    # Summary
    if "parallel" in results and "sequential" in results:
        speedup = results["sequential"]["time_s"] / results["parallel"]["time_s"]
        print(f"\n  SPEEDUP: {speedup:.2f}x faster with parallel")

    return results


def benchmark_python_codegen_runner(num_kernels: int = 128, verbose: bool = True):
    """Benchmark Python CodegenRunner with parallel execution."""

    print(f"\n{'=' * 70}")
    print("Python CodegenRunner Benchmark (GEMM)")
    print(f"{'=' * 70}")

    # Add path for ctypes_utils
    sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

    try:
        from ctypes_utils import CodegenRunner
    except ImportError:
        print("  SKIPPED: ctypes_utils not available")
        return {}

    results = {}

    for parallel in [True, False]:
        mode = "parallel" if parallel else "sequential"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            codegen = CodegenRunner(
                output_dir=output_dir,
                datatype="fp16",
                layout="rcr",
                gpu_target="gfx942",
            )

            start = time.perf_counter()
            if parallel:
                result = codegen.generate_all_parallel(
                    output_dir=output_dir,
                    variants=["standard"],
                    verbose=False,
                )
            else:
                result = codegen.generate_all(output_dir=output_dir)
            elapsed = time.perf_counter() - start

            num_generated = (
                sum(r.kernel_count for r in result) if isinstance(result, list) else 0
            )

            results[mode] = {
                "time_s": elapsed,
                "num_kernels": num_generated,
            }

            if verbose:
                print(f"\n  {mode.upper()}:")
                print(
                    f"    Variants processed: {len(result) if isinstance(result, list) else 1}"
                )
                print(f"    Kernels generated: {num_generated}")
                print(f"    Time: {elapsed:.2f}s")

    if (
        "parallel" in results
        and "sequential" in results
        and results["sequential"]["time_s"] > 0
    ):
        speedup = results["sequential"]["time_s"] / results["parallel"]["time_s"]
        print(f"\n  SPEEDUP: {speedup:.2f}x faster with parallel")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark parallel kernel generation")
    parser.add_argument(
        "--num-kernels",
        type=int,
        default=128,
        help="Target number of kernels to generate (default: 128)",
    )
    parser.add_argument("--gemm-only", action="store_true", help="Only benchmark GEMM")
    parser.add_argument("--conv-only", action="store_true", help="Only benchmark Conv")
    parser.add_argument(
        "--python-only", action="store_true", help="Only benchmark Python CodegenRunner"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PARALLEL KERNEL GENERATION BENCHMARK")
    print("=" * 70)
    print(f"\nCPU cores available: {os.cpu_count()}")
    print(f"Target kernels: {args.num_kernels}")

    all_results = {}

    if not args.conv_only and not args.python_only:
        all_results["gemm"] = benchmark_gemm_generation(args.num_kernels)

    if not args.gemm_only and not args.python_only:
        all_results["conv"] = benchmark_conv_generation(args.num_kernels)

    if not args.gemm_only and not args.conv_only:
        all_results["python_codegen"] = benchmark_python_codegen_runner(
            args.num_kernels
        )

    # Final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for name, results in all_results.items():
        if results and "parallel" in results and "sequential" in results:
            par = results["parallel"]
            seq = results["sequential"]
            if seq["time_s"] > 0:
                speedup = seq["time_s"] / par["time_s"]
                print(f"\n{name.upper()}:")
                print(f"  Sequential: {seq['time_s']:.2f}s")
                print(f"  Parallel:   {par['time_s']:.2f}s")
                print(f"  Speedup:    {speedup:.2f}x")

    print(f"\n{'=' * 70}")
    print("Parallel is DEFAULT (--no-parallel to disable)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
