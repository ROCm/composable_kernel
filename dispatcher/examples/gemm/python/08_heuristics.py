#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 08: Custom Heuristics

Demonstrates custom kernel selection heuristics based on problem characteristics.

This example shows how to:
1. Define multiple kernel configurations for different workloads
2. Implement custom heuristics to select the best kernel
3. Test heuristic selection across different problem sizes

Heuristic strategies:
- Size-based: Small tiles for small problems, large tiles for large problems
- Compute-bound: Maximize compute utilization for large matrices
- Memory-bound: Optimize memory access for bandwidth-limited cases
- Latency-focused: Minimize kernel launch overhead for small problems


Usage:
    python3 08_heuristics.py
    python3 08_heuristics.py --help
    python3 08_heuristics.py --strategy compute
    python3 08_heuristics.py --dtype bf16
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
    reset_for_example,
    detect_gpu_arch,
)


# =============================================================================
# Kernel Specifications
# =============================================================================


@dataclass
class KernelSpec:
    """Kernel specification with metadata for heuristic selection"""

    name: str
    tile_m: int
    tile_n: int
    tile_k: int
    pipeline: str = "compv3"
    scheduler: str = "intrawave"
    # Metadata for heuristics
    category: str = "balanced"  # small, balanced, large, compute, memory
    min_problem_size: int = 0
    max_problem_size: int = float("inf")


# Define kernel pool for heuristic selection (20+ kernels)
KERNEL_POOL = [
    # ==========================================================================
    # SMALL TILES - Low latency, good for small problems
    # ==========================================================================
    KernelSpec(
        "small_64x64_k32",
        64,
        64,
        32,
        "compv3",
        "intrawave",
        category="small",
        max_problem_size=256 * 256,
    ),
    KernelSpec(
        "small_64x64_k64",
        64,
        64,
        64,
        "compv3",
        "intrawave",
        category="small",
        max_problem_size=256 * 256,
    ),
    KernelSpec(
        "small_64x64_v4",
        64,
        64,
        32,
        "compv4",
        "intrawave",
        category="small",
        max_problem_size=256 * 256,
    ),
    # ==========================================================================
    # MEDIUM TILES - Balanced performance
    # ==========================================================================
    KernelSpec(
        "medium_128x128_k32",
        128,
        128,
        32,
        "compv3",
        "intrawave",
        category="balanced",
        min_problem_size=128 * 128,
        max_problem_size=2048 * 2048,
    ),
    KernelSpec(
        "medium_128x128_k64",
        128,
        128,
        64,
        "compv3",
        "intrawave",
        category="balanced",
        min_problem_size=256 * 256,
    ),
    KernelSpec(
        "medium_128x128_k128",
        128,
        128,
        128,
        "compv3",
        "intrawave",
        category="balanced",
        min_problem_size=256 * 256,
    ),
    KernelSpec(
        "medium_128x128_v4_k32",
        128,
        128,
        32,
        "compv4",
        "intrawave",
        category="balanced",
        min_problem_size=256 * 256,
    ),
    KernelSpec(
        "medium_128x128_v4_k64",
        128,
        128,
        64,
        "compv4",
        "intrawave",
        category="balanced",
        min_problem_size=256 * 256,
    ),
    # Rectangular medium tiles
    KernelSpec(
        "rect_64x128_k32",
        64,
        128,
        32,
        "compv3",
        "intrawave",
        category="balanced",
        min_problem_size=128 * 128,
    ),
    KernelSpec(
        "rect_128x64_k32",
        128,
        64,
        32,
        "compv3",
        "intrawave",
        category="balanced",
        min_problem_size=128 * 128,
    ),
    KernelSpec(
        "rect_64x128_k64",
        64,
        128,
        64,
        "compv3",
        "intrawave",
        category="balanced",
        min_problem_size=256 * 256,
    ),
    KernelSpec(
        "rect_128x64_k64",
        128,
        64,
        64,
        "compv3",
        "intrawave",
        category="balanced",
        min_problem_size=256 * 256,
    ),
    # ==========================================================================
    # LARGE TILES - High throughput for large problems
    # ==========================================================================
    KernelSpec(
        "large_256x128_k32",
        256,
        128,
        32,
        "compv3",
        "intrawave",
        category="large",
        min_problem_size=512 * 512,
    ),
    KernelSpec(
        "large_256x128_k64",
        256,
        128,
        64,
        "compv3",
        "intrawave",
        category="large",
        min_problem_size=512 * 512,
    ),
    KernelSpec(
        "large_128x256_k32",
        128,
        256,
        32,
        "compv3",
        "intrawave",
        category="large",
        min_problem_size=512 * 512,
    ),
    KernelSpec(
        "large_128x256_k64",
        128,
        256,
        64,
        "compv3",
        "intrawave",
        category="large",
        min_problem_size=512 * 512,
    ),
    KernelSpec(
        "large_256x256_k32",
        256,
        256,
        32,
        "compv3",
        "intrawave",
        category="large",
        min_problem_size=1024 * 1024,
    ),
    KernelSpec(
        "large_256x256_k64",
        256,
        256,
        64,
        "compv3",
        "intrawave",
        category="large",
        min_problem_size=1024 * 1024,
    ),
    # ==========================================================================
    # COMPUTE-OPTIMIZED - compv4 pipeline for compute-bound workloads
    # ==========================================================================
    KernelSpec(
        "compute_128x128_v4_k32",
        128,
        128,
        32,
        "compv4",
        "intrawave",
        category="compute",
        min_problem_size=256 * 256,
    ),
    KernelSpec(
        "compute_128x128_v4_k64",
        128,
        128,
        64,
        "compv4",
        "intrawave",
        category="compute",
        min_problem_size=256 * 256,
    ),
    KernelSpec(
        "compute_256x128_v4",
        256,
        128,
        64,
        "compv4",
        "intrawave",
        category="compute",
        min_problem_size=512 * 512,
    ),
    KernelSpec(
        "compute_256x256_v4",
        256,
        256,
        64,
        "compv4",
        "intrawave",
        category="compute",
        min_problem_size=1024 * 1024,
    ),
    # ==========================================================================
    # MEMORY-OPTIMIZED - Good cache utilization for memory-bound workloads
    # ==========================================================================
    KernelSpec(
        "memory_128x128_k16",
        128,
        128,
        16,
        "compv3",
        "intrawave",
        category="memory",
        min_problem_size=256 * 256,
    ),
    KernelSpec(
        "memory_64x128_k16",
        64,
        128,
        16,
        "compv3",
        "intrawave",
        category="memory",
        min_problem_size=128 * 128,
    ),
]


def create_kernel_config(spec: KernelSpec, dtype: str, arch: str) -> KernelConfig:
    """Create KernelConfig from spec"""
    warp_m = 16 if spec.tile_m <= 64 else 32
    warp_n = 16 if spec.tile_n <= 64 else 32

    return KernelConfig(
        dtype_a=dtype,
        dtype_b=dtype,
        dtype_c=dtype,
        dtype_acc="fp32",
        layout_a="row",
        layout_b="col",
        layout_c="row",
        tile_m=spec.tile_m,
        tile_n=spec.tile_n,
        tile_k=spec.tile_k,
        wave_m=2,
        wave_n=2,
        wave_k=1,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_k=16,
        pipeline=spec.pipeline,
        scheduler=spec.scheduler,
        epilogue="cshuffle",
        gfx_arch=arch,
    )


# =============================================================================
# Heuristic Strategies
# =============================================================================


class HeuristicStrategy(Enum):
    SIZE_BASED = "size"
    COMPUTE_BOUND = "compute"
    MEMORY_BOUND = "memory"
    LATENCY_FOCUSED = "latency"


def size_based_heuristic(
    M: int, N: int, K: int, kernels: List[KernelSpec]
) -> KernelSpec:
    """
    Select kernel based on problem size.
    - Small problems: Use small tiles for low latency
    - Medium problems: Use balanced tiles
    - Large problems: Use large tiles for high throughput

    Also considers K dimension for tile_k selection.
    """
    total_elements = M * N

    # Filter by problem size constraints
    candidates = [
        k for k in kernels if k.min_problem_size <= total_elements <= k.max_problem_size
    ]

    if not candidates:
        candidates = kernels  # Fall back to all kernels

    # Determine target category based on problem size
    if total_elements < 256 * 256:
        target_category = "small"
    elif total_elements < 1024 * 1024:
        target_category = "balanced"
    else:
        target_category = "large"

    # Filter by category if possible
    category_candidates = [k for k in candidates if k.category == target_category]
    if category_candidates:
        candidates = category_candidates

    # Select best tile_k based on K dimension
    # Prefer tile_k that divides K well
    def tile_k_score(k):
        if K % k.tile_k == 0:
            return 0  # Perfect division
        return K % k.tile_k  # Remainder (lower is better)

    # Sort by tile_k fit, then by tile size
    candidates.sort(key=lambda k: (tile_k_score(k), -k.tile_m * k.tile_n))

    return candidates[0]


def compute_bound_heuristic(
    M: int, N: int, K: int, kernels: List[KernelSpec]
) -> KernelSpec:
    """
    Select kernel optimized for compute-bound workloads.
    Prefers compv4 pipeline and larger tiles.
    Selects based on problem size to maximize compute utilization.
    """
    total_elements = M * N

    # Prefer compute category kernels
    compute_kernels = [k for k in kernels if k.category == "compute"]

    if not compute_kernels:
        # Fall back to compv4 kernels
        compute_kernels = [k for k in kernels if k.pipeline == "compv4"]

    if not compute_kernels:
        compute_kernels = kernels

    # Filter by problem size
    valid = [k for k in compute_kernels if k.min_problem_size <= total_elements]
    if valid:
        compute_kernels = valid

    # For large problems, prefer larger tiles
    if total_elements >= 1024 * 1024:
        return max(compute_kernels, key=lambda k: k.tile_m * k.tile_n * k.tile_k)
    else:
        # For smaller problems, prefer medium tiles
        return min(
            compute_kernels, key=lambda k: abs(k.tile_m - 128) + abs(k.tile_n - 128)
        )


def memory_bound_heuristic(
    M: int, N: int, K: int, kernels: List[KernelSpec]
) -> KernelSpec:
    """
    Select kernel optimized for memory-bound workloads.
    Prefers smaller tile_k for better memory access patterns.
    """
    # Prefer memory category kernels first
    memory_kernels = [k for k in kernels if k.category == "memory"]
    if memory_kernels:
        # Select based on problem size
        total = M * N
        if total < 512 * 512:
            return min(memory_kernels, key=lambda k: k.tile_m * k.tile_n)
        return max(memory_kernels, key=lambda k: k.tile_m * k.tile_n)

    # Fall back to balanced with smaller tile_k
    balanced = [k for k in kernels if k.category == "balanced"]
    if balanced:
        # Prefer smaller tile_k for memory-bound
        return min(balanced, key=lambda k: k.tile_k)

    # Fall back to medium-sized tile with small tile_k
    return min(
        kernels, key=lambda k: (k.tile_k, abs(k.tile_m - 128) + abs(k.tile_n - 128))
    )


def latency_focused_heuristic(
    M: int, N: int, K: int, kernels: List[KernelSpec]
) -> KernelSpec:
    """
    Select kernel optimized for low latency.
    Prefers smaller tiles and compv4 for faster execution.
    """
    # Prefer small category
    small_kernels = [k for k in kernels if k.category == "small"]

    if small_kernels:
        # Among small kernels, prefer compv4 for lower latency
        v4_small = [k for k in small_kernels if k.pipeline == "compv4"]
        if v4_small:
            return v4_small[0]
        return small_kernels[0]

    # Fall back to smallest tile with compv4 if available
    all_v4 = [k for k in kernels if k.pipeline == "compv4"]
    if all_v4:
        return min(all_v4, key=lambda k: k.tile_m * k.tile_n)

    # Fall back to smallest tile
    return min(kernels, key=lambda k: k.tile_m * k.tile_n)


HEURISTICS = {
    HeuristicStrategy.SIZE_BASED: size_based_heuristic,
    HeuristicStrategy.COMPUTE_BOUND: compute_bound_heuristic,
    HeuristicStrategy.MEMORY_BOUND: memory_bound_heuristic,
    HeuristicStrategy.LATENCY_FOCUSED: latency_focused_heuristic,
}


# =============================================================================
# Main
# =============================================================================


def print_kernel_pool(kernels: List[KernelSpec]):
    """Print available kernels"""
    print("\n" + "=" * 75)
    print("  KERNEL POOL")
    print("=" * 75)
    print(f"\n  {'#':<3} {'Name':<22} {'Tile':<14} {'Pipeline':<10} {'Category':<12}")
    print("  " + "-" * 73)

    for i, k in enumerate(kernels, 1):
        tile = f"{k.tile_m}x{k.tile_n}x{k.tile_k}"
        print(f"  {i:<3} {k.name:<22} {tile:<14} {k.pipeline:<10} {k.category:<12}")

    print("  " + "-" * 73)


def main():
    parser = argparse.ArgumentParser(
        description="Custom Heuristics Example - intelligent kernel selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 08_heuristics.py                    # Default size-based heuristic
  python3 08_heuristics.py --strategy compute # Compute-bound heuristic
  python3 08_heuristics.py --strategy memory  # Memory-bound heuristic
  python3 08_heuristics.py --strategy latency # Latency-focused heuristic
  python3 08_heuristics.py --dtype bf16       # BF16 mode
        """,
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--strategy",
        default="size",
        choices=["size", "compute", "memory", "latency"],
        help="Heuristic strategy (default: size)",
    )
    parser.add_argument(
        "--arch",
        default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo, override with --arch gfxNNN)",
    )
    args = parser.parse_args()

    reset_for_example()

    print("=" * 75)
    print("Example 08: Custom Heuristics")
    print("=" * 75)

    # Map strategy string to enum
    strategy_map = {
        "size": HeuristicStrategy.SIZE_BASED,
        "compute": HeuristicStrategy.COMPUTE_BOUND,
        "memory": HeuristicStrategy.MEMORY_BOUND,
        "latency": HeuristicStrategy.LATENCY_FOCUSED,
    }
    strategy = strategy_map[args.strategy]
    heuristic_fn = HEURISTICS[strategy]

    print(f"\n  Strategy: {strategy.value}")
    print(f"  Data type: {args.dtype}")

    # Print kernel pool
    print_kernel_pool(KERNEL_POOL)

    # =========================================================================
    # Test heuristic selection across different problem sizes
    # =========================================================================
    print("\n" + "=" * 75)
    print("  HEURISTIC SELECTION TEST")
    print("=" * 75)

    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    test_sizes = [
        (128, 128, 64),  # Small
        (256, 256, 128),  # Small-medium
        (512, 512, 256),  # Medium
        (1024, 1024, 512),  # Medium-large
        (2048, 2048, 1024),  # Large
    ]

    print(
        f"\n  {'Size':<20} {'Selected Kernel':<25} {'Time (ms)':>10} {'TFLOPS':>10} {'Status':<8}"
    )
    print("  " + "-" * 78)

    results = []

    for M, N, K in test_sizes:
        # Use heuristic to select kernel
        selected_spec = heuristic_fn(M, N, K, KERNEL_POOL)

        # Create config and setup
        config = create_kernel_config(selected_spec, args.dtype, args.arch)

        setup = setup_gemm_dispatcher(
            config=config,
            registry_name=f"heuristic_{selected_spec.name}",
            verbose=False,
            auto_rebuild=True,
        )

        size_str = f"{M}x{N}x{K}"

        if not setup.success:
            print(
                f"  {size_str:<20} {selected_spec.name:<25} {'N/A':>10} {'N/A':>10} {'FAIL':<8}"
            )
            results.append((size_str, selected_spec.name, False, 0, 0))
            cleanup_gemm()
            continue

        dispatcher = setup.dispatcher

        if not dispatcher.is_supported(M, N, K):
            print(
                f"  {size_str:<20} {selected_spec.name:<25} {'N/A':>10} {'N/A':>10} {'SKIP':<8}"
            )
            results.append((size_str, selected_spec.name, False, 0, 0))
            cleanup_gemm()
            continue

        # Run GEMM
        np.random.seed(42)
        A = (np.random.randn(M, K) * 0.1).astype(np_dtype)
        B = (np.random.randn(K, N) * 0.1).astype(np_dtype)

        result = dispatcher.run(A, B, M, N, K)

        if not result.success:
            print(
                f"  {size_str:<20} {selected_spec.name:<25} {'N/A':>10} {'N/A':>10} {'FAIL':<8}"
            )
            results.append((size_str, selected_spec.name, False, 0, 0))
            cleanup_gemm()
            continue

        # Validate
        C_ref = np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(np_dtype)
        max_err = np.max(np.abs(result.output - C_ref))
        passed = max_err < 1e-2

        status = "PASS" if passed else "FAIL"
        print(
            f"  {size_str:<20} {selected_spec.name:<25} {result.time_ms:>10.4f} {result.tflops:>10.2f} {status:<8}"
        )
        results.append(
            (size_str, selected_spec.name, passed, result.time_ms, result.tflops)
        )

        cleanup_gemm()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 75)
    print("  SUMMARY")
    print("=" * 75)

    passed = sum(1 for r in results if r[2])
    failed = len(results) - passed

    print(f"\n  Strategy: {strategy.value}")
    print(f"  Results: {passed}/{len(results)} tests passed")

    # Show kernel selection distribution
    kernel_usage = {}
    for r in results:
        kernel_usage[r[1]] = kernel_usage.get(r[1], 0) + 1

    print("\n  Kernel Selection Distribution:")
    for kernel, count in sorted(kernel_usage.items(), key=lambda x: -x[1]):
        print(f"    {kernel}: {count} times")

    if results:
        valid_results = [r for r in results if r[2]]
        if valid_results:
            avg_tflops = sum(r[4] for r in valid_results) / len(valid_results)
            print(f"\n  Average TFLOPS: {avg_tflops:.2f}")

    if failed == 0:
        print("\n  *** ALL TESTS PASSED ***")
    else:
        print(f"\n  *** {failed} TESTS FAILED ***")

    print("=" * 75)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
