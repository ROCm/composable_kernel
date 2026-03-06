#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 01: Basic GEMM with Multiple Kernels

Demonstrates:
1. Declaring multiple kernel configurations
2. Printing all registered kernels
3. Running each kernel and validating output
4. Comparing performance across kernels

Complexity: ★★☆☆☆

Usage:
    python3 01_basic_gemm.py
    python3 01_basic_gemm.py --help
    python3 01_basic_gemm.py --dtype bf16
    python3 01_basic_gemm.py --size 2048
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
    reset_for_example,
)


@dataclass
class KernelSpec:
    """Specification for a kernel configuration"""

    name: str
    tile_m: int
    tile_n: int
    tile_k: int
    pipeline: str = "compv3"
    scheduler: str = "intrawave"


# Define multiple kernel configurations to test (50+ kernels)
KERNEL_SPECS = [
    # Small tiles - compv3
    KernelSpec("small_64x64_k32", 64, 64, 32, "compv3"),
    KernelSpec("small_64x64_k64", 64, 64, 64, "compv3"),
    # Small tiles - compv4
    KernelSpec("small_64x64_v4_k32", 64, 64, 32, "compv4"),
    KernelSpec("small_64x64_v4_k64", 64, 64, 64, "compv4"),
    # Medium tiles - compv3
    KernelSpec("med_128x128_k32", 128, 128, 32, "compv3"),
    KernelSpec("med_128x128_k64", 128, 128, 64, "compv3"),
    KernelSpec("med_128x128_k128", 128, 128, 128, "compv3"),
    # Medium tiles - compv4
    KernelSpec("med_128x128_v4_k32", 128, 128, 32, "compv4"),
    KernelSpec("med_128x128_v4_k64", 128, 128, 64, "compv4"),
    KernelSpec("med_128x128_v4_k128", 128, 128, 128, "compv4"),
    # Rectangular tiles - compv3
    KernelSpec("rect_64x128_k32", 64, 128, 32, "compv3"),
    KernelSpec("rect_64x128_k64", 64, 128, 64, "compv3"),
    KernelSpec("rect_128x64_k32", 128, 64, 32, "compv3"),
    KernelSpec("rect_128x64_k64", 128, 64, 64, "compv3"),
    # Rectangular tiles - compv4
    KernelSpec("rect_64x128_v4_k32", 64, 128, 32, "compv4"),
    KernelSpec("rect_64x128_v4_k64", 64, 128, 64, "compv4"),
    KernelSpec("rect_128x64_v4_k32", 128, 64, 32, "compv4"),
    KernelSpec("rect_128x64_v4_k64", 128, 64, 64, "compv4"),
    # Large tiles - compv3
    KernelSpec("large_256x128_k32", 256, 128, 32, "compv3"),
    KernelSpec("large_256x128_k64", 256, 128, 64, "compv3"),
    KernelSpec("large_128x256_k32", 128, 256, 32, "compv3"),
    KernelSpec("large_128x256_k64", 128, 256, 64, "compv3"),
    KernelSpec("large_256x256_k32", 256, 256, 32, "compv3"),
    KernelSpec("large_256x256_k64", 256, 256, 64, "compv3"),
    # Large tiles - compv4
    KernelSpec("large_256x128_v4_k32", 256, 128, 32, "compv4"),
    KernelSpec("large_256x128_v4_k64", 256, 128, 64, "compv4"),
    KernelSpec("large_128x256_v4_k32", 128, 256, 32, "compv4"),
    KernelSpec("large_128x256_v4_k64", 128, 256, 64, "compv4"),
    KernelSpec("large_256x256_v4_k32", 256, 256, 32, "compv4"),
    KernelSpec("large_256x256_v4_k64", 256, 256, 64, "compv4"),
    # Interwave scheduler variants
    KernelSpec("int_64x64_k32", 64, 64, 32, "compv3", "interwave"),
    KernelSpec("int_128x128_k32", 128, 128, 32, "compv3", "interwave"),
    KernelSpec("int_128x128_k64", 128, 128, 64, "compv3", "interwave"),
    KernelSpec("int_256x128_k32", 256, 128, 32, "compv3", "interwave"),
    # More tile_k variations - compv3
    KernelSpec("med_128x128_k16", 128, 128, 16, "compv3"),
    KernelSpec("rect_64x128_k16", 64, 128, 16, "compv3"),
    KernelSpec("rect_128x64_k16", 128, 64, 16, "compv3"),
    # More tile_k variations - compv4
    KernelSpec("med_128x128_v4_k16", 128, 128, 16, "compv4"),
    KernelSpec("rect_64x128_v4_k16", 64, 128, 16, "compv4"),
    KernelSpec("rect_128x64_v4_k16", 128, 64, 16, "compv4"),
    # Additional rectangular
    KernelSpec("rect_32x64_k32", 32, 64, 32, "compv3"),
    KernelSpec("rect_64x32_k32", 64, 32, 32, "compv3"),
    KernelSpec("rect_32x128_k32", 32, 128, 32, "compv3"),
    KernelSpec("rect_128x32_k32", 128, 32, 32, "compv3"),
    # Additional compv4 variants
    KernelSpec("rect_32x64_v4_k32", 32, 64, 32, "compv4"),
    KernelSpec("rect_64x32_v4_k32", 64, 32, 32, "compv4"),
    KernelSpec("rect_32x128_v4_k32", 32, 128, 32, "compv4"),
    KernelSpec("rect_128x32_v4_k32", 128, 32, 32, "compv4"),
]


def create_kernel_config(spec: KernelSpec, dtype: str, arch: str) -> KernelConfig:
    """Create a KernelConfig from a spec"""
    # Adjust warp tiles based on tile size
    if spec.tile_m <= 64:
        warp_m, warp_n = 16, 16
    else:
        warp_m, warp_n = 32, 32

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


def print_kernel_table(specs: List[KernelSpec], dtype: str):
    """Print a formatted table of kernel configurations"""
    print("\n" + "=" * 70)
    print(f"  DECLARED KERNEL CONFIGURATIONS ({len(specs)} kernels)")
    print("=" * 70)
    print(f"\n  {'#':<3} {'Name':<18} {'Tile':<14} {'Pipeline':<10} {'Scheduler':<12}")
    print("  " + "-" * 68)

    for i, spec in enumerate(specs, 1):
        tile = f"{spec.tile_m}x{spec.tile_n}x{spec.tile_k}"
        print(
            f"  {i:<3} {spec.name:<18} {tile:<14} {spec.pipeline:<10} {spec.scheduler:<12}"
        )

    print("  " + "-" * 68)
    print(f"  Data type: {dtype}")


def main():
    parser = argparse.ArgumentParser(
        description="Basic GEMM Example with Multiple Kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 01_basic_gemm.py                    # Default FP16 with 4 kernels
  python3 01_basic_gemm.py --dtype bf16       # BF16 mode
  python3 01_basic_gemm.py --size 2048        # Larger problem size
  python3 01_basic_gemm.py --num-kernels 2    # Test only 2 kernels
        """,
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--arch",
        default="gfx942",
        help="Target architecture (default: gfx942)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Problem size MxNxK (default: 512)",
    )
    parser.add_argument(
        "--num-kernels",
        type=int,
        default=0,
        help="Number of kernels to test (0 = all)",
    )
    args = parser.parse_args()

    reset_for_example()

    print("=" * 70)
    print("Example 01: Basic GEMM with Multiple Kernels")
    print("=" * 70)

    # Select kernels to test
    specs = KERNEL_SPECS[: args.num_kernels] if args.num_kernels > 0 else KERNEL_SPECS

    # =========================================================================
    # Step 1: Print all kernel configurations
    # =========================================================================
    print_kernel_table(specs, args.dtype)

    # =========================================================================
    # Step 2: Setup and test each kernel
    # =========================================================================
    print("\n" + "=" * 70)
    print("  RUNNING KERNELS")
    print("=" * 70)

    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32
    M, N, K = args.size, args.size, args.size

    results = []

    print(f"\n  Problem size: {M}x{N}x{K}\n")
    print(
        f"  {'#':<3} {'Name':<18} {'Tile':<14} {'Time (ms)':>10} {'TFLOPS':>10} {'Max Err':>10} {'Status':<8}"
    )
    print("  " + "-" * 78)

    for i, spec in enumerate(specs, 1):
        # Create unique test data per kernel
        np.random.seed(42 + i * 1000)
        A = (np.random.randn(M, K) * 0.1).astype(np_dtype)
        B = (np.random.randn(K, N) * 0.1).astype(np_dtype)

        # Create config and setup dispatcher
        config = create_kernel_config(spec, args.dtype, args.arch)

        setup = setup_gemm_dispatcher(
            config=config,
            registry_name=f"kernel_{spec.name}",
            verbose=False,
            auto_rebuild=True,
        )

        tile = f"{spec.tile_m}x{spec.tile_n}x{spec.tile_k}"

        if not setup.success:
            print(
                f"  {i:<3} {spec.name:<18} {tile:<14} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'FAIL':<8}"
            )
            results.append((spec.name, False, 0, 0, 0))
            cleanup_gemm()
            continue

        dispatcher = setup.dispatcher

        # Check if size is supported
        if not dispatcher.is_supported(M, N, K):
            print(
                f"  {i:<3} {spec.name:<18} {tile:<14} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'SKIP':<8}"
            )
            results.append((spec.name, False, 0, 0, 0))
            cleanup_gemm()
            continue

        # Run GEMM
        result = dispatcher.run(A, B, M, N, K)

        if not result.success:
            print(
                f"  {i:<3} {spec.name:<18} {tile:<14} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'FAIL':<8}"
            )
            results.append((spec.name, False, 0, 0, 0))
            cleanup_gemm()
            continue

        # Validate against NumPy reference
        C_ref = np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(np_dtype)
        max_err = np.max(np.abs(result.output - C_ref))

        # Check if within tolerance
        passed = max_err < 1e-2
        status = "PASS" if passed else "FAIL"

        print(
            f"  {i:<3} {spec.name:<18} {tile:<14} {result.time_ms:>10.4f} {result.tflops:>10.2f} {max_err:>10.2e} {status:<8}"
        )
        results.append((spec.name, passed, result.time_ms, result.tflops, max_err))

        cleanup_gemm()

    # =========================================================================
    # Step 3: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r[1])
    failed = len(results) - passed

    print(f"\n  Results: {passed}/{len(results)} kernels passed")
    print(f"  Problem: {M}x{N}x{K}, dtype={args.dtype}")

    if results:
        valid_results = [r for r in results if r[1]]
        if valid_results:
            best = max(valid_results, key=lambda x: x[3])
            print(f"\n  Best kernel: {best[0]} ({best[3]:.2f} TFLOPS)")

    if failed == 0:
        print("\n  *** ALL KERNELS PASSED ***")
    else:
        print(f"\n  *** {failed} KERNELS FAILED ***")

    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
