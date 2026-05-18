#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 07: Stress Test - Multiple Kernels with Validation

Consolidated stress test that:
1. Declares multiple kernel configurations (various tiles, pipelines, layouts)
2. Prints all registered kernels with details
3. Validates each kernel against NumPy reference
4. Optional benchmarking mode

This tests:
- Multiple tile sizes (64x64, 128x128, 256x256)
- Multiple pipelines (compv3, compv4)
- Multiple data types (fp16, bf16)
- Different schedulers (intrawave, interwave)


Usage:
    python3 07_stress_test.py
    python3 07_stress_test.py --help
    python3 07_stress_test.py --num-kernels 10
    python3 07_stress_test.py --benchmark
    python3 07_stress_test.py --dtype bf16
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
    Validator,
    detect_gpu_arch,
)


@dataclass
class KernelSpec:
    """A kernel specification for testing"""

    name: str
    tile_m: int
    tile_n: int
    tile_k: int
    wave_m: int = 2
    wave_n: int = 2
    wave_k: int = 1
    warp_m: int = 32
    warp_n: int = 32
    warp_k: int = 16
    pipeline: str = "compv3"
    scheduler: str = "intrawave"
    layout: str = "rcr"

    def to_config(self, dtype: str, arch: str) -> KernelConfig:
        """Convert to KernelConfig"""
        # Adjust warp tiles for smaller tiles
        warp_m = min(self.warp_m, self.tile_m // self.wave_m)
        warp_n = min(self.warp_n, self.tile_n // self.wave_n)
        warp_k = self.warp_k

        return KernelConfig(
            dtype_a=dtype,
            dtype_b=dtype,
            dtype_c=dtype,
            dtype_acc="fp32",
            layout_a={"r": "row", "c": "col"}[self.layout[0]],
            layout_b={"r": "row", "c": "col"}[self.layout[1]],
            layout_c={"r": "row", "c": "col"}[self.layout[2]],
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            wave_m=self.wave_m,
            wave_n=self.wave_n,
            wave_k=self.wave_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=warp_k,
            pipeline=self.pipeline,
            scheduler=self.scheduler,
            epilogue="cshuffle",
            gfx_arch=arch,
        )


# Define stress test kernel configurations
KERNEL_SPECS = [
    # Small tiles - compv3
    KernelSpec(
        "small_compv3",
        64,
        64,
        32,
        wave_m=2,
        wave_n=2,
        warp_m=16,
        warp_n=16,
        warp_k=32,
        pipeline="compv3",
    ),
    KernelSpec(
        "small_compv4",
        64,
        64,
        32,
        wave_m=2,
        wave_n=2,
        warp_m=16,
        warp_n=16,
        warp_k=32,
        pipeline="compv4",
    ),
    # Medium tiles
    KernelSpec(
        "medium_compv3",
        128,
        128,
        32,
        wave_m=2,
        wave_n=2,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        pipeline="compv3",
    ),
    KernelSpec(
        "medium_compv4",
        128,
        128,
        32,
        wave_m=2,
        wave_n=2,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        pipeline="compv4",
    ),
    KernelSpec(
        "medium_k64",
        128,
        128,
        64,
        wave_m=2,
        wave_n=2,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        pipeline="compv3",
    ),
    # Rectangular tiles
    KernelSpec(
        "rect_64x128",
        64,
        128,
        32,
        wave_m=2,
        wave_n=2,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        pipeline="compv3",
    ),
    KernelSpec(
        "rect_128x64",
        128,
        64,
        32,
        wave_m=2,
        wave_n=2,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        pipeline="compv3",
    ),
    # Different schedulers
    KernelSpec(
        "interwave",
        128,
        128,
        32,
        wave_m=2,
        wave_n=2,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        pipeline="compv3",
        scheduler="interwave",
    ),
    # Large tiles
    KernelSpec(
        "large_compv3",
        256,
        128,
        32,
        wave_m=2,
        wave_n=2,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        pipeline="compv3",
    ),
    KernelSpec(
        "large_compv4",
        256,
        128,
        64,
        wave_m=2,
        wave_n=2,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        pipeline="compv4",
    ),
]


def print_kernel_summary(specs: List[KernelSpec], dtype: str):
    """Print a summary table of all kernel specs"""
    print("\n" + "=" * 80)
    print(f"  DECLARED KERNEL CONFIGURATIONS ({len(specs)} kernels)")
    print("=" * 80)
    print(
        f"\n  {'#':<3} {'Name':<18} {'Tile':<12} {'Wave':<10} {'Warp':<12} {'Pipeline':<10} {'Sched':<10}"
    )
    print("  " + "-" * 78)

    for i, spec in enumerate(specs, 1):
        tile = f"{spec.tile_m}x{spec.tile_n}x{spec.tile_k}"
        wave = f"{spec.wave_m}x{spec.wave_n}x{spec.wave_k}"
        warp = f"{spec.warp_m}x{spec.warp_n}x{spec.warp_k}"
        print(
            f"  {i:<3} {spec.name:<18} {tile:<12} {wave:<10} {warp:<12} {spec.pipeline:<10} {spec.scheduler:<10}"
        )

    print("  " + "-" * 78)
    print(f"  Data type: {dtype}\n")


def validate_kernel(
    spec: KernelSpec,
    dtype: str,
    arch: str,
    size: int,
    validator: Validator,
    kernel_index: int = 0,
    verbose: bool = False,
) -> Tuple[bool, float, str]:
    """
    Validate a single kernel configuration.
    Returns: (passed, max_error, message)
    """
    np_dtype = np.float16 if dtype in ["fp16", "bf16"] else np.float32

    # Create config
    config = spec.to_config(dtype, arch)

    # Setup dispatcher
    setup = setup_gemm_dispatcher(
        config=config,
        registry_name=f"stress_{spec.name}",
        verbose=False,
        auto_rebuild=True,
    )

    if not setup.success:
        return False, 0.0, f"Setup failed: {setup.error}"

    dispatcher = setup.dispatcher
    M, N, K = size, size, size

    if not dispatcher.is_supported(M, N, K):
        cleanup_gemm()
        return False, 0.0, f"Size {M}x{N}x{K} not supported"

    # Use different seed per kernel to get unique test data
    # This ensures each kernel is tested with different matrices
    np.random.seed(42 + kernel_index * 1000)
    A = (np.random.randn(M, K) * 0.1).astype(np_dtype)
    B = (np.random.randn(K, N) * 0.1).astype(np_dtype)

    # Run GPU GEMM
    result = dispatcher.run(A, B, M, N, K)

    if not result.success:
        cleanup_gemm()
        return False, 0.0, "GPU execution failed"

    # Validate against NumPy
    C_ref = np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(np_dtype)
    is_valid, max_err, _ = validator.check(result.output, C_ref)

    cleanup_gemm()

    return is_valid, max_err, f"{result.time_ms:.2f}ms, {result.tflops:.1f} TFLOPS"


def benchmark_kernel(
    spec: KernelSpec,
    dtype: str,
    arch: str,
    size: int,
    warmup: int = 3,
    iterations: int = 10,
) -> Tuple[bool, float, float]:
    """
    Benchmark a kernel configuration.
    Returns: (success, avg_time_ms, tflops)
    """
    np_dtype = np.float16 if dtype in ["fp16", "bf16"] else np.float32

    config = spec.to_config(dtype, arch)
    setup = setup_gemm_dispatcher(
        config=config,
        registry_name=f"bench_{spec.name}",
        verbose=False,
        auto_rebuild=True,
    )

    if not setup.success:
        return False, 0.0, 0.0

    dispatcher = setup.dispatcher
    M, N, K = size, size, size

    if not dispatcher.is_supported(M, N, K):
        cleanup_gemm()
        return False, 0.0, 0.0

    A = (np.random.randn(M, K) * 0.1).astype(np_dtype)
    B = (np.random.randn(K, N) * 0.1).astype(np_dtype)

    # Warmup
    for _ in range(warmup):
        dispatcher.run(A, B, M, N, K)

    # Benchmark
    times = []
    for _ in range(iterations):
        result = dispatcher.run(A, B, M, N, K)
        if result.success:
            times.append(result.time_ms)

    cleanup_gemm()

    if not times:
        return False, 0.0, 0.0

    avg_time = sum(times) / len(times)
    tflops = (2.0 * M * N * K / (avg_time * 1e-3)) / 1e12

    return True, avg_time, tflops


def main():
    parser = argparse.ArgumentParser(
        description="GEMM Stress Test - Multiple kernels with validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 07_stress_test.py                    # Test all kernels
  python3 07_stress_test.py --num-kernels 5    # Test first 5 kernels
  python3 07_stress_test.py --benchmark        # Include benchmarks
  python3 07_stress_test.py --dtype bf16       # Test BF16
  python3 07_stress_test.py --size 2048        # Use 2048x2048 matrices
        """,
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--num-kernels",
        type=int,
        default=0,
        help="Number of kernels to test (0 = all)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Problem size MxNxK (default: 512)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Include benchmark timing",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance (default: 1e-2)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance (default: 1e-2)",
    )
    parser.add_argument(
        "--arch",
        default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo, override with --arch gfxNNN)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Example 07: GEMM Stress Test - Multiple Kernels")
    print("=" * 80)

    # Select kernels to test
    specs = KERNEL_SPECS[: args.num_kernels] if args.num_kernels > 0 else KERNEL_SPECS

    # Print kernel summary
    print_kernel_summary(specs, args.dtype)

    # Run validation
    print("\n" + "=" * 80)
    print("  VALIDATION RESULTS")
    print("=" * 80)

    validator = Validator(rtol=args.rtol, atol=args.atol)

    if args.benchmark:
        print(
            f"\n  {'#':<3} {'Name':<18} {'Tile':<12} {'Max Err':>10} {'Time':>10} {'TFLOPS':>8} {'Status':<8}"
        )
    else:
        print(
            f"\n  {'#':<3} {'Name':<18} {'Tile':<12} {'Max Err':>10} {'Info':<25} {'Status':<8}"
        )
    print("  " + "-" * 78)

    passed = 0
    failed = 0
    skipped = 0

    for i, spec in enumerate(specs, 1):
        tile = f"{spec.tile_m}x{spec.tile_n}x{spec.tile_k}"

        try:
            is_valid, max_err, info = validate_kernel(
                spec, args.dtype, args.arch, args.size, validator, kernel_index=i
            )

            if is_valid:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1

            if args.benchmark:
                success, avg_time, tflops = benchmark_kernel(
                    spec, args.dtype, args.arch, args.size
                )
                if success:
                    print(
                        f"  {i:<3} {spec.name:<18} {tile:<12} {max_err:>10.2e} {avg_time:>9.2f}ms {tflops:>7.1f} {status:<8}"
                    )
                else:
                    print(
                        f"  {i:<3} {spec.name:<18} {tile:<12} {max_err:>10.2e} {'N/A':>10} {'N/A':>8} {status:<8}"
                    )
            else:
                print(
                    f"  {i:<3} {spec.name:<18} {tile:<12} {max_err:>10.2e} {info:<25} {status:<8}"
                )

        except Exception as e:
            skipped += 1
            print(
                f"  {i:<3} {spec.name:<18} {tile:<12} {'N/A':>10} {str(e)[:25]:<25} {'SKIP':<8}"
            )

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    total = passed + failed + skipped
    print(f"\n  Results: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    print(f"  Settings: dtype={args.dtype}, size={args.size}x{args.size}x{args.size}")
    print(f"  Tolerance: rtol={args.rtol}, atol={args.atol}")
    print(f"  Architecture: {args.arch}")

    if failed == 0 and skipped == 0:
        print("\n  *** ALL KERNELS PASSED ***")
    elif failed > 0:
        print(f"\n  *** {failed} KERNELS FAILED ***")

    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
