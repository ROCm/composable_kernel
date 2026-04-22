#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 01: Basic GEMM with Multiple Kernels

Demonstrates:
1. Building a Registry with multiple kernel configurations
2. Parallel JIT compilation via registry.build()
3. Running each kernel and validating output against NumPy reference
4. Comparing performance across kernels

Usage:
    python3 01_basic_gemm.py
    python3 01_basic_gemm.py --dtype bf16
    python3 01_basic_gemm.py --size 2048
    python3 01_basic_gemm.py --num-kernels 4
    python3 01_basic_gemm.py --workers 4
"""

import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    Registry,
    detect_gpu_arch,
)


@dataclass
class KernelSpec:
    name: str
    tile_m: int
    tile_n: int
    tile_k: int
    pipeline: str = "compv3"
    scheduler: str = "intrawave"


KERNEL_SPECS = [
    # Small tiles
    KernelSpec("small_64x64_k32", 64, 64, 32, "compv3"),
    KernelSpec("small_64x64_k64", 64, 64, 64, "compv3"),
    KernelSpec("small_64x64_v4_k32", 64, 64, 32, "compv4"),
    # Medium tiles
    KernelSpec("med_128x128_k32", 128, 128, 32, "compv3"),
    KernelSpec("med_128x128_k64", 128, 128, 64, "compv3"),
    KernelSpec("med_128x128_v4_k32", 128, 128, 32, "compv4"),
    KernelSpec("med_128x128_v4_k64", 128, 128, 64, "compv4"),
    # Rectangular tiles
    KernelSpec("rect_64x128_k32", 64, 128, 32, "compv3"),
    KernelSpec("rect_64x128_k64", 64, 128, 64, "compv3"),
    KernelSpec("rect_128x64_k32", 128, 64, 32, "compv3"),
    KernelSpec("rect_128x64_k64", 128, 64, 64, "compv3"),
    KernelSpec("rect_64x128_v4_k32", 64, 128, 32, "compv4"),
    KernelSpec("rect_128x64_v4_k32", 128, 64, 32, "compv4"),
    # Large tiles
    KernelSpec("large_256x128_k32", 256, 128, 32, "compv3"),
    KernelSpec("large_128x256_k32", 128, 256, 32, "compv3"),
    KernelSpec("large_256x256_k32", 256, 256, 32, "compv3"),
    KernelSpec("large_256x128_v4_k32", 256, 128, 32, "compv4"),
    KernelSpec("large_256x256_v4_k32", 256, 256, 32, "compv4"),
    # Interwave scheduler
    KernelSpec("int_128x128_k32", 128, 128, 32, "compv3", "interwave"),
    KernelSpec("int_256x128_k32", 256, 128, 32, "compv3", "interwave"),
]


def spec_to_config(spec: KernelSpec, dtype: str, arch: str) -> KernelConfig:
    warp_m, warp_n = (16, 16) if spec.tile_m <= 64 else (32, 32)
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


def main():
    parser = argparse.ArgumentParser(description="Basic GEMM with Multiple Kernels")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--size", type=int, default=512, help="Problem size MxNxK")
    parser.add_argument("--num-kernels", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--workers", type=int, default=0, help="Max parallel JIT workers (0 = auto)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 01: Basic GEMM with Multiple Kernels")
    print("=" * 70)

    specs = KERNEL_SPECS[: args.num_kernels] if args.num_kernels > 0 else KERNEL_SPECS

    # Step 1: Build registry
    print(
        f"\n  {len(specs)} kernel configurations, dtype={args.dtype}, arch={args.arch}"
    )
    print(f"\n  {'#':<3} {'Name':<22} {'Tile':<14} {'Pipeline':<10} {'Scheduler':<12}")
    print("  " + "-" * 64)
    for i, s in enumerate(specs, 1):
        print(
            f"  {i:<3} {s.name:<22} {s.tile_m}x{s.tile_n}x{s.tile_k:<6} {s.pipeline:<10} {s.scheduler:<12}"
        )

    reg = Registry(name="basic_gemm")
    for s in specs:
        reg.register_kernel(spec_to_config(s, args.dtype, args.arch))

    # Step 2: Parallel JIT build via registry.build()
    workers = args.workers if args.workers > 0 else None
    print(
        f"\n--- Parallel JIT Build ({len(specs)} kernels{f', workers={workers}' if workers else ''}) ---"
    )

    t0 = time.perf_counter()
    setups = reg.build(verbose=False, max_workers=workers)
    jit_build_s = time.perf_counter() - t0

    built = sum(1 for s in setups if s.success)
    print(f"  Built: {built}/{len(specs)} kernels in {jit_build_s:.1f} s")

    if built == 0:
        print("  ERROR: No kernels built")
        return 1

    # Step 3: Run each kernel and validate
    print(f"\n--- Running Kernels (problem {args.size}x{args.size}x{args.size}) ---")
    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32
    M = N = K = args.size

    np.random.seed(42)
    A = (np.random.randn(M, K) * 0.1).astype(np_dtype)
    B = (np.random.randn(K, N) * 0.1).astype(np_dtype)
    C_ref = np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(np_dtype)

    print(
        f"\n  {'#':<3} {'Name':<22} {'Tile':<14} {'Time(ms)':>10} {'TFLOPS':>10} {'MaxErr':>10} {'Status':<6}"
    )
    print("  " + "-" * 80)

    results = []
    for i, (spec, setup) in enumerate(zip(specs, setups), 1):
        tile = f"{spec.tile_m}x{spec.tile_n}x{spec.tile_k}"

        if not setup.success:
            print(
                f"  {i:<3} {spec.name:<22} {tile:<14} {'---':>10} {'---':>10} {'---':>10} {'SKIP':<6}"
            )
            results.append((spec.name, False, 0.0, 0.0, 0.0))
            continue

        disp = setup.dispatcher
        if not disp.is_supported(M, N, K):
            print(
                f"  {i:<3} {spec.name:<22} {tile:<14} {'---':>10} {'---':>10} {'---':>10} {'SKIP':<6}"
            )
            results.append((spec.name, False, 0.0, 0.0, 0.0))
            continue

        res = disp.run(A, B, M, N, K)
        if not res.success:
            print(
                f"  {i:<3} {spec.name:<22} {tile:<14} {'---':>10} {'---':>10} {'---':>10} {'FAIL':<6}"
            )
            results.append((spec.name, False, 0.0, 0.0, 0.0))
            continue

        max_err = float(np.max(np.abs(res.output - C_ref)))
        ok = max_err < 1e-2
        tag = "PASS" if ok else "FAIL"
        print(
            f"  {i:<3} {spec.name:<22} {tile:<14} {res.time_ms:>10.4f} {res.tflops:>10.2f} {max_err:>10.2e} {tag:<6}"
        )
        results.append((spec.name, ok, res.time_ms, res.tflops, max_err))

    # Step 4: Summary
    passed = sum(1 for r in results if r[1])
    failed = len(results) - passed
    valid = [r for r in results if r[1]]

    print("\n" + "=" * 70)
    print(f"  Results:  {passed}/{len(results)} passed")
    print(f"  Problem:  {M}x{N}x{K}, dtype={args.dtype}")
    print(f"  JIT time: {jit_build_s:.1f} s (parallel)")
    if valid:
        best = max(valid, key=lambda x: x[3])
        print(f"  Best:     {best[0]} ({best[3]:.2f} TFLOPS)")
    print(f"  Status:   {'PASS' if failed == 0 else 'FAIL'}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
