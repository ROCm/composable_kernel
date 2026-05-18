#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 07: Stress Test - Multiple FMHA Kernels with Validation

Generates many FmhaKernelSpec configurations across pipelines, head
dimensions, and data types, registers them in an FmhaRegistry, builds
all in parallel, and validates each against a CPU reference.

Usage:
    python3 07_stress_test.py
    python3 07_stress_test.py --help
    python3 07_stress_test.py --num-kernels 4
    python3 07_stress_test.py --workers 8
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaKernelSpec,
    FmhaProblem,
    FmhaRegistry,
    FmhaValidator,
    cpu_attention_fwd,
    spec_to_config,
    detect_gpu_arch,
)


# FmhaKernelSpec fields:
#   name      -- human-readable kernel identifier
#   hdim      -- head dimension (hdim_q = hdim_v)
#   pipeline  -- "qr_async" (async prefetch) or "qr" (synchronous)
#   tile_m0   -- Stage 0 tile along seqlen_q  (Q*K^T M dimension)
#   tile_n0   -- Stage 0 tile along seqlen_k  (Q*K^T N dimension)
#   tile_k0   -- Stage 0 tile along hdim_q    (Q*K^T K dimension)
KERNEL_SPECS: List[FmhaKernelSpec] = [
    # qr_async pipeline -- various tile sizes
    FmhaKernelSpec(
        name="qr_async_h128_t128",
        hdim=128,
        pipeline="qr_async",
        tile_m0=128,
        tile_n0=128,
        tile_k0=32,
    ),
    FmhaKernelSpec(
        name="qr_async_h128_t64",
        hdim=128,
        pipeline="qr_async",
        tile_m0=64,
        tile_n0=128,
        tile_k0=32,
    ),
    FmhaKernelSpec(
        name="qr_async_h64_t128",
        hdim=64,
        pipeline="qr_async",
        tile_m0=128,
        tile_n0=64,
        tile_k0=32,
    ),
    FmhaKernelSpec(
        name="qr_async_h64_t64",
        hdim=64,
        pipeline="qr_async",
        tile_m0=64,
        tile_n0=64,
        tile_k0=32,
    ),
    # qr pipeline -- various tile sizes
    FmhaKernelSpec(
        name="qr_h128_t128",
        hdim=128,
        pipeline="qr",
        tile_m0=128,
        tile_n0=128,
        tile_k0=32,
    ),
    FmhaKernelSpec(
        name="qr_h128_t64", hdim=128, pipeline="qr", tile_m0=64, tile_n0=128, tile_k0=32
    ),
    FmhaKernelSpec(
        name="qr_h64_t128", hdim=64, pipeline="qr", tile_m0=128, tile_n0=64, tile_k0=32
    ),
    FmhaKernelSpec(
        name="qr_h64_t64", hdim=64, pipeline="qr", tile_m0=64, tile_n0=64, tile_k0=32
    ),
]


def print_spec_table(specs: List[FmhaKernelSpec]):
    print(
        f"\n  {'#':<3} {'Name':<25} {'Pipeline':<12} {'Hdim':>5} "
        f"{'TileM':>6} {'TileN':>6} {'TileK':>6}"
    )
    print("  " + "-" * 70)
    for i, s in enumerate(specs, 1):
        print(
            f"  {i:<3} {s.name:<25} {s.pipeline:<12} {s.hdim:>5} "
            f"{s.tile_m0:>6} {s.tile_n0:>6} {s.tile_k0:>6}"
        )
    print("  " + "-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="FMHA Stress Test - multiple kernels with validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 07_stress_test.py                   # Test all kernels
  python3 07_stress_test.py --num-kernels 4   # First 4 only
  python3 07_stress_test.py --workers 8       # 8 parallel compile workers
        """,
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument(
        "--num-kernels", type=int, default=0, help="Number of kernels to test (0 = all)"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Max parallel build workers (0 = auto)"
    )
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 07: FMHA Stress Test - Multiple Kernels")
    print("=" * 70)

    specs = KERNEL_SPECS[: args.num_kernels] if args.num_kernels > 0 else KERNEL_SPECS

    print(f"\n  Arch:    {args.arch}")
    print(f"  Kernels: {len(specs)}")
    print_spec_table(specs)

    # Step 1: Register all in FmhaRegistry and build
    print("\n" + "=" * 70)
    print("  JIT BUILD")
    print("=" * 70)

    reg = FmhaRegistry("stress_test")
    for spec in specs:
        cfg = spec_to_config(spec, dtype="fp16", arch=args.arch)
        reg.register_kernel(cfg)

    workers = args.workers if args.workers > 0 else None
    print(f"\n  Building {len(reg)} kernels (workers={workers or 'auto'}) ...")

    t0 = time.perf_counter()
    build_results = reg.build(verbose=False, max_workers=workers)
    build_time = time.perf_counter() - t0

    built = sum(1 for r in build_results if r.success)
    print(f"  Built: {built}/{len(specs)} in {build_time:.1f} s")

    for i, r in enumerate(build_results, 1):
        tag = "OK" if r.success else f"FAIL: {r.error[:50]}"
        name = r.config.name if r.config else f"kernel_{i}"
        print(f"    [{i}] {name}: {tag}")

    if built == 0:
        print("\n  No kernels built -- aborting")
        return 1

    # Step 2: Validate each built kernel
    print("\n" + "=" * 70)
    print("  VALIDATION")
    print("=" * 70)

    prob = FmhaProblem(
        batch=2, nhead_q=4, nhead_k=4, seqlen_q=64, seqlen_k=64, hdim_q=128, hdim_v=128
    )

    np.random.seed(42)
    Q = (np.random.randn(*prob.q_shape()) * 0.5).astype(np.float16)
    K = (np.random.randn(*prob.k_shape()) * 0.5).astype(np.float16)
    V = (np.random.randn(*prob.v_shape()) * 0.5).astype(np.float16)
    O_ref = cpu_attention_fwd(
        Q.astype(np.float32), K.astype(np.float32), V.astype(np.float32), prob.scale
    )

    validator = FmhaValidator(rtol=args.rtol, atol=args.atol)

    print(
        f"\n  Problem: B={prob.batch} Hq={prob.nhead_q} Sq={prob.seqlen_q} D={prob.hdim_q}"
    )
    print(f"\n  {'#':<3} {'Name':<35} {'Time':>8} {'MaxErr':>10} {'Status':<6}")
    print("  " + "-" * 66)

    total_pass = 0
    total_fail = 0

    for i, r in enumerate(build_results, 1):
        name = r.config.name if r.config else f"kernel_{i}"

        if not r.success or r.runner is None:
            print(f"  {i:<3} {name:<35} {'---':>8} {'---':>10} {'SKIP':<6}")
            continue

        hdim = r.config.hdim_q if r.config else 128
        if hdim != prob.hdim_q:
            print(f"  {i:<3} {name:<35} {'---':>8} {'---':>10} {'SKIP':<6}")
            continue

        res = r.runner.run(Q, K, V, prob)
        if not res.success:
            print(f"  {i:<3} {name:<35} {'---':>8} {'---':>10} {'FAIL':<6}")
            total_fail += 1
            continue

        ok, max_abs, _ = validator.check(res.output, O_ref)
        tag = "PASS" if ok else "FAIL"
        print(f"  {i:<3} {name:<35} {res.time_ms:>7.4f}ms {max_abs:>10.2e} {tag:<6}")

        if ok:
            total_pass += 1
        else:
            total_fail += 1

        r.runner.cleanup()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Total:      {len(specs)}")
    print(f"  Built:      {built}")
    print(f"  Passed:     {total_pass}")
    print(f"  Failed:     {total_fail}")
    print(f"  Build time: {build_time:.1f} s")
    print(f"  Tolerance:  rtol={args.rtol}, atol={args.atol}")

    if total_fail == 0 and total_pass > 0:
        print("\n  *** ALL VALIDATED KERNELS PASSED ***")
    elif total_fail > 0:
        print(f"\n  *** {total_fail} KERNELS FAILED ***")

    print("=" * 70)

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
