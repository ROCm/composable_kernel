#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 01: Basic FMHA with Multiple Kernels

Demonstrates:
1. Building a Registry with multiple kernel configurations
2. Parallel JIT compilation via registry.build()
3. Running each kernel and validating output against CPU reference
4. Comparing performance across kernels

Usage:
    python3 01_basic_fmha.py
    python3 01_basic_fmha.py --dtype bf16
    python3 01_basic_fmha.py --size 256
    python3 01_basic_fmha.py --num-kernels 4
    python3 01_basic_fmha.py --workers 4
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaKernelSpec,
    FmhaRegistry,
    FmhaProblem,
    cpu_attention_fwd,
    detect_gpu_arch,
    spec_to_config,
)


# FmhaKernelSpec fields:
#   name      -- human-readable kernel identifier
#   hdim      -- head dimension (hdim_q = hdim_v for symmetric attention)
#   pipeline  -- "qr_async" (async prefetch) or "qr" (synchronous)
#   tile_m0   -- Stage 0 tile along seqlen_q  (Q*K^T M dimension)
#   tile_n0   -- Stage 0 tile along seqlen_k  (Q*K^T N dimension)
#   tile_k0   -- Stage 0 tile along hdim_q    (Q*K^T K dimension)
#
# spec_to_config() fills in Stage 1 automatically:
#   tile_n1 = hdim, tile_k1 = tile_k0, tile_k0max = hdim
#   wave/warp use sensible defaults (4x1x1 wave, 32x32x16 warp)
KERNEL_SPECS = [
    # Async pipelines -- different tile_m0 x tile_n0 combos
    FmhaKernelSpec(
        name="async_128x128_k32",
        hdim=128,
        pipeline="qr_async",
        tile_m0=128,
        tile_n0=128,
        tile_k0=32,
    ),
    FmhaKernelSpec(
        name="async_128x64_k32",
        hdim=128,
        pipeline="qr_async",
        tile_m0=128,
        tile_n0=64,
        tile_k0=32,
    ),
    FmhaKernelSpec(
        name="async_64x128_k32",
        hdim=128,
        pipeline="qr_async",
        tile_m0=64,
        tile_n0=128,
        tile_k0=32,
    ),
    FmhaKernelSpec(
        name="async_64x64_k32",
        hdim=128,
        pipeline="qr_async",
        tile_m0=64,
        tile_n0=64,
        tile_k0=32,
    ),
    # Synchronous pipelines
    FmhaKernelSpec(
        name="sync_128x128_k32",
        hdim=128,
        pipeline="qr",
        tile_m0=128,
        tile_n0=128,
        tile_k0=32,
    ),
    FmhaKernelSpec(
        name="sync_64x128_k32",
        hdim=128,
        pipeline="qr",
        tile_m0=64,
        tile_n0=128,
        tile_k0=32,
    ),
    FmhaKernelSpec(
        name="sync_128x64_k32",
        hdim=128,
        pipeline="qr",
        tile_m0=128,
        tile_n0=64,
        tile_k0=32,
    ),
    # Different tile_k0 (K dimension of Q*K^T)
    FmhaKernelSpec(
        name="async_128x128_k64",
        hdim=128,
        pipeline="qr_async",
        tile_m0=128,
        tile_n0=128,
        tile_k0=64,
    ),
    FmhaKernelSpec(
        name="async_64x128_k64",
        hdim=128,
        pipeline="qr_async",
        tile_m0=64,
        tile_n0=128,
        tile_k0=64,
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Basic FMHA with Multiple Kernels")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--size", type=int, default=128, help="Sequence length")
    parser.add_argument("--num-kernels", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--workers", type=int, default=0, help="Max parallel JIT workers (0 = auto)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 01: Basic FMHA with Multiple Kernels")
    print("=" * 70)

    specs = KERNEL_SPECS[: args.num_kernels] if args.num_kernels > 0 else KERNEL_SPECS

    # Step 1: Build registry
    print(
        f"\n  {len(specs)} kernel configurations, dtype={args.dtype}, arch={args.arch}"
    )
    print(f"\n  {'#':<3} {'Name':<24} {'Tile':<14} {'Pipeline':<12}")
    print("  " + "-" * 56)
    for i, s in enumerate(specs, 1):
        print(
            f"  {i:<3} {s.name:<24} {s.tile_m0}x{s.tile_n0}x{s.tile_k0:<6} {s.pipeline:<12}"
        )

    reg = FmhaRegistry(name="basic_fmha")
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
    seqlen = args.size
    prob = FmhaProblem(
        batch=2,
        nhead_q=8,
        nhead_k=8,
        seqlen_q=seqlen,
        seqlen_k=seqlen,
        hdim_q=128,
        hdim_v=128,
    )

    print(
        f"\n--- Running Kernels (B={prob.batch} H={prob.nhead_q} S={seqlen} D={prob.hdim_q}) ---"
    )

    np.random.seed(42)
    Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float16)
    K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float16)
    V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float16)
    O_ref = cpu_attention_fwd(
        Q.astype(np.float32),
        K.astype(np.float32),
        V.astype(np.float32),
        prob.scale,
    )

    print(
        f"\n  {'#':<3} {'Name':<24} {'Pipeline':<12} {'Time(ms)':>10} {'TFLOPS':>10} {'MaxErr':>10} {'Status':<6}"
    )
    print("  " + "-" * 80)

    results = []
    for i, (spec, setup) in enumerate(zip(specs, setups), 1):
        if not setup.success or setup.runner is None:
            print(
                f"  {i:<3} {spec.name:<24} {spec.pipeline:<12} {'---':>10} {'---':>10} {'---':>10} {'SKIP':<6}"
            )
            results.append((spec.name, False, 0.0, 0.0, 0.0))
            continue

        res = setup.runner.run(Q, K, V, prob)
        if not res.success:
            print(
                f"  {i:<3} {spec.name:<24} {spec.pipeline:<12} {'---':>10} {'---':>10} {'---':>10} {'FAIL':<6}"
            )
            results.append((spec.name, False, 0.0, 0.0, 0.0))
            continue

        max_err = float(np.abs(res.output.astype(np.float32) - O_ref).max())
        ok = max_err < 1e-2
        tag = "PASS" if ok else "FAIL"
        print(
            f"  {i:<3} {spec.name:<24} {spec.pipeline:<12} {res.time_ms:>10.4f} {res.tflops:>10.2f} {max_err:>10.2e} {tag:<6}"
        )
        results.append((spec.name, ok, res.time_ms, res.tflops, max_err))
        setup.runner.cleanup()

    # Step 4: Summary
    passed = sum(1 for r in results if r[1])
    failed = len(results) - passed
    valid = [r for r in results if r[1]]

    print("\n" + "=" * 70)
    print(f"  Results:  {passed}/{len(results)} passed")
    print(
        f"  Problem:  B={prob.batch} H={prob.nhead_q} S={seqlen} D={prob.hdim_q}, dtype={args.dtype}"
    )
    print(f"  JIT time: {jit_build_s:.1f} s (parallel)")
    if valid:
        best = max(valid, key=lambda x: x[3])
        print(f"  Best:     {best[0]} ({best[3]:.2f} TFLOPS)")
    print(f"  Status:   {'PASS' if failed == 0 else 'FAIL'}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
