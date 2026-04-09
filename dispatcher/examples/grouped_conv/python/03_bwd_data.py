#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 03: Backward Data Convolution (2D + 3D)

dX = ConvBwdData(dY, W)

Declares backward-data kernels with explicit parameters,
builds a registry, JIT compiles, runs on GPU, and validates
against a CPU reference.

Usage:
    python3 03_bwd_data.py
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from grouped_conv_utils import (
    GroupedConvKernelConfig,
    GroupedConvProblem,
    GroupedConvRegistry,
    detect_gpu_arch,
)


def cpu_conv2d_bwd_data(dy, wei, prob):
    """CPU ref: compute dX from dY and W."""
    N, Ho, Wo, G, Kpg = dy.shape
    _, _, Y, X, C = wei.shape
    Hi, Wi = prob.Hi, prob.Wi
    dx = np.zeros((N, Hi, Wi, G, C), dtype=np.float32)
    for n in range(N):
        for g in range(G):
            for hi in range(Hi):
                for wi in range(Wi):
                    for c in range(C):
                        s = 0.0
                        for y in range(Y):
                            for x in range(X):
                                ho = hi + prob.pad_h - y
                                wo = wi + prob.pad_w - x
                                if ho % prob.stride_h == 0 and wo % prob.stride_w == 0:
                                    ho //= prob.stride_h
                                    wo //= prob.stride_w
                                    if 0 <= ho < Ho and 0 <= wo < Wo:
                                        for k in range(Kpg):
                                            s += float(dy[n, ho, wo, g, k]) * float(
                                                wei[g, k, y, x, c]
                                            )
                        dx[n, hi, wi, g, c] = s
    return dx


def main():
    parser = argparse.ArgumentParser(description="Backward Data (2D + 3D)")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    arch = args.arch
    print("=" * 70)
    print("Example 03: Backward Data Convolution (2D + 3D)")
    print("=" * 70)
    print(f"\n  Arch: {arch}, Dtype: {args.dtype}")
    print("  dX = ConvBwdData(dY, W)")

    # =========================================================================
    # Step 1: Declare bwd_data kernels
    # =========================================================================
    print("\n--- Step 1: Declare BwdData Kernels ---")
    reg = GroupedConvRegistry("bwd_data_conv")

    # BwdData 2D: compv3, 128x128 tile
    reg.add(
        GroupedConvKernelConfig(
            variant="bwd_data",
            ndim_spatial=2,
            arch=arch,
            dtype=args.dtype,
            tile_m=1,
            tile_n=128,
            tile_k=128,
            wave_m=2,
            wave_n=2,
            wave_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="compv3",
            scheduler="intrawave",
            epilogue="cshuffle",
            vector_size_a=4,
            vector_size_b=8,
            vector_size_c=8,
            block_per_cu=1,
        )
    )
    # BwdData 3D: compv3, 64x64 tile
    reg.add(
        GroupedConvKernelConfig(
            variant="bwd_data",
            ndim_spatial=3,
            arch=arch,
            dtype=args.dtype,
            tile_m=1,
            tile_n=64,
            tile_k=64,
            wave_m=1,
            wave_n=4,
            wave_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
            pipeline="compv3",
            scheduler="intrawave",
            epilogue="cshuffle",
            vector_size_a=4,
            vector_size_b=8,
            vector_size_c=8,
            block_per_cu=1,
        )
    )
    reg.print_registry()

    # =========================================================================
    # Step 2: JIT build
    # =========================================================================
    print("\n--- Step 2: JIT Build ---")
    workers = args.workers if args.workers > 0 else None
    t0 = time.perf_counter()
    runners = reg.build(verbose=False, max_workers=workers)
    jit_s = time.perf_counter() - t0
    print(f"  Built {len(runners)} runners in {jit_s:.1f}s")

    if ("bwd_data", 2) not in runners:
        print("  ERROR: bwd_data 2D JIT failed")
        return 1

    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    # =========================================================================
    # Step 3: BwdData 2D -- GPU + CPU reference
    # =========================================================================
    print("\n--- Step 3: Backward Data 2D ---")
    prob = GroupedConvProblem(
        N=1, C=32, K=32, Hi=8, Wi=8, Y=3, X=3, pad_h=1, pad_w=1, direction="bwd_data"
    )
    prob.print_problem()

    dy = np.random.uniform(-0.5, 0.5, prob.output_shape()).astype(np_dtype)
    w = np.random.uniform(-0.5, 0.5, prob.weight_shape()).astype(np_dtype)

    res = runners[("bwd_data", 2)].run(dy, w, prob)
    print(f"  Time:   {res.time_ms:.4f} ms")
    print(f"  TFLOPS: {res.tflops:.2f}")
    print(f"  NonZero: {np.count_nonzero(res.output)}/{res.output.size}")

    ref = cpu_conv2d_bwd_data(dy, w, prob)
    diff = np.abs(res.output.astype(np.float32) - ref)
    match_2d = np.allclose(res.output.astype(np.float32), ref, atol=0.1)
    print(f"  CPU ref: max_abs={diff.max():.6f}, match={match_2d}")

    # =========================================================================
    # Step 4: BwdData 3D -- GPU + non-zero check
    # =========================================================================
    ok_3d = True
    if ("bwd_data", 3) in runners:
        print("\n--- Step 4: Backward Data 3D ---")
        prob3 = GroupedConvProblem(
            N=1,
            C=32,
            K=32,
            Di=6,
            Hi=6,
            Wi=6,
            Z=3,
            Y=3,
            X=3,
            pad_d=1,
            pad_h=1,
            pad_w=1,
            direction="bwd_data",
        )
        dy3 = np.random.uniform(-0.5, 0.5, prob3.output_shape()).astype(np_dtype)
        w3 = np.random.uniform(-0.5, 0.5, prob3.weight_shape()).astype(np_dtype)
        res3 = runners[("bwd_data", 3)].run(dy3, w3, prob3)
        nz = np.count_nonzero(res3.output)
        ok_3d = res3.success and nz > 0
        print(f"  Time:   {res3.time_ms:.4f} ms, NonZero: {nz}/{res3.output.size}")

    for r in runners.values():
        r.cleanup()

    passed = res.success and match_2d and ok_3d
    print("\n" + "=" * 70)
    print(f"  BwdData 2D: {'PASS' if match_2d else 'FAIL'} (CPU validated)")
    print(f"  BwdData 3D: {'PASS' if ok_3d else 'FAIL'}")
    print(f"  Status:     {'PASS' if passed else 'FAIL'}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
