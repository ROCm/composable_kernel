#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 02: Forward Convolution (2D + 3D)

Declares forward kernels with explicit tile/wave/warp/pipeline parameters,
builds a registry, JIT compiles, runs on GPU, and validates against CPU reference.

Usage:
    python3 02_forward.py
    python3 02_forward.py --arch gfx942
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


def cpu_conv2d_fwd(inp, wei, prob):
    """Naive CPU reference: 2D forward, NHWGC layout."""
    N, Hi, Wi, G, C = inp.shape
    _, Kpg, Y, X, _ = wei.shape
    Ho, Wo = prob.Ho, prob.Wo
    out = np.zeros((N, Ho, Wo, G, Kpg), dtype=np.float32)
    for n in range(N):
        for g in range(G):
            for ho in range(Ho):
                for wo in range(Wo):
                    for k in range(Kpg):
                        s = 0.0
                        for y in range(Y):
                            for x in range(X):
                                hi = ho * prob.stride_h - prob.pad_h + y
                                wi = wo * prob.stride_w - prob.pad_w + x
                                if 0 <= hi < Hi and 0 <= wi < Wi:
                                    for c in range(C):
                                        s += float(inp[n, hi, wi, g, c]) * float(
                                            wei[g, k, y, x, c]
                                        )
                        out[n, ho, wo, g, k] = s
    return out


def main():
    parser = argparse.ArgumentParser(description="Forward Convolution (2D + 3D)")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument(
        "--workers", type=int, default=0, help="Max JIT workers (0=auto)"
    )
    args = parser.parse_args()

    arch = args.arch
    print("=" * 70)
    print("Example 02: Forward Convolution (2D + 3D)")
    print("=" * 70)
    print(f"\n  Arch: {arch}, Dtype: {args.dtype}")

    # =========================================================================
    # Step 1: Declare forward kernels with explicit parameters
    # =========================================================================
    print("\n--- Step 1: Declare Forward Kernels ---")
    reg = GroupedConvRegistry("forward_conv")

    # Forward 2D: compv4, 128x128 tile, wave 2x2x1, warp 32x32x16
    reg.add(
        GroupedConvKernelConfig(
            variant="forward",
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
            pipeline="compv4",
            scheduler="intrawave",
            epilogue="cshuffle",
            vector_size_a=4,
            vector_size_b=8,
            vector_size_c=8,
            block_per_cu=1,
        )
    )
    # Forward 3D: compv3, 64x64 tile, wave 1x4x1, warp 16x16x32
    reg.add(
        GroupedConvKernelConfig(
            variant="forward",
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
    # Step 2: JIT build via registry
    # =========================================================================
    print("\n--- Step 2: JIT Build ---")
    workers = args.workers if args.workers > 0 else None
    t0 = time.perf_counter()
    runners = reg.build(verbose=False, max_workers=workers)
    jit_s = time.perf_counter() - t0
    print(f"  Built {len(runners)} runners in {jit_s:.1f}s")

    for key in [("forward", 2), ("forward", 3)]:
        tag = "OK" if key in runners else "FAILED"
        print(f"  {key[0]} {key[1]}D: {tag}")

    if ("forward", 2) not in runners:
        print("  ERROR: forward 2D JIT failed")
        return 1

    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    # =========================================================================
    # Step 3: Forward 2D -- GPU + CPU reference
    # =========================================================================
    print("\n--- Step 3: Forward 2D ---")
    prob_2d = GroupedConvProblem(
        N=1, C=64, K=64, Hi=8, Wi=8, Y=3, X=3, pad_h=1, pad_w=1, direction="forward"
    )
    prob_2d.print_problem()

    x = np.random.uniform(-0.5, 0.5, prob_2d.input_shape()).astype(np_dtype)
    w = np.random.uniform(-0.5, 0.5, prob_2d.weight_shape()).astype(np_dtype)

    res = runners[("forward", 2)].run(x, w, prob_2d)
    print(f"  Time:   {res.time_ms:.4f} ms")
    print(f"  TFLOPS: {res.tflops:.2f}")
    print(
        f"  Output: shape={res.output.shape}, nonzero={np.count_nonzero(res.output)}/{res.output.size}"
    )

    ref = cpu_conv2d_fwd(x, w, prob_2d)
    diff = np.abs(res.output.astype(np.float32) - ref)
    match_2d = np.allclose(res.output.astype(np.float32), ref, atol=0.05)
    print(f"  CPU ref: max_abs={diff.max():.6f}, match={match_2d}")

    # =========================================================================
    # Step 4: Forward 3D -- GPU + non-zero check
    # =========================================================================
    ok_3d = True
    if ("forward", 3) in runners:
        print("\n--- Step 4: Forward 3D ---")
        prob_3d = GroupedConvProblem(
            N=1,
            C=64,
            K=64,
            Di=8,
            Hi=8,
            Wi=8,
            Z=3,
            Y=3,
            X=3,
            pad_d=1,
            pad_h=1,
            pad_w=1,
            direction="forward",
        )
        prob_3d.print_problem()

        x3 = np.random.uniform(-0.5, 0.5, prob_3d.input_shape()).astype(np_dtype)
        w3 = np.random.uniform(-0.5, 0.5, prob_3d.weight_shape()).astype(np_dtype)

        res3 = runners[("forward", 3)].run(x3, w3, prob_3d)
        nz = np.count_nonzero(res3.output)
        ok_3d = res3.success and nz > 0
        print(f"  Time:   {res3.time_ms:.4f} ms")
        print(f"  TFLOPS: {res3.tflops:.2f}")
        print(f"  NonZero: {nz}/{res3.output.size}")

    for r in runners.values():
        r.cleanup()

    passed = res.success and match_2d and ok_3d
    print("\n" + "=" * 70)
    print(f"  Forward 2D: {'PASS' if match_2d else 'FAIL'} (CPU validated)")
    print(f"  Forward 3D: {'PASS' if ok_3d else 'FAIL'} (non-zero check)")
    print(f"  JIT build:  {jit_s:.1f}s")
    print(f"  Status:     {'PASS' if passed else 'FAIL'}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
