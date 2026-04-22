#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 05: Multi-Problem GPU Benchmark

Declares kernels with explicit tile/wave/warp/pipeline parameters for
all directions, builds registries, JIT compiles, and benchmarks across
ResNet-like problem sizes with configurable warmup/repeat.

Usage:
    python3 05_benchmark.py
    python3 05_benchmark.py --warmup 3 --repeat 10
    python3 05_benchmark.py --workers 4
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


def compute_bytes(prob, dtype_bytes=2):
    in_elems = 1
    for d in prob.input_shape():
        in_elems *= d
    wei_elems = 1
    for d in prob.weight_shape():
        wei_elems *= d
    out_elems = 1
    for d in prob.output_shape():
        out_elems *= d
    return (in_elems + wei_elems + out_elems) * dtype_bytes


def main():
    parser = argparse.ArgumentParser(description="Multi-Problem GPU Benchmark")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=5, help="Benchmark iterations")
    parser.add_argument(
        "--workers", type=int, default=0, help="Max JIT workers (0=auto)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 05: Multi-Problem GPU Benchmark")
    print("=" * 70)
    print(f"\n  Arch: {args.arch}, Dtype: {args.dtype}")
    print(f"  Warmup: {args.warmup}, Repeat: {args.repeat}")

    # =========================================================================
    # Step 1: Declare all kernels with explicit parameters
    # =========================================================================
    print("\n--- Step 1: Declare Kernels ---")
    reg = GroupedConvRegistry("benchmark")

    # Forward 2D: compv4, 128x128 tile
    reg.add(
        GroupedConvKernelConfig(
            variant="forward",
            ndim_spatial=2,
            arch=args.arch,
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
    # Forward 3D: compv3, 64x64 tile
    reg.add(
        GroupedConvKernelConfig(
            variant="forward",
            ndim_spatial=3,
            arch=args.arch,
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
    # BwdData 2D: compv3, 128x128 tile
    reg.add(
        GroupedConvKernelConfig(
            variant="bwd_data",
            ndim_spatial=2,
            arch=args.arch,
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
    # BwdWeight 2D: compv3, 128x128 tile
    reg.add(
        GroupedConvKernelConfig(
            variant="bwd_weight",
            ndim_spatial=2,
            arch=args.arch,
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
    reg.print_registry()

    # =========================================================================
    # Step 2: JIT build
    # =========================================================================
    print("\n--- Step 2: JIT Build ---")
    workers = args.workers if args.workers > 0 else None
    t0 = time.perf_counter()
    runner_by_key = reg.build(verbose=False, max_workers=workers)
    jit_s = time.perf_counter() - t0

    for key in [("forward", 2), ("forward", 3), ("bwd_data", 2), ("bwd_weight", 2)]:
        tag = "OK" if key in runner_by_key else "FAILED"
        print(f"  {key[0]:12s} {key[1]}D: {tag}")
    print(f"  JIT build time: {jit_s:.3f} s")

    missing = [
        k
        for k in [("forward", 2), ("forward", 3), ("bwd_data", 2), ("bwd_weight", 2)]
        if k not in runner_by_key
    ]
    if missing:
        print(f"\n  ERROR: missing {missing}")
        return 1

    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    def bench_run(runner, inp, wei, prob):
        for _ in range(args.warmup):
            runner.run(inp, wei, prob)
        times = []
        for _ in range(args.repeat):
            r = runner.run(inp, wei, prob)
            if r.success:
                times.append(r.time_ms)
        if not times:
            return 0.0, 0.0
        return min(times), sum(times) / len(times)

    # =========================================================================
    # Step 3: 2D Forward benchmark
    # =========================================================================
    print("\n--- Step 3: Forward 2D Benchmark ---")
    print(
        f"{'Problem':<18} {'N':>3} {'C':>4} {'K':>4} {'H':>3} {'W':>3} "
        f"{'F':>3} {'Min(ms)':>9} {'Avg(ms)':>9} {'TFLOPS':>8} {'GB/s':>8}"
    )
    print("-" * 85)

    all_ok = True
    for label, n, c, k, h, w, y, x, s, p in [
        ("ResNet-stage2", 1, 64, 64, 56, 56, 3, 3, 1, 1),
        ("ResNet-stage3", 1, 128, 128, 28, 28, 3, 3, 1, 1),
        ("ResNet-stage4", 1, 256, 256, 14, 14, 3, 3, 1, 1),
        ("ResNet-stage5", 1, 512, 512, 7, 7, 3, 3, 1, 1),
        ("Pointwise-1x1", 1, 256, 256, 56, 56, 1, 1, 1, 0),
        ("Batch-8", 8, 64, 128, 56, 56, 3, 3, 1, 1),
        ("Batch-32", 32, 64, 128, 56, 56, 3, 3, 1, 1),
    ]:
        prob = GroupedConvProblem(
            N=n,
            C=c,
            K=k,
            Hi=h,
            Wi=w,
            Y=y,
            X=x,
            stride_h=s,
            stride_w=s,
            pad_h=p,
            pad_w=p,
            direction="forward",
        )
        inp = np.random.uniform(-0.3, 0.3, prob.input_shape()).astype(np_dtype)
        wei = np.random.uniform(-0.3, 0.3, prob.weight_shape()).astype(np_dtype)
        min_ms, avg_ms = bench_run(runner_by_key[("forward", 2)], inp, wei, prob)
        if avg_ms > 0:
            tflops = prob.flops / (avg_ms * 1e9)
            bw = compute_bytes(prob) / (avg_ms * 1e6)
            print(
                f"{label:<18} {n:>3} {c:>4} {k:>4} {h:>3} {w:>3} "
                f"{y}x{x} {min_ms:>9.4f} {avg_ms:>9.4f} {tflops:>8.2f} {bw:>8.1f}"
            )
        else:
            all_ok = False

    # =========================================================================
    # Step 4: 3D Forward
    # =========================================================================
    print("\n--- Step 4: Forward 3D ---")
    for label, n, c, k, d, h, w, z, y, x in [
        ("3D-small", 1, 64, 64, 8, 16, 16, 3, 3, 3),
        ("3D-medium", 1, 64, 128, 16, 32, 32, 3, 3, 3),
    ]:
        prob = GroupedConvProblem(
            N=n, C=c, K=k, Di=d, Hi=h, Wi=w, Z=z, Y=y, X=x, direction="forward"
        )
        inp = np.random.uniform(-0.3, 0.3, prob.input_shape()).astype(np_dtype)
        wei = np.random.uniform(-0.3, 0.3, prob.weight_shape()).astype(np_dtype)
        min_ms, avg_ms = bench_run(runner_by_key[("forward", 3)], inp, wei, prob)
        if avg_ms > 0:
            tflops = prob.flops / (avg_ms * 1e9)
            print(f"  {label:<14} {min_ms:.4f} / {avg_ms:.4f} ms  {tflops:.2f} TFLOPS")

    # =========================================================================
    # Step 5: Backward directions
    # =========================================================================
    print("\n--- Step 5: Backward Directions ---")
    for label, direction in [
        ("bwd_data ResNet-s3", "bwd_data"),
        ("bwd_weight ResNet-s3", "bwd_weight"),
    ]:
        prob = GroupedConvProblem(
            N=1,
            C=128,
            K=128,
            Hi=28,
            Wi=28,
            Y=3,
            X=3,
            stride_h=1,
            stride_w=1,
            pad_h=1,
            pad_w=1,
            direction=direction,
        )
        inp = np.random.uniform(-0.3, 0.3, prob.input_shape()).astype(np_dtype)
        wei = np.random.uniform(-0.3, 0.3, prob.weight_shape()).astype(np_dtype)
        min_ms, avg_ms = bench_run(runner_by_key[(direction, 2)], inp, wei, prob)
        if avg_ms > 0:
            tflops = prob.flops / (avg_ms * 1e9)
            print(
                f"  {label:<14} {direction:>12} {min_ms:.4f} / {avg_ms:.4f} ms  {tflops:.2f} TFLOPS"
            )

    for runner in runner_by_key.values():
        runner.cleanup()

    print("\n" + "=" * 70)
    print(f"  JIT build:  {jit_s:.3f} s")
    print(f"  Warmup: {args.warmup}, Repeat: {args.repeat}")
    print(f"  Status: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
