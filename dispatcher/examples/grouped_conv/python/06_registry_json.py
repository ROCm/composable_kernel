#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 06: Registry, Heuristic Selection & JSON Export

Declares multiple kernel configurations with different tile sizes,
builds a registry, demonstrates heuristic runtime kernel selection,
JSON round-trip, and GPU execution.

Usage:
    python3 06_registry_json.py
    python3 06_registry_json.py --workers 4
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from grouped_conv_utils import (
    GroupedConvKernelConfig,
    GroupedConvProblem,
    GroupedConvRegistry,
    detect_gpu_arch,
)


def conv_heuristic(problem):
    spatial = problem.Ho * problem.Wo
    if spatial > 400:
        return ["256", "128", "64"]
    return ["64", "128", "256"]


def main():
    parser = argparse.ArgumentParser(description="Registry, Heuristic & JSON")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    arch = args.arch
    print("=" * 70)
    print("Example 06: Registry, Heuristic Selection & JSON Export")
    print("=" * 70)
    print(f"\n  Arch: {arch}, Dtype: {args.dtype}")

    # Step 1: Declare kernels with full explicit parameters
    print("\n--- Step 1: Declare Kernels + Build Registry ---")
    reg = GroupedConvRegistry("conv_tiles")

    reg.add(
        GroupedConvKernelConfig(
            variant="forward",
            ndim_spatial=2,
            arch=arch,
            dtype=args.dtype,
            tile_m=1,
            tile_n=256,
            tile_k=256,
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
            num_wave_groups=1,
            num_groups_to_merge=1,
        )
    )
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
            num_wave_groups=1,
            num_groups_to_merge=1,
        )
    )
    reg.add(
        GroupedConvKernelConfig(
            variant="forward",
            ndim_spatial=2,
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
            num_wave_groups=1,
            num_groups_to_merge=1,
        )
    )
    reg.print_registry()

    # Step 2: Heuristic kernel selection
    print("\n--- Step 2: Heuristic Kernel Selection ---")
    problems = [
        (
            "small_7x7",
            GroupedConvProblem(
                N=1,
                C=512,
                K=512,
                Hi=7,
                Wi=7,
                Y=3,
                X=3,
                pad_h=1,
                pad_w=1,
                direction="forward",
            ),
        ),
        (
            "medium_14x14",
            GroupedConvProblem(
                N=1,
                C=256,
                K=256,
                Hi=14,
                Wi=14,
                Y=3,
                X=3,
                pad_h=1,
                pad_w=1,
                direction="forward",
            ),
        ),
        (
            "large_56x56",
            GroupedConvProblem(
                N=1,
                C=64,
                K=128,
                Hi=56,
                Wi=56,
                Y=3,
                X=3,
                pad_h=1,
                pad_w=1,
                direction="forward",
            ),
        ),
    ]
    print(f"  {'Problem':<16} {'Spatial':>8} {'Selected Kernel':<50}")
    print(f"  {'-' * 74}")
    for label, prob in problems:
        selected = reg.select(prob, heuristic=conv_heuristic)
        spatial = prob.Ho * prob.Wo
        sel_name = selected.name if selected else "none"
        print(f"  {label:<16} {spatial:>8} {sel_name:<50}")

    # Step 3: JSON round-trip
    print("\n--- Step 3: JSON Round-Trip ---")
    json_str = reg.to_json()
    print(f"  Exported: {len(json_str)} bytes, {len(reg)} kernels")
    imported = GroupedConvRegistry.from_json(json_str)
    print(f"  Imported: {len(imported)} kernels")
    orig = reg.kernels[0]
    imp = imported.kernels[0]
    rt_ok = (
        orig.vector_size_a == imp.vector_size_a
        and orig.block_per_cu == imp.block_per_cu
        and orig.tile_n == imp.tile_n
    )
    print(f"  Full fields round-trip: {'OK' if rt_ok else 'FAIL'}")

    # Step 4: JIT build + GPU execution
    print("\n--- Step 4: JIT Build + GPU Execution ---")
    workers = args.workers if args.workers > 0 else None
    jit_reg = GroupedConvRegistry("jit_conv")
    jit_reg.add(
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
        )
    )
    t0 = time.perf_counter()
    runners = jit_reg.build(verbose=False, max_workers=workers)
    jit_s = time.perf_counter() - t0

    if ("forward", 2) not in runners:
        print("  JIT build failed")
        return 1
    runner = runners[("forward", 2)]
    print(f"  JIT build: {jit_s:.3f} s")
    print(f"  Library:   {runner.library_path}")

    prob = GroupedConvProblem(
        N=1, C=128, K=128, Hi=16, Wi=16, Y=3, X=3, pad_h=1, pad_w=1, direction="forward"
    )
    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32
    inp = np.random.uniform(-0.3, 0.3, prob.input_shape()).astype(np_dtype)
    wei = np.random.uniform(-0.3, 0.3, prob.weight_shape()).astype(np_dtype)
    res = runner.run(inp, wei, prob)
    runner.cleanup()

    if res.success:
        print(f"  Time:    {res.time_ms:.4f} ms")
        print(f"  TFLOPS:  {res.tflops:.2f}")
        print(f"  NonZero: {np.count_nonzero(res.output)}/{res.output.size}")

    gpu_ok = res.success
    print("\n" + "=" * 70)
    print(f"  Registry:   {len(reg)} kernels (3 tile configs)")
    print("  Heuristic:  spatial-based selection demonstrated")
    print(f"  JSON:       round-trip {'OK' if rt_ok else 'FAIL'}")
    print(f"  GPU:        {'OK' if gpu_ok else 'FAIL'}")
    print(f"  Status:     {'PASS' if gpu_ok and rt_ok else 'FAIL'}")
    print("=" * 70)
    return 0 if gpu_ok and rt_ok else 1


if __name__ == "__main__":
    sys.exit(main())
