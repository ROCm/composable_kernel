#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 01: Basic Grouped Convolution

Demonstrates:
1. Three kernel configuration patterns (minimal, explicit, full ConvConfigBase)
2. Adding kernels to a registry
3. Validation and auto-correction
4. JIT compilation via registry.build()
5. GPU execution with CPU reference verification

Usage:
    python3 01_basic_grouped_conv.py
    python3 01_basic_grouped_conv.py --variant bwd_data
    python3 01_basic_grouped_conv.py --arch gfx942
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
    validate_grouped_conv_config,
    auto_correct_grouped_conv_config,
    detect_gpu_arch,
)


def cpu_conv2d_fwd(inp, wei, prob):
    """Naive CPU reference: 2D forward, NHWGC layout."""
    N, Hi, Wi, G, Cpg = inp.shape
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
                                hi = (
                                    ho * prob.stride_h
                                    - prob.pad_h
                                    + y * prob.dilation_h
                                )
                                wi = (
                                    wo * prob.stride_w
                                    - prob.pad_w
                                    + x * prob.dilation_w
                                )
                                if 0 <= hi < Hi and 0 <= wi < Wi:
                                    for c in range(Cpg):
                                        s += float(inp[n, hi, wi, g, c]) * float(
                                            wei[g, k, y, x, c]
                                        )
                        out[n, ho, wo, g, k] = s
    return out


def main():
    parser = argparse.ArgumentParser(description="Basic Grouped Conv Example")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument(
        "--variant", default="forward", choices=["forward", "bwd_data", "bwd_weight"]
    )
    parser.add_argument("--ndim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument(
        "--workers", type=int, default=0, help="Max JIT workers (0=auto)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 01: Basic Grouped Convolution")
    print("=" * 70)

    # =========================================================================
    # Step 1: Three kernel configuration patterns
    # =========================================================================
    print("\n--- Step 1: Kernel Configuration Patterns ---")

    # Tile constraint (TileGemmShape, see grouped_config_rules_default.get_tiles_for_variant):
    #   tile_m == wave_m * warp_tile_m   AND   LDS fits the pipeline limit
    #   (compv4 limit = 32768 B, default = 65536 B)

    # Pattern 1: MINIMAL -- only variant/dtype/arch + a valid tile/wave combo
    # (the auto-filled defaults need a matching tile_m to satisfy the constraint)
    config_minimal = GroupedConvKernelConfig(
        variant=args.variant,
        ndim_spatial=args.ndim,
        arch=args.arch,
        dtype=args.dtype,
        tile_m=64,
        tile_n=128,
        tile_k=64,
        pipeline="compv4",  # LDS = 64*64*2 + 128*64*2 = 24576 B (fits compv4 32 KiB)
        double_smem_buffer=True,  # required by compv4 pipeline (C++ static_assert)
    )
    print("\n  Pattern 1: MINIMAL (defaults auto-filled)")
    config_minimal.print_config(indent="    ")

    # Pattern 2: EXPLICIT tile/wave/warp -- user controls tiling strategy
    config_explicit = GroupedConvKernelConfig(
        variant=args.variant,
        ndim_spatial=args.ndim,
        arch=args.arch,
        dtype=args.dtype,
        tile_m=16,  # = wave_m(1) * warp_tile_m(16)
        tile_n=64,
        tile_k=128,
        wave_m=1,
        wave_n=4,
        wave_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
        pipeline="compv3",
        scheduler="intrawave",
        epilogue="cshuffle",
    )
    print("\n  Pattern 2: EXPLICIT tile/wave/warp")
    config_explicit.print_config(indent="    ")

    # Pattern 3: FULL ConvConfigBase -- every parameter specified
    config_full = GroupedConvKernelConfig(
        variant=args.variant,
        ndim_spatial=args.ndim,
        arch=args.arch,
        dtype=args.dtype,
        tile_m=64,  # = wave_m(2) * warp_tile_m(32)
        tile_n=128,
        tile_k=64,
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
    print("\n  Pattern 3: FULL (all ConvConfigBase fields)")
    config_full.print_config(indent="    ")

    # =========================================================================
    # Step 2: Build a registry with multiple configs
    # =========================================================================
    print("\n--- Step 2: Build Registry ---")
    registry = GroupedConvRegistry("basic_conv")
    registry.add(config_minimal)
    registry.add(config_explicit)
    registry.add(config_full)
    registry.print_registry()

    # =========================================================================
    # Step 3: Validate and auto-correct
    # =========================================================================
    print("\n--- Step 3: Validate & Auto-Correct ---")
    for i, cfg in enumerate(registry.kernels):
        result = validate_grouped_conv_config(cfg.to_dict())
        if result.is_valid:
            print(f"  Config [{i}] {cfg.tile_str}: VALID")
        else:
            print(f"  Config [{i}] {cfg.tile_str}: needs correction")
            corrected, result = auto_correct_grouped_conv_config(cfg.to_dict())
            print(f"    After correction: valid={result.is_valid}")

    # =========================================================================
    # Step 4: JIT compile via registry.build()
    # =========================================================================
    print("\n--- Step 4: JIT Build (via registry.build()) ---")

    # Use only the first config for the actual GPU run
    jit_reg = GroupedConvRegistry("jit")
    jit_reg.add(config_minimal)

    workers = args.workers if args.workers > 0 else None
    t0 = time.perf_counter()
    runners = jit_reg.build(verbose=False, max_workers=workers)
    jit_build_s = time.perf_counter() - t0

    key = (args.variant, args.ndim)
    if key not in runners:
        print("  JIT build failed")
        return 1
    runner = runners[key]
    print(f"  JIT build: {jit_build_s:.3f} s")
    print(f"  Library:   {runner.library_path}")
    print(f"  Kernels:   {runner.lib.kernel_names()}")

    # =========================================================================
    # Step 5: Define problem + GPU execution
    # =========================================================================
    print("\n--- Step 5: GPU Execution ---")
    prob = GroupedConvProblem(
        N=1,
        C=64,
        K=128,
        Hi=16,
        Wi=16,
        Y=3,
        X=3,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        direction=args.variant,
    )
    prob.print_problem()

    inp = np.random.uniform(-0.5, 0.5, prob.input_shape()).astype(np.float16)
    wei = np.random.uniform(-0.5, 0.5, prob.weight_shape()).astype(np.float16)

    res = runner.run(inp, wei, prob)
    if not res.success:
        print(f"  GPU execution failed: {res.error}")
        runner.cleanup()
        return 1

    print(f"  Time:   {res.time_ms:.4f} ms")
    print(f"  TFLOPS: {res.tflops:.2f}")
    print(
        f"  Output: shape={res.output.shape}, range=[{res.output.min():.3f}, {res.output.max():.3f}]"
    )

    # =========================================================================
    # Step 6: CPU reference (forward 2D only)
    # =========================================================================
    verified = False
    if args.variant == "forward" and args.ndim == 2:
        print("\n--- Step 6: CPU Reference Verification ---")
        ref = cpu_conv2d_fwd(inp, wei, prob)
        gpu_f32 = res.output.astype(np.float32)
        diff = np.abs(gpu_f32 - ref)
        max_abs = diff.max()
        max_rel = (diff / (np.abs(ref) + 1e-6)).max()
        match = np.allclose(gpu_f32, ref, atol=0.05, rtol=0.05)
        print(f"  max_abs_diff: {max_abs:.6f}")
        print(f"  max_rel_diff: {max_rel:.6f}")
        print(f"  Match: {match}")
        verified = match

    runner.cleanup()

    # Summary
    print("\n" + "=" * 70)
    status = (
        "PASS" if res.success and (verified or args.variant != "forward") else "FAIL"
    )
    print(f"  Status: {status}")
    print(
        f"  {config_minimal.name} | {prob.gflops:.2f} GFLOPs | {res.tflops:.2f} TFLOPS"
    )
    print(f"  JIT build time: {jit_build_s:.3f} s")
    print(f"  Registry: {len(registry)} configs (3 patterns demonstrated)")
    print("=" * 70)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
