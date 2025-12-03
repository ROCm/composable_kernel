#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 10: 3D Convolution Forward with GPU Execution

Demonstrates 3D convolution (e.g., for video or volumetric data) with GPU execution
and kernel configuration validation.

Usage:
    python3 10_conv3d_forward.py
    python3 10_conv3d_forward.py --dtype bf16
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from conv_utils import (
    ConvSignature,
    ConvAlgorithm,
    ArchInfo,
    ConvKernelSet,
    ConvProblem,
    GpuConvRunner,
    validate_conv_config,
    auto_correct_conv_config,
    reset_for_conv_example,
    cleanup_conv,
    print_conv_kernel_config,
    print_conv_auto_correction,
)


def main():
    parser = argparse.ArgumentParser(description="3D Convolution Forward Example")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="compv3",
        choices=["compv3", "compv4", "mem"],
        help="Pipeline version (default: compv3)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="intrawave",
        choices=["intrawave", "interwave"],
        help="Scheduler (default: intrawave)",
    )
    parser.add_argument("--tile-k", type=int, default=128, help="Tile K size")
    parser.add_argument("--tile-c", type=int, default=128, help="Tile C size")
    parser.add_argument(
        "--arch", type=str, default="gfx942", help="Target architecture"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 10: 3D Convolution Forward with GPU Execution")
    print("=" * 70)
    print()

    # =========================================================================
    # Step 0: Reset state for clean example run
    # =========================================================================
    reset_for_conv_example(verbose=True)

    # =========================================================================
    # Step 1: Define 3D kernel configuration
    # =========================================================================
    print("\nStep 1: Define 3D Kernel Configuration")
    print("-" * 50)

    sig = ConvSignature()
    sig.dtype(args.dtype, args.dtype, args.dtype, "fp32")
    sig.layout = "ndhwgc"
    sig.direction = "forward"
    sig.num_dims = 3  # 3D convolution

    algo = ConvAlgorithm()
    algo.tile(1, args.tile_k, args.tile_c)
    algo.wave(2, 2, 1)
    algo.warp(32, 32, 16)
    algo.pipeline = args.pipeline
    algo.scheduler = args.scheduler

    arch = ArchInfo(name=args.arch)

    # Print the EXACT configuration requested
    print_conv_kernel_config(sig, algo, arch, "REQUESTED KERNEL CONFIGURATION")

    # =========================================================================
    # Step 2: Validate and auto-correct configuration
    # =========================================================================
    print("Step 2: Validate Config Against Arch Filter")
    print("-" * 50)

    validation = validate_conv_config(
        pipeline=algo.pipeline,
        scheduler=algo.scheduler,
        epilogue=algo.epilogue,
        wave_m=algo.wave_m,
        wave_n=algo.wave_n,
        wave_k=algo.wave_k,
        warp_m=algo.warp_m,
        warp_n=algo.warp_n,
        warp_k=algo.warp_k,
        dtype=sig.dtype_in,
        arch=arch.name,
    )
    validation.print_result()

    if not validation.is_valid:
        print("\n  ⚠ Auto-correcting configuration...")
        corrected, was_modified, corrections = auto_correct_conv_config(
            pipeline=algo.pipeline,
            scheduler=algo.scheduler,
            epilogue=algo.epilogue,
            wave_m=algo.wave_m,
            wave_n=algo.wave_n,
            wave_k=algo.wave_k,
            warp_m=algo.warp_m,
            warp_n=algo.warp_n,
            warp_k=algo.warp_k,
            dtype=sig.dtype_in,
            arch=arch.name,
        )
        if was_modified:
            print_conv_auto_correction(corrections)
            algo.scheduler = corrected["scheduler"]
            algo.wave_m = corrected["wave_m"]
            algo.wave_n = corrected["wave_n"]
            algo.warp_m = corrected["warp_m"]
            algo.warp_n = corrected["warp_n"]
            algo.warp_k = corrected["warp_k"]
            print_conv_kernel_config(sig, algo, arch, "CORRECTED KERNEL CONFIGURATION")
    print()

    # =========================================================================
    # Step 3: Create kernel set
    # =========================================================================
    print("Step 3: Create Kernel Set")
    print("-" * 50)

    kernel_set = ConvKernelSet("conv3d_fwd_kernels")
    kernel_set.add(sig, algo, arch)

    print(f"  Kernel Set: {kernel_set.name}")
    print(f"  Configurations: {len(kernel_set.configs)}")
    for cfg in kernel_set.configs:
        print(f"    - {cfg.name()}")
    print()

    # =========================================================================
    # Step 4: Define 3D problem
    # =========================================================================
    print("Step 4: Define 3D Problem")
    print("-" * 50)

    # 3D problem: N=1, C=32, K=64, D=8, H=16, W=16, filter 3x3x3
    problem = ConvProblem(
        N=1,
        C=32,
        K=64,
        Di=8,
        Hi=16,
        Wi=16,  # 3D spatial dimensions
        Z=3,
        Y=3,
        X=3,  # 3D filter
        pad_d=1,
        pad_h=1,
        pad_w=1,
        stride_d=1,
        stride_h=1,
        stride_w=1,
        direction="forward",
    )

    print(f"  N={problem.N}, C={problem.C}, K={problem.K}")
    print(f"  Input (3D): {problem.Di}x{problem.Hi}x{problem.Wi}")
    print(f"  Filter (3D): {problem.Z}x{problem.Y}x{problem.X}")
    print(f"  Output (3D): {problem.Do}x{problem.Ho}x{problem.Wo}")
    print(f"  FLOPs: {problem.flops_3d:.2e}")
    print()

    # =========================================================================
    # Step 5: Generate test data
    # =========================================================================
    print("Step 5: Generate Test Data")
    print("-" * 50)

    np_dtype = {
        "fp16": np.float16,
        "bf16": np.float16,
        "fp32": np.float32,
    }[args.dtype]

    # 3D tensor sizes (NDHWGC layout)
    input_host = np.random.randn(
        problem.N, problem.Di, problem.Hi, problem.Wi, problem.G, problem.C
    ).astype(np_dtype)
    weight_host = np.random.randn(
        problem.G, problem.K, problem.Z, problem.Y, problem.X, problem.C
    ).astype(np_dtype)

    print(f"  Input (3D):  {input_host.shape} ({np_dtype.__name__})")
    print(f"  Weight (3D): {weight_host.shape} ({np_dtype.__name__})")
    print()

    # =========================================================================
    # Step 6: GPU Execution
    # =========================================================================
    print("Step 6: GPU Execution")
    print("-" * 50)

    runner = GpuConvRunner()
    if runner.is_available():
        print(f"  Library: {runner.library_path}")
        print(f"  Input (3D):  {input_host.shape} -> GPU")
        print(f"  Weight (3D): {weight_host.shape} -> GPU")

        # Run 3D convolution
        result = runner.run(input_host, weight_host, problem)

        if result.get("success"):
            print("\n  *** 3D CONV GPU EXECUTION SUCCESSFUL ***")
            print(f"  Time:   {result['time_ms']:.4f} ms")
            print(f"  TFLOPS: {result['tflops']:.2f}")
        else:
            print(f"  [GPU execution returned: {result.get('error', 'unknown')}]")

        runner.cleanup()
    else:
        print("  [Dispatcher library not found]")
        print(
            "  Build with: cd dispatcher/build && cmake .. && make dispatcher_conv_lib"
        )

    # =========================================================================
    # Cleanup and Summary
    # =========================================================================
    cleanup_conv()

    print()
    print("=" * 70)
    print("SUMMARY: 3D Convolution")
    print("=" * 70)
    print(f"  Kernel:  {args.dtype} {sig.direction} {sig.num_dims}D")
    print(f"  Config:  tile={args.tile_k}x{args.tile_c}, pipeline={args.pipeline}")
    print("  Use for: video, medical imaging, volumetric data")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
