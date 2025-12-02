#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 02: 2D Convolution Forward (Python)

Demonstrates generating and running 2D forward convolution using Python.
Uses conv_utils.py for Signature/Algorithm/Arch pattern with validation.

Usage:
    python3 02_conv2d_fwd.py
    python3 02_conv2d_fwd.py --verify
    python3 02_conv2d_fwd.py --dtype bf16 --arch gfx942
    python3 02_conv2d_fwd.py -n 2 -c 64 -k 128 -hi 56 -y 3
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
    ConvValidator,
    GpuConvRunner,
    validate_conv_config,
    auto_correct_conv_config,
    reset_for_conv_example,
    cleanup_conv,
    print_conv_kernel_config,
)


def main():
    parser = argparse.ArgumentParser(description="2D Convolution Forward Example")
    parser.add_argument("-n", type=int, default=1, help="Batch size")
    parser.add_argument("-g", type=int, default=1, help="Groups")
    parser.add_argument("-c", type=int, default=64, help="Input channels")
    parser.add_argument("-k", type=int, default=128, help="Output channels")
    parser.add_argument("-hi", type=int, default=28, help="Input height")
    parser.add_argument("-wi", type=int, default=28, help="Input width")
    parser.add_argument("-y", type=int, default=3, help="Filter height")
    parser.add_argument("-x", type=int, default=3, help="Filter width")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--pad", type=int, default=1, help="Padding")
    parser.add_argument("--verify", action="store_true", help="Run CPU verification")
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
        default="compv4",
        choices=["compv3", "compv4", "mem"],
        help="Pipeline version (default: compv4)",
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
    print("Example 02: 2D Convolution Forward (Signature/Algorithm/Arch Pattern)")
    print("=" * 70)

    # =========================================================================
    # Step 0: Reset state for clean example run
    # =========================================================================
    reset_for_conv_example(verbose=True)

    # =========================================================================
    # Step 1: Define problem using ConvProblem
    # =========================================================================
    print("\nStep 1: Define ConvProblem")
    print("-" * 50)

    problem = ConvProblem(
        N=args.n,
        G=args.g,
        C=args.c,
        K=args.k,
        Hi=args.hi,
        Wi=args.wi,
        Y=args.y,
        X=args.x,
        stride_h=args.stride,
        stride_w=args.stride,
        pad_h=args.pad,
        pad_w=args.pad,
        direction="forward",
    )

    print(f"  Batch:    N={problem.N}, G={problem.G}")
    print(f"  Channels: C={problem.C}, K={problem.K}")
    print(f"  Input:    Hi={problem.Hi}, Wi={problem.Wi}")
    print(f"  Filter:   Y={problem.Y}, X={problem.X}")
    print(f"  Output:   Ho={problem.Ho}, Wo={problem.Wo}")
    print(f"  FLOPs:    {problem.flops:.2e}")

    # =========================================================================
    # Step 2: Define kernel config using Signature/Algorithm/Arch
    # =========================================================================
    print("\nStep 2: Define Kernel Config (Signature/Algorithm/Arch)")
    print("-" * 50)

    sig = ConvSignature()
    sig.dtype(args.dtype, args.dtype, args.dtype, "fp32")
    sig.layout = "nhwgc"
    sig.direction = "forward"
    sig.num_dims = 2
    sig.groups = args.g

    algo = ConvAlgorithm()
    algo.tile(1, args.tile_k, args.tile_c)
    algo.tile_output(1, 16)
    algo.wave(2, 2, 1)
    algo.warp(32, 32, 16)
    algo.pipeline = args.pipeline
    algo.scheduler = args.scheduler
    algo.epilogue = "cshuffle"

    arch = ArchInfo(name=args.arch)

    # Print the EXACT configuration requested
    print_conv_kernel_config(sig, algo, arch, "REQUESTED KERNEL CONFIGURATION")

    # =========================================================================
    # Step 3: Validate and auto-correct configuration
    # =========================================================================
    print("Step 3: Validate Config Against Arch Filter")
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
        corrected, was_modified = auto_correct_conv_config(
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
            algo.scheduler = corrected["scheduler"]
            algo.wave_m = corrected["wave_m"]
            algo.wave_n = corrected["wave_n"]
            algo.warp_m = corrected["warp_m"]
            algo.warp_n = corrected["warp_n"]
            algo.warp_k = corrected["warp_k"]
            print_conv_kernel_config(sig, algo, arch, "CORRECTED KERNEL CONFIGURATION")
    print()

    # =========================================================================
    # Step 4: Create kernel set
    # =========================================================================
    print("Step 4: Create Kernel Set")
    print("-" * 50)

    kernel_set = ConvKernelSet("conv2d_fwd_set")
    kernel_set.add(sig, algo, arch)

    # Add additional tile sizes
    for tile_k, tile_c in [(64, 64), (256, 256)]:
        algo_variant = algo.copy()
        algo_variant.tile_k = tile_k
        algo_variant.tile_c = tile_c
        kernel_set.add(sig.copy(), algo_variant, arch)

    kernel_set.print()

    # =========================================================================
    # Step 5: Generate test data
    # =========================================================================
    print("\nStep 5: Generate Test Data")
    print("-" * 50)

    np_dtype = {
        "fp16": np.float16,
        "bf16": np.float16,  # bf16 uses float16 storage
        "fp32": np.float32,
    }[args.dtype]

    # NHWGC layout for grouped conv
    input_np = np.random.uniform(
        -0.5,
        0.5,
        (problem.N, problem.Hi, problem.Wi, problem.G, problem.C // problem.G),
    ).astype(np_dtype)

    # GKYXC layout for weights
    weight_np = np.random.uniform(
        -0.5,
        0.5,
        (
            problem.G,
            problem.K // problem.G,
            problem.Y,
            problem.X,
            problem.C // problem.G,
        ),
    ).astype(np_dtype)

    print(f"  Input:  {input_np.shape} ({np_dtype.__name__})")
    print(f"  Weight: {weight_np.shape} ({np_dtype.__name__})")

    # =========================================================================
    # Step 6: CPU verification (optional)
    # =========================================================================
    if args.verify:
        print("\nStep 6: CPU Reference Verification")
        print("-" * 50)

        validator = ConvValidator(rtol=1e-3, atol=1e-3)

        # Simple CPU reference
        output_ref = validator.reference_conv2d_forward(
            input_np.reshape(problem.N, problem.Hi, problem.Wi, -1),
            weight_np.reshape(problem.K, problem.Y, problem.X, -1),
            stride=(problem.stride_h, problem.stride_w),
            padding=(problem.pad_h, problem.pad_w),
        )

        print(f"  Output shape: {output_ref.shape}")
        print(f"  Output range: [{output_ref.min():.4f}, {output_ref.max():.4f}]")
        print(f"  Sample values: {output_ref[0, 0, 0, :4]}")
        print("  CPU reference computed successfully!")

    # =========================================================================
    # Step 7: GPU Execution
    # =========================================================================
    print("\nStep 7: GPU Execution")
    print("-" * 50)

    runner = GpuConvRunner()
    if runner.is_available():
        print(f"  Library: {runner.library_path}")
        print(f"  Input:  {input_np.shape} -> GPU")
        print(f"  Weight: {weight_np.shape} -> GPU")

        result = runner.run_forward(input_np, weight_np, problem)

        if result.get("success"):
            print("\n  *** GPU EXECUTION SUCCESSFUL ***")
            print(f"  Time:   {result['time_ms']:.4f} ms")
            print(f"  TFLOPS: {result['tflops']:.2f}")
        else:
            print(f"  Execution returned: {result.get('error', 'unknown')}")

        runner.cleanup()
    else:
        print("  GPU library not available")
        print(
            "  Build with: cd dispatcher/build && cmake .. && make dispatcher_conv_lib"
        )

    # =========================================================================
    # Cleanup and Summary
    # =========================================================================
    cleanup_conv()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Kernel:  {args.dtype} {sig.direction} {sig.num_dims}D")
    print(f"  Config:  tile={args.tile_k}x{args.tile_c}, pipeline={args.pipeline}")
    print(
        f"  Problem: N={problem.N}, C={problem.C}, K={problem.K}, {problem.Hi}x{problem.Wi}"
    )
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
