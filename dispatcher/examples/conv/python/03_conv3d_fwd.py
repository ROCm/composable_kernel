#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 03: 3D Convolution Forward (Python)

Demonstrates 3D forward convolution using the Signature/Algorithm/Arch pattern.

Usage:
    python3 03_conv3d_fwd.py
    python3 03_conv3d_fwd.py --verify
    python3 03_conv3d_fwd.py --dtype bf16
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
)


def reference_conv3d_fwd(input_np, weight_np, stride=1, pad=0):
    """Simple CPU reference for 3D convolution forward."""
    N, Di, Hi, Wi, G, C = input_np.shape
    _, K, Z, Y, X, _ = weight_np.shape

    Do = (Di + 2 * pad - Z) // stride + 1
    Ho = (Hi + 2 * pad - Y) // stride + 1
    Wo = (Wi + 2 * pad - X) // stride + 1

    if pad > 0:
        input_padded = np.pad(
            input_np,
            ((0, 0), (pad, pad), (pad, pad), (pad, pad), (0, 0), (0, 0)),
            mode="constant",
        )
    else:
        input_padded = input_np

    output = np.zeros((N, Do, Ho, Wo, G, K), dtype=np.float32)

    for n in range(N):
        for g in range(G):
            for k in range(K):
                for do in range(Do):
                    for ho in range(Ho):
                        for wo in range(Wo):
                            acc = 0.0
                            for c in range(C):
                                for z in range(Z):
                                    for y in range(Y):
                                        for x in range(X):
                                            di = do * stride + z
                                            hi = ho * stride + y
                                            wi = wo * stride + x
                                            acc += float(
                                                input_padded[n, di, hi, wi, g, c]
                                            ) * float(weight_np[g, k, z, y, x, c])
                            output[n, do, ho, wo, g, k] = acc

    return output.astype(input_np.dtype)


def main():
    parser = argparse.ArgumentParser(description="3D Convolution Forward Example")
    parser.add_argument("-n", type=int, default=1, help="Batch size")
    parser.add_argument("-c", type=int, default=16, help="Input channels")
    parser.add_argument("-k", type=int, default=32, help="Output channels")
    parser.add_argument("-d", type=int, default=8, help="Input depth/height/width")
    parser.add_argument("-z", type=int, default=3, help="Filter depth/height/width")
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
    parser.add_argument("--tile-k", type=int, default=64, help="Tile K size")
    parser.add_argument("--tile-c", type=int, default=64, help="Tile C size")
    parser.add_argument(
        "--arch", type=str, default="gfx942", help="Target architecture"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 03: 3D Convolution Forward (Signature/Algorithm/Arch Pattern)")
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

    N, G, C, K = args.n, 1, args.c, args.k
    Di, Hi, Wi = args.d, args.d, args.d
    Z, Y, X = args.z, args.z, args.z
    stride, pad = 1, 1

    problem = ConvProblem(
        N=N,
        G=G,
        C=C,
        K=K,
        Di=Di,
        Hi=Hi,
        Wi=Wi,
        Z=Z,
        Y=Y,
        X=X,
        stride_d=stride,
        stride_h=stride,
        stride_w=stride,
        pad_d=pad,
        pad_h=pad,
        pad_w=pad,
        direction="forward",
    )

    print(f"  Batch:    N={problem.N}, G={problem.G}")
    print(f"  Channels: C={problem.C}, K={problem.K}")
    print(f"  Input:    Di={problem.Di}, Hi={problem.Hi}, Wi={problem.Wi}")
    print(f"  Filter:   Z={problem.Z}, Y={problem.Y}, X={problem.X}")
    print(f"  Output:   Do={problem.Do}, Ho={problem.Ho}, Wo={problem.Wo}")
    print(f"  FLOPs:    {problem.flops_3d:.2e}")

    # =========================================================================
    # Step 2: Define kernel config (Signature/Algorithm/Arch)
    # =========================================================================
    print("\nStep 2: Define Kernel Config")
    print("-" * 50)

    sig = ConvSignature()
    sig.dtype(args.dtype, args.dtype, args.dtype, "fp32")
    sig.layout = "ndhwgc"
    sig.direction = "forward"
    sig.num_dims = 3
    sig.groups = G

    algo = ConvAlgorithm()
    algo.tile(1, args.tile_k, args.tile_c)
    algo.wave(2, 2, 1)
    algo.warp(16, 16, 32)
    algo.pipeline = args.pipeline
    algo.scheduler = args.scheduler

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

    kernel_set = ConvKernelSet("conv3d_fwd_set")
    kernel_set.add(sig, algo, arch)
    kernel_set.print()

    # =========================================================================
    # Step 5: Generate test data (NDHWGC layout)
    # =========================================================================
    print("\nStep 5: Generate Test Data")
    print("-" * 50)

    np_dtype = {
        "fp16": np.float16,
        "bf16": np.float16,
        "fp32": np.float32,
    }[args.dtype]

    input_np = np.random.uniform(-0.5, 0.5, (N, Di, Hi, Wi, G, C)).astype(np_dtype)
    weight_np = np.random.uniform(-0.5, 0.5, (G, K, Z, Y, X, C)).astype(np_dtype)

    print(f"  Input:  {input_np.shape} ({np_dtype.__name__})")
    print(f"  Weight: {weight_np.shape} ({np_dtype.__name__})")

    # =========================================================================
    # Step 6: CPU verification (optional)
    # =========================================================================
    if args.verify:
        print("\nStep 6: CPU Reference Verification")
        print("-" * 50)

        output_ref = reference_conv3d_fwd(input_np, weight_np, stride=stride, pad=pad)
        print(f"  Output shape: {output_ref.shape}")
        print(f"  Output range: [{output_ref.min():.4f}, {output_ref.max():.4f}]")
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

        result = runner.run(input_np, weight_np, problem)

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
    print("SUMMARY: 3D Convolution")
    print("=" * 70)
    print(f"  Kernel:  {args.dtype} {sig.direction} {sig.num_dims}D")
    print(f"  Config:  tile={args.tile_k}x{args.tile_c}, pipeline={args.pipeline}")
    print("  Use for: video, medical imaging, volumetric data")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
