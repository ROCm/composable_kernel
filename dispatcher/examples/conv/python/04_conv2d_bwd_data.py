#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 04: 2D Convolution Backward Data (Python)

Computes gradient w.r.t. input: dX = ConvBwdData(dY, W)
Uses the Signature/Algorithm/Arch pattern with validation.

Usage:
    python3 04_conv2d_bwd_data.py
    python3 04_conv2d_bwd_data.py --verify
    python3 04_conv2d_bwd_data.py --dtype bf16
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

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


def reference_conv2d_bwd_data(grad_output, weight, stride=1, pad=0, Hi=None, Wi=None):
    """
    CPU reference for conv backward data (gradient w.r.t. input).

    Matches CK Tile's reference_grouped_conv_bwd_data algorithm.
    For each input position (hi, wi), compute which output positions
    contributed to it and accumulate the gradients.
    """
    N, Ho, Wo, G, K = grad_output.shape
    G_w, K_w, Y, X, C = weight.shape  # GKYXC layout

    if Hi is None:
        Hi = (Ho - 1) * stride + Y - 2 * pad
    if Wi is None:
        Wi = (Wo - 1) * stride + X - 2 * pad

    grad_input = np.zeros((N, Hi, Wi, G, C), dtype=np.float32)

    # For each input position, find which output positions affect it
    for n in range(N):
        for g in range(G):
            for c in range(C):
                for hi in range(Hi):
                    for wi in range(Wi):
                        v_acc = 0.0
                        for y in range(Y):
                            # h_tmp = hi + pad - y (for stride=1, dilation=1)
                            h_tmp = hi + pad - y
                            if h_tmp % stride == 0:
                                ho = h_tmp // stride
                                if 0 <= ho < Ho:
                                    for x in range(X):
                                        w_tmp = wi + pad - x
                                        if w_tmp % stride == 0:
                                            wo = w_tmp // stride
                                            if 0 <= wo < Wo:
                                                for k in range(K):
                                                    v_acc += float(
                                                        grad_output[n, ho, wo, g, k]
                                                    ) * float(weight[g, k, y, x, c])
                        grad_input[n, hi, wi, g, c] = v_acc

    return grad_input.astype(grad_output.dtype)


def main():
    parser = argparse.ArgumentParser(description="2D Conv Backward Data Example")
    parser.add_argument("-n", type=int, default=1, help="Batch size")
    parser.add_argument("-c", type=int, default=64, help="Input channels")
    parser.add_argument("-k", type=int, default=128, help="Output channels")
    parser.add_argument("-hi", type=int, default=28, help="Input height")
    parser.add_argument("-wi", type=int, default=28, help="Input width")
    parser.add_argument("-y", type=int, default=3, help="Filter height")
    parser.add_argument("-x", type=int, default=3, help="Filter width")
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
    print("Example 04: 2D Conv Backward Data (Signature/Algorithm/Arch Pattern)")
    print("=" * 70)

    # =========================================================================
    # Step 0: Reset state for clean example run
    # =========================================================================
    reset_for_conv_example(verbose=True)

    # =========================================================================
    # Step 1: Define problem
    # =========================================================================
    print("\nStep 1: Define ConvProblem")
    print("-" * 50)

    N, G, C, K = args.n, 1, args.c, args.k
    Hi, Wi = args.hi, args.wi
    Y, X = args.y, args.x
    stride, pad = 1, 1

    Ho = (Hi + 2 * pad - Y) // stride + 1
    Wo = (Wi + 2 * pad - X) // stride + 1

    problem = ConvProblem(
        N=N,
        G=G,
        C=C,
        K=K,
        Hi=Hi,
        Wi=Wi,
        Y=Y,
        X=X,
        stride_h=stride,
        stride_w=stride,
        pad_h=pad,
        pad_w=pad,
        direction="bwd_data",
    )

    print("  Backward Data: dX = ConvBwdData(dY, W)")
    print(f"  dY (grad_output): (N={N}, Ho={Ho}, Wo={Wo}, G={G}, K={K})")
    print(f"  W (weight):       (G={G}, K={K}, Y={Y}, X={X}, C={C})")
    print(f"  dX (grad_input):  (N={N}, Hi={Hi}, Wi={Wi}, G={G}, C={C})")

    flops = 2 * N * G * C * Hi * Wi * K * Y * X
    print(f"  FLOPs: {flops:.2e}")

    # =========================================================================
    # Step 2: Define kernel config
    # =========================================================================
    print("\nStep 2: Define Kernel Config")
    print("-" * 50)

    sig = ConvSignature()
    sig.dtype(args.dtype, args.dtype, args.dtype, "fp32")
    sig.layout = "nhwgc"
    sig.direction = "bwd_data"
    sig.num_dims = 2
    sig.groups = G

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
    # Step 4: Create kernel set
    # =========================================================================
    print("Step 4: Create Kernel Set")
    print("-" * 50)

    kernel_set = ConvKernelSet("conv2d_bwd_data_set")
    kernel_set.add(sig, algo, arch)
    kernel_set.print()

    # =========================================================================
    # Step 5: Generate test data
    # =========================================================================
    print("\nStep 5: Generate Test Data")
    print("-" * 50)

    np_dtype = {
        "fp16": np.float16,
        "bf16": np.float16,
        "fp32": np.float32,
    }[args.dtype]

    grad_output = np.random.uniform(-0.5, 0.5, (N, Ho, Wo, G, K)).astype(np_dtype)
    weight = np.random.uniform(-0.5, 0.5, (G, K, Y, X, C)).astype(np_dtype)

    print(f"  grad_output: {grad_output.shape} ({np_dtype.__name__})")
    print(f"  weight:      {weight.shape} ({np_dtype.__name__})")

    # =========================================================================
    # Step 6: CPU verification (optional)
    # =========================================================================
    grad_input_cpu = None
    if args.verify:
        print("\nStep 6: CPU Reference Verification")
        print("-" * 50)

        grad_input_cpu = reference_conv2d_bwd_data(
            grad_output, weight, stride, pad, Hi, Wi
        )
        print(f"  grad_input shape: {grad_input_cpu.shape}")
        print(f"  Range: [{grad_input_cpu.min():.4f}, {grad_input_cpu.max():.4f}]")
        print(f"  CPU[0,0,0,0,0]: {float(grad_input_cpu[0, 0, 0, 0, 0]):.4f}")
        print("  CPU reference computed successfully!")

    # =========================================================================
    # Step 7: GPU Execution
    # =========================================================================
    print("\nStep 7: GPU Execution")
    print("-" * 50)

    runner = GpuConvRunner()
    if runner.is_available():
        print(f"  Library: {runner.library_path}")
        print(f"  grad_output: {grad_output.shape} -> GPU")
        print(f"  weight:      {weight.shape} -> GPU")

        # Allocate output array to get GPU results back
        grad_input_gpu = np.zeros((N, Hi, Wi, G, C), dtype=np_dtype)
        result = runner.run(grad_output, weight, problem, output_np=grad_input_gpu)

        if result.get("success"):
            print("\n  *** GPU EXECUTION SUCCESSFUL ***")
            print(f"  Time:   {result['time_ms']:.4f} ms")
            print(f"  TFLOPS: {result['tflops']:.2f}")
            print(f"  GPU[0,0,0,0,0]: {float(grad_input_gpu[0, 0, 0, 0, 0]):.4f}")

            # Compare GPU vs CPU if verification requested
            if args.verify and grad_input_cpu is not None:
                # Compute error metrics
                abs_diff = np.abs(
                    grad_input_gpu.astype(np.float32)
                    - grad_input_cpu.astype(np.float32)
                )
                max_abs = abs_diff.max()

                nonzero = np.abs(grad_input_cpu.astype(np.float32)) > 1e-6
                if np.any(nonzero):
                    rel_diff = abs_diff[nonzero] / np.abs(
                        grad_input_cpu.astype(np.float32)[nonzero]
                    )
                    max_rel = rel_diff.max()
                else:
                    max_rel = max_abs

                passed = max_rel < 0.05  # 5% tolerance for FP16
                print("\n  GPU vs CPU Validation:")
                print(f"    Max abs diff: {max_abs:.4e}")
                print(f"    Max rel diff: {max_rel:.4e}")
                print(f"    Status: {'PASSED' if passed else 'FAILED'}")
        else:
            print("  [NOTE] Backward data kernel not found")
            print("  See C++ example conv_10_bwd_data for GPU execution")

        runner.cleanup()
    else:
        print("  GPU library not available")

    # =========================================================================
    # Cleanup and Summary
    # =========================================================================
    cleanup_conv()

    print("\n" + "=" * 70)
    print("SUMMARY: Backward Data Convolution")
    print("=" * 70)
    print(f"  Kernel:  {args.dtype} {sig.direction} {sig.num_dims}D")
    print(f"  Config:  tile={args.tile_k}x{args.tile_c}, pipeline={args.pipeline}")
    print("  Purpose: Compute dL/dInput for backpropagation")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
