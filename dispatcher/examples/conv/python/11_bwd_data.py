#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 11: Backward Data Convolution

Demonstrates the backward data gradient computation (dL/dInput) API
with kernel configuration validation.

Used during neural network backpropagation.

Usage:
    python3 11_bwd_data.py
    python3 11_bwd_data.py --dtype bf16
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


def conv2d_bwd_data_reference(
    grad_output: np.ndarray,
    weight: np.ndarray,
    input_shape: tuple,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
) -> np.ndarray:
    """
    CPU reference implementation for 2D backward data convolution.

    Computes dL/dInput = conv_transpose(dOutput, Weight)

    Args:
        grad_output: Gradient from next layer (N, Ho, Wo, G, K) - NHWGK layout
        weight: Filter weights (G, K, Y, X, C) - GKYXC layout
        input_shape: Original input shape (N, Hi, Wi, G, C)
        stride: (stride_h, stride_w)
        padding: (pad_h, pad_w)
        dilation: (dilation_h, dilation_w)

    Returns:
        grad_input: Input gradient (N, Hi, Wi, G, C) - NHWGC layout
    """
    N, Ho, Wo, G, K = grad_output.shape
    _, _, Y, X, C = weight.shape
    _, Hi, Wi, _, _ = input_shape
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    # Use float32 for accumulation
    grad_input = np.zeros((N, Hi, Wi, G, C), dtype=np.float32)

    # Backward data: transpose convolution
    for n in range(N):
        for g in range(G):
            for hi in range(Hi):
                for wi in range(Wi):
                    for c in range(C):
                        acc = 0.0
                        for k in range(K):
                            for y in range(Y):
                                for x in range(X):
                                    # Compute corresponding output position
                                    ho_f = hi + pad_h - y * dilation_h
                                    wo_f = wi + pad_w - x * dilation_w

                                    # Check if this is a valid strided position
                                    if (
                                        ho_f >= 0
                                        and ho_f % stride_h == 0
                                        and wo_f >= 0
                                        and wo_f % stride_w == 0
                                    ):
                                        ho = ho_f // stride_h
                                        wo = wo_f // stride_w

                                        if 0 <= ho < Ho and 0 <= wo < Wo:
                                            acc += float(
                                                grad_output[n, ho, wo, g, k]
                                            ) * float(weight[g, k, y, x, c])
                        grad_input[n, hi, wi, g, c] = acc

    return grad_input.astype(grad_output.dtype)


def main():
    parser = argparse.ArgumentParser(description="Backward Data Convolution Example")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable CPU reference validation",
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
    print("Example 11: Backward Data Convolution")
    print("=" * 70)
    print()

    # =========================================================================
    # Step 0: Reset state for clean example run
    # =========================================================================
    reset_for_conv_example(verbose=True)

    # =========================================================================
    # Step 1: Define backward data kernel configuration
    # =========================================================================
    print("\nStep 1: Define Backward Data Kernel Configuration")
    print("-" * 50)

    sig = ConvSignature()
    sig.dtype(args.dtype, args.dtype, args.dtype, "fp32")
    sig.layout = "nhwgc"
    sig.direction = "bwd_data"  # Backward data direction
    sig.num_dims = 2

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
        direction=sig.direction,  # Pass direction for operator-specific validation
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
            direction=sig.direction,  # Pass direction for operator-specific validation
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

    kernel_set = ConvKernelSet("conv_bwd_data_kernels")
    kernel_set.add(sig, algo, arch)

    print(f"  Kernel Set: {kernel_set.name}")
    print(f"  Configurations: {len(kernel_set.configs)}")
    for cfg in kernel_set.configs:
        print(f"    - {cfg.name()}")
    print()

    # =========================================================================
    # Step 4: Define problem
    # =========================================================================
    print("Step 4: Define Problem")
    print("-" * 50)

    problem = ConvProblem(
        N=1,
        C=64,
        K=128,
        Hi=28,
        Wi=28,
        Y=3,
        X=3,
        pad_h=1,
        pad_w=1,
        stride_h=1,
        stride_w=1,
        direction="bwd_data",
    )

    print(f"  N={problem.N}, C={problem.C}, K={problem.K}")
    print(f"  Input: {problem.Hi}x{problem.Wi}")
    print(f"  Filter: {problem.Y}x{problem.X}")
    print(f"  FLOPs: {problem.flops:.2e}")
    print()

    # =========================================================================
    # Step 5: Tensor Semantics
    # =========================================================================
    print("Step 5: Backward Data Tensor Semantics")
    print("-" * 50)
    print("""
  Backward Data computes: dL/dInput

  Inputs:
    - dOutput: Gradient from next layer (N, Ho, Wo, K)
    - Weight:  Filter weights (K, Y, X, C)

  Output:
    - dInput:  Input gradient to propagate (N, Hi, Wi, C)

  Computation:
    dInput = transposed_conv(dOutput, Weight)
""")

    # =========================================================================
    # Step 6: Generate test data
    # =========================================================================
    print("Step 6: Generate Test Data")
    print("-" * 50)

    np_dtype = {
        "fp16": np.float16,
        "bf16": np.float16,
        "fp32": np.float32,
    }[args.dtype]

    # Create test problem
    prob = ConvProblem(
        N=1, C=64, K=128, Hi=14, Wi=14, Y=3, X=3, pad_h=1, pad_w=1, direction="bwd_data"
    )

    # Generate test data
    doutput = np.random.uniform(
        -0.5, 0.5, (prob.N, prob.Ho, prob.Wo, prob.G, prob.K)
    ).astype(np_dtype)
    weight = np.random.uniform(
        -0.5, 0.5, (prob.G, prob.K, prob.Y, prob.X, prob.C)
    ).astype(np_dtype)

    print(f"  dOutput: {doutput.shape} ({np_dtype.__name__})")
    print(f"  Weight:  {weight.shape} ({np_dtype.__name__})")
    print()

    # =========================================================================
    # Step 7: GPU Execution
    # =========================================================================
    print("Step 7: GPU Execution")
    print("-" * 50)

    runner = GpuConvRunner()
    gpu_output = None
    if runner.is_available():
        print(f"  Library: {runner.library_path}")

        result = runner.run(doutput, weight, prob)

        if result.get("success"):
            print("\n  *** GPU EXECUTION SUCCESSFUL ***")
            print(f"  Time:   {result['time_ms']:.4f} ms")
            print(f"  TFLOPS: {result['tflops']:.2f}")
            gpu_output = result.get("output")
        else:
            print(f"  Execution: {result.get('error', 'kernel not found')}")

        runner.cleanup()
    else:
        print("  GPU library not available")

    # =========================================================================
    # Step 8: CPU Reference Validation (optional)
    # =========================================================================
    if args.verify and gpu_output is not None:
        print("\nStep 8: CPU Reference Validation")
        print("-" * 50)

        input_shape = (prob.N, prob.Hi, prob.Wi, prob.G, prob.C)
        cpu_output = conv2d_bwd_data_reference(
            doutput,
            weight,
            input_shape,
            stride=(prob.stride_h, prob.stride_w),
            padding=(prob.pad_h, prob.pad_w),
        )

        # Compare GPU and CPU results
        gpu_flat = gpu_output.flatten().astype(np.float32)
        cpu_flat = cpu_output.flatten().astype(np.float32)

        abs_diff = np.abs(gpu_flat - cpu_flat)
        rel_diff = np.where(cpu_flat != 0, abs_diff / np.abs(cpu_flat), abs_diff)

        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)

        print(f"  GPU[0]: {gpu_flat[0]:.4f}")
        print(f"  CPU[0]: {cpu_flat[0]:.4f}")
        print(f"\n  Max abs diff: {max_abs_diff:.4e}")
        print(f"  Max rel diff: {max_rel_diff:.4e}")

        # FP16 tolerance
        passed = max_rel_diff < 0.1  # 10% for FP16 with accumulation differences
        print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    # =========================================================================
    # Cleanup and Summary
    # =========================================================================
    cleanup_conv()

    print()
    print("=" * 70)
    print("SUMMARY: Backward Data Convolution")
    print("=" * 70)
    print(f"  Kernel:  {args.dtype} {sig.direction} {sig.num_dims}D")
    print(f"  Config:  tile={args.tile_k}x{args.tile_c}, pipeline={args.pipeline}")
    print("  Purpose: Compute dL/dInput for backpropagation")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
