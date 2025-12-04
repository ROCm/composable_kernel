#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 12: Backward Weight Convolution

Demonstrates the backward weight gradient computation (dL/dWeight) API
with kernel configuration validation.

Used during neural network training to update filter weights.

Usage:
    python3 12_bwd_weight.py
    python3 12_bwd_weight.py --dtype bf16
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
    GpuConvBwdWeightRunner,
    validate_conv_config,
    auto_correct_conv_config,
    reset_for_conv_example,
    cleanup_conv,
    print_conv_kernel_config,
    print_conv_auto_correction,
)


def conv2d_bwd_weight_reference(
    input_data: np.ndarray,
    grad_output: np.ndarray,
    filter_shape: tuple,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
) -> np.ndarray:
    """
    CPU reference implementation for 2D backward weight convolution.

    Computes dL/dWeight = correlation(Input, dOutput)

    Args:
        input_data: Forward activation (N, Hi, Wi, G, C) - NHWGC layout
        grad_output: Gradient from next layer (N, Ho, Wo, G, K) - NHWGK layout
        filter_shape: (K, Y, X, C) - filter dimensions
        stride: (stride_h, stride_w)
        padding: (pad_h, pad_w)
        dilation: (dilation_h, dilation_w)

    Returns:
        grad_weight: Weight gradient (G, K, Y, X, C) - GKYXC layout
    """
    N, Hi, Wi, G, C = input_data.shape
    _, Ho, Wo, _, K = grad_output.shape
    _, Y, X, _ = filter_shape
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    # Pad input if needed
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(
            input_data, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0), (0, 0))
        )
    else:
        padded = input_data

    # Use float32 for accumulation
    grad_weight = np.zeros((G, K, Y, X, C), dtype=np.float32)

    # Backward weight: correlate input with grad_output
    for g in range(G):
        for k in range(K):
            for y in range(Y):
                for x in range(X):
                    for c in range(C):
                        acc = 0.0
                        for n in range(N):
                            for ho in range(Ho):
                                for wo in range(Wo):
                                    hi = ho * stride_h + y * dilation_h
                                    wi = wo * stride_w + x * dilation_w
                                    acc += float(padded[n, hi, wi, g, c]) * float(
                                        grad_output[n, ho, wo, g, k]
                                    )
                        grad_weight[g, k, y, x, c] = acc

    return grad_weight.astype(input_data.dtype)


def main():
    parser = argparse.ArgumentParser(description="Backward Weight Convolution Example")
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
    print("Example 12: Backward Weight Convolution")
    print("=" * 70)
    print()

    # =========================================================================
    # Step 0: Reset state for clean example run
    # =========================================================================
    reset_for_conv_example(verbose=True)

    # =========================================================================
    # Step 1: Define backward weight kernel configuration
    # =========================================================================
    print("\nStep 1: Define Backward Weight Kernel Configuration")
    print("-" * 50)

    sig = ConvSignature()
    sig.dtype(args.dtype, args.dtype, args.dtype, "fp32")
    sig.layout = "nhwgc"
    sig.direction = "bwd_weight"  # Backward weight direction
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

    kernel_set = ConvKernelSet("conv_bwd_weight_kernels")
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
        direction="bwd_weight",
    )

    print(f"  N={problem.N}, C={problem.C}, K={problem.K}")
    print(f"  Input: {problem.Hi}x{problem.Wi}")
    print(f"  Filter: {problem.Y}x{problem.X}")
    print(f"  FLOPs: {problem.flops:.2e}")
    print()

    # =========================================================================
    # Step 5: Tensor Semantics
    # =========================================================================
    print("Step 5: Backward Weight Tensor Semantics")
    print("-" * 50)
    print("""
  Backward Weight computes: dL/dWeight

  Inputs:
    - Input:   Forward activation (N, Hi, Wi, C)
    - dOutput: Gradient from next layer (N, Ho, Wo, K)

  Output:
    - dWeight: Weight gradient for optimizer (K, Y, X, C)

  Computation:
    dWeight = conv(Input^T, dOutput)
    (Cross-correlation of input activations with output gradients)
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

    # Create test problem (reuse problem from above)
    prob = ConvProblem(
        N=1,
        C=64,
        K=128,
        Hi=14,
        Wi=14,
        Y=3,
        X=3,
        pad_h=1,
        pad_w=1,
        direction="bwd_weight",
    )

    # Generate test data
    input_data = np.random.uniform(
        -0.5, 0.5, (prob.N, prob.Hi, prob.Wi, prob.G, prob.C)
    ).astype(np_dtype)
    doutput = np.random.uniform(
        -0.5, 0.5, (prob.N, prob.Ho, prob.Wo, prob.G, prob.K)
    ).astype(np_dtype)

    print(f"  Input:   {input_data.shape} ({np_dtype.__name__})")
    print(f"  dOutput: {doutput.shape} ({np_dtype.__name__})")
    print()

    # =========================================================================
    # Step 7: GPU Execution
    # =========================================================================
    print("Step 7: GPU Execution")
    print("-" * 50)

    # Use dedicated backward weight runner (separate library due to CK Tile template conflicts)
    runner = GpuConvBwdWeightRunner()
    gpu_output = None
    if runner.is_available():
        print(f"  Library: {runner.library_path}")

        result = runner.run(input_data, doutput, prob)

        if result.get("success"):
            print("\n  *** BACKWARD WEIGHT GPU EXECUTION SUCCESSFUL ***")
            print(f"  Time:   {result['time_ms']:.4f} ms")
            print(f"  TFLOPS: {result['tflops']:.2f}")
            gpu_output = result.get("output")
        else:
            print(f"  Execution: {result.get('error', 'kernel not found')}")

        runner.cleanup()
    else:
        print("  GPU library not available (need libdispatcher_conv_bwdw_lib.so)")

    # =========================================================================
    # Step 8: CPU Reference Validation (optional)
    # =========================================================================
    if args.verify and gpu_output is not None:
        print("\nStep 8: CPU Reference Validation")
        print("-" * 50)

        filter_shape = (prob.K, prob.Y, prob.X, prob.C)
        cpu_output = conv2d_bwd_weight_reference(
            input_data,
            doutput,
            filter_shape,
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
    print("SUMMARY: Backward Weight Convolution")
    print("=" * 70)
    print(f"  Kernel:  {args.dtype} {sig.direction} {sig.num_dims}D")
    print(f"  Config:  tile={args.tile_k}x{args.tile_c}, pipeline={args.pipeline}")
    print("  Purpose: Compute dL/dWeight for training")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
