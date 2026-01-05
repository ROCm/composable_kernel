#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 05: 2D Convolution Backward Weight (Python)

Computes gradient w.r.t. weight: dW = ConvBwdWeight(X, dY)
Uses the Signature/Algorithm/Arch pattern with full GPU execution.

Usage:
    python3 05_conv2d_bwd_weight.py
    python3 05_conv2d_bwd_weight.py --verify
    python3 05_conv2d_bwd_weight.py --dtype bf16
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


def reference_conv2d_bwd_weight(input_np, grad_output, Y, X, stride=1, pad=0):
    """CPU reference for conv backward weight (gradient w.r.t. weight)."""
    N, Hi, Wi, G, C = input_np.shape
    _, Ho, Wo, _, K = grad_output.shape

    # Pad input
    if pad > 0:
        input_padded = np.pad(
            input_np, ((0, 0), (pad, pad), (pad, pad), (0, 0), (0, 0)), mode="constant"
        )
    else:
        input_padded = input_np

    grad_weight = np.zeros((G, K, Y, X, C), dtype=np.float32)

    for g in range(G):
        for k in range(K):
            for y in range(Y):
                for x in range(X):
                    for c in range(C):
                        acc = 0.0
                        for n in range(N):
                            for ho in range(Ho):
                                for wo in range(Wo):
                                    hi = ho * stride + y
                                    wi = wo * stride + x
                                    acc += float(input_padded[n, hi, wi, g, c]) * float(
                                        grad_output[n, ho, wo, g, k]
                                    )
                        grad_weight[g, k, y, x, c] = acc

    return grad_weight.astype(input_np.dtype)


def main():
    parser = argparse.ArgumentParser(description="2D Conv Backward Weight Example")
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
    print("Example 05: 2D Conv Backward Weight (Signature/Algorithm/Arch Pattern)")
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
        direction="bwd_weight",
    )

    print("  Backward Weight: dW = ConvBwdWeight(X, dY)")
    print(f"  X (input):        (N={N}, Hi={Hi}, Wi={Wi}, G={G}, C={C})")
    print(f"  dY (grad_output): (N={N}, Ho={Ho}, Wo={Wo}, G={G}, K={K})")
    print(f"  dW (grad_weight): (G={G}, K={K}, Y={Y}, X={X}, C={C})")

    flops = 2 * N * G * K * Ho * Wo * C * Y * X
    print(f"  FLOPs: {flops:.2e}")

    # =========================================================================
    # Step 2: Define kernel config
    # =========================================================================
    print("\nStep 2: Define Kernel Config")
    print("-" * 50)

    sig = ConvSignature()
    sig.dtype(args.dtype, args.dtype, args.dtype, "fp32")
    sig.layout = "nhwgc"
    sig.direction = "bwd_weight"
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

    kernel_set = ConvKernelSet("conv2d_bwd_weight_set")
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

    input_np = np.random.uniform(-0.5, 0.5, (N, Hi, Wi, G, C)).astype(np_dtype)
    grad_output = np.random.uniform(-0.5, 0.5, (N, Ho, Wo, G, K)).astype(np_dtype)

    print(f"  input:       {input_np.shape} ({np_dtype.__name__})")
    print(f"  grad_output: {grad_output.shape} ({np_dtype.__name__})")

    # =========================================================================
    # Step 6: CPU verification (optional)
    # =========================================================================
    grad_weight_cpu = None
    if args.verify:
        print("\nStep 6: CPU Reference Verification")
        print("-" * 50)

        grad_weight_cpu = reference_conv2d_bwd_weight(
            input_np, grad_output, Y, X, stride, pad
        )
        print(f"  grad_weight shape: {grad_weight_cpu.shape}")
        print(f"  Range: [{grad_weight_cpu.min():.4f}, {grad_weight_cpu.max():.4f}]")
        print(f"  CPU[0,0,0,0,0]: {float(grad_weight_cpu[0, 0, 0, 0, 0]):.4f}")
        print("  CPU reference computed successfully!")

    # =========================================================================
    # Step 7: GPU Execution (using separate backward weight library)
    # =========================================================================
    print("\nStep 7: GPU Execution")
    print("-" * 50)

    runner = GpuConvBwdWeightRunner()
    if runner.is_available():
        print(f"  Library: {runner.library_path}")
        print(f"  input:       {input_np.shape} -> GPU")
        print(f"  grad_output: {grad_output.shape} -> GPU")

        # Allocate output for grad_weight
        grad_weight_gpu = np.zeros((G, K, Y, X, C), dtype=np_dtype)

        result = runner.run(input_np, grad_output, problem, grad_weight_gpu)

        if result.get("success"):
            print("\n  *** BACKWARD WEIGHT GPU EXECUTION SUCCESSFUL ***")
            print(f"  Time:   {result['time_ms']:.4f} ms")
            print(f"  TFLOPS: {result['tflops']:.2f}")
            print(f"  GPU[0,0,0,0,0]: {float(grad_weight_gpu[0, 0, 0, 0, 0]):.4f}")

            # Validation
            if args.verify and grad_weight_cpu is not None:
                abs_diff = np.abs(
                    grad_weight_gpu.astype(np.float32)
                    - grad_weight_cpu.astype(np.float32)
                )
                max_abs = abs_diff.max()

                nonzero = np.abs(grad_weight_cpu.astype(np.float32)) > 1e-6
                if np.any(nonzero):
                    rel_diff = abs_diff[nonzero] / np.abs(
                        grad_weight_cpu.astype(np.float32)[nonzero]
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
            print(f"  Execution failed: {result.get('error', 'unknown error')}")

        runner.cleanup()
    else:
        print("  GPU backward weight library not available")
        print("  Build with: make dispatcher_conv_bwdw_lib")

    # =========================================================================
    # Cleanup and Summary
    # =========================================================================
    cleanup_conv()

    print("\n" + "=" * 70)
    print("SUMMARY: Backward Weight Convolution")
    print("=" * 70)
    print(f"  Kernel:  {args.dtype} {sig.direction} {sig.num_dims}D")
    print(f"  Config:  tile={args.tile_k}x{args.tile_c}, pipeline={args.pipeline}")
    print("  Purpose: Compute dL/dWeight for training")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
