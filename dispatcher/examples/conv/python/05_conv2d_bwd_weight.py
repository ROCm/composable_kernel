#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 05: 2D Convolution Backward Weight (Python)

Computes gradient w.r.t. weight: dW = ConvBwdWeight(X, dY)
Uses the Signature/Algorithm/Arch pattern with full GPU execution.

Usage:
    python3 05_conv2d_bwd_weight.py
    python3 05_conv2d_bwd_weight.py --verify
"""

import sys
import argparse
import numpy as np

# Import conv utilities
from conv_utils import (
    ConvSignature,
    ConvAlgorithm,
    ArchInfo,
    ConvKernelConfig,
    ConvKernelSet,
    ConvProblem,
    create_conv2d_bwd_weight_config,
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
    parser.add_argument("--dtype", type=str, default="fp16", help="Data type")
    parser.add_argument(
        "--arch", type=str, default="gfx942", help="Target architecture"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 05: 2D Conv Backward Weight (Signature/Algorithm/Arch Pattern)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Define problem
    # -------------------------------------------------------------------------
    print("\nStep 1: Define ConvProblem")
    print("-" * 40)

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

    # -------------------------------------------------------------------------
    # Step 2: Define kernel config
    # -------------------------------------------------------------------------
    print("\nStep 2: Define Kernel Config")
    print("-" * 40)

    # Method 1: Using convenience function
    config_simple = create_conv2d_bwd_weight_config(
        dtype=args.dtype, tile_k=128, tile_c=128, arch=args.arch
    )
    print(f"  Simple config: {config_simple.name()}")

    # Method 2: Full explicit specification
    sig = ConvSignature()
    sig.dtype(args.dtype, args.dtype, args.dtype, "fp32")
    sig.layout = "nhwc"
    sig.direction = "bwd_weight"
    sig.num_dims = 2
    sig.groups = G

    algo = ConvAlgorithm()
    algo.tile(1, 128, 128)
    algo.wave(2, 2, 1)
    algo.warp(32, 32, 16)
    algo.pipeline = "compv4"
    algo.scheduler = "intrawave"

    arch = ArchInfo(name=args.arch)

    config_explicit = ConvKernelConfig(signature=sig, algorithm=algo, arch=arch)

    print(f"  Explicit config: {config_explicit.name()}")
    print(f"  Brief: {config_explicit.brief()}")

    # -------------------------------------------------------------------------
    # Step 3: Create kernel set
    # -------------------------------------------------------------------------
    print("\nStep 3: Create Kernel Set")
    print("-" * 40)

    kernel_set = ConvKernelSet("conv2d_bwd_weight_set")
    kernel_set.add(sig, algo, arch)
    kernel_set.print()

    # -------------------------------------------------------------------------
    # Step 4: Generate test data
    # -------------------------------------------------------------------------
    print("\nStep 4: Generate Test Data")
    print("-" * 40)

    np_dtype = np.float16 if args.dtype == "fp16" else np.float32
    input_np = np.random.uniform(-0.5, 0.5, (N, Hi, Wi, G, C)).astype(np_dtype)
    grad_output = np.random.uniform(-0.5, 0.5, (N, Ho, Wo, G, K)).astype(np_dtype)

    print(f"  input:       {input_np.shape} ({input_np.dtype})")
    print(f"  grad_output: {grad_output.shape} ({grad_output.dtype})")

    # -------------------------------------------------------------------------
    # Step 5: CPU verification (optional)
    # -------------------------------------------------------------------------
    grad_weight_cpu = None
    if args.verify:
        print("\nStep 5: CPU Reference Verification")
        print("-" * 40)

        grad_weight_cpu = reference_conv2d_bwd_weight(
            input_np, grad_output, Y, X, stride, pad
        )
        print(f"  grad_weight shape: {grad_weight_cpu.shape}")
        print(f"  Range: [{grad_weight_cpu.min():.4f}, {grad_weight_cpu.max():.4f}]")
        print(f"  CPU[0,0,0,0,0]: {float(grad_weight_cpu[0, 0, 0, 0, 0]):.4f}")
        print("  CPU reference computed successfully!")

    # -------------------------------------------------------------------------
    # Step 6: GPU Execution (using separate backward weight library)
    # -------------------------------------------------------------------------
    print("\nStep 6: GPU Execution")
    print("-" * 40)

    from conv_utils import GpuConvBwdWeightRunner

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

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BACKWARD WEIGHT CONFIG PATTERN")
    print("=" * 70)
    print("""
sig = ConvSignature()
sig.dtype("fp16")
sig.layout = "nhwc"
sig.direction = "bwd_weight"  # Key difference from forward
sig.num_dims = 2

algo = ConvAlgorithm()
algo.tile(1, 128, 128)
algo.wave(2, 2, 1)
algo.warp(32, 32, 16)
algo.pipeline = "compv4"

config = ConvKernelConfig(signature=sig, algorithm=algo, arch=ArchInfo(name="gfx942"))
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
