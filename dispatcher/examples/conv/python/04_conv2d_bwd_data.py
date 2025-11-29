#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 04: 2D Convolution Backward Data (Python)

Computes gradient w.r.t. input: dX = ConvBwdData(dY, W)
Uses the Signature/Algorithm/Arch pattern.

Usage:
    python3 04_conv2d_bwd_data.py
    python3 04_conv2d_bwd_data.py --verify
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
    create_conv2d_bwd_data_config,
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
    parser.add_argument("--dtype", type=str, default="fp16", help="Data type")
    parser.add_argument(
        "--arch", type=str, default="gfx942", help="Target architecture"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 04: 2D Conv Backward Data (Signature/Algorithm/Arch Pattern)")
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
        direction="bwd_data",
    )

    print("  Backward Data: dX = ConvBwdData(dY, W)")
    print(f"  dY (grad_output): (N={N}, Ho={Ho}, Wo={Wo}, G={G}, K={K})")
    print(f"  W (weight):       (G={G}, K={K}, Y={Y}, X={X}, C={C})")
    print(f"  dX (grad_input):  (N={N}, Hi={Hi}, Wi={Wi}, G={G}, C={C})")

    flops = 2 * N * G * C * Hi * Wi * K * Y * X
    print(f"  FLOPs: {flops:.2e}")

    # -------------------------------------------------------------------------
    # Step 2: Define kernel config
    # -------------------------------------------------------------------------
    print("\nStep 2: Define Kernel Config")
    print("-" * 40)

    # Method 1: Using convenience function
    config_simple = create_conv2d_bwd_data_config(
        dtype=args.dtype, tile_k=128, tile_c=128, arch=args.arch
    )
    print(f"  Simple config: {config_simple.name()}")

    # Method 2: Full explicit specification
    sig = ConvSignature()
    sig.dtype(args.dtype, args.dtype, args.dtype, "fp32")
    sig.layout = "nhwc"
    sig.direction = "bwd_data"
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

    kernel_set = ConvKernelSet("conv2d_bwd_data_set")
    kernel_set.add(sig, algo, arch)
    kernel_set.print()

    # -------------------------------------------------------------------------
    # Step 4: Generate test data
    # -------------------------------------------------------------------------
    print("\nStep 4: Generate Test Data")
    print("-" * 40)

    np_dtype = np.float16 if args.dtype == "fp16" else np.float32
    grad_output = np.random.uniform(-0.5, 0.5, (N, Ho, Wo, G, K)).astype(np_dtype)
    weight = np.random.uniform(-0.5, 0.5, (G, K, Y, X, C)).astype(np_dtype)

    print(f"  grad_output: {grad_output.shape} ({grad_output.dtype})")
    print(f"  weight:      {weight.shape} ({weight.dtype})")

    # -------------------------------------------------------------------------
    # Step 5: CPU verification (optional)
    # -------------------------------------------------------------------------
    grad_input_cpu = None
    if args.verify:
        print("\nStep 5: CPU Reference Verification")
        print("-" * 40)

        grad_input_cpu = reference_conv2d_bwd_data(
            grad_output, weight, stride, pad, Hi, Wi
        )
        print(f"  grad_input shape: {grad_input_cpu.shape}")
        print(f"  Range: [{grad_input_cpu.min():.4f}, {grad_input_cpu.max():.4f}]")
        print(f"  CPU[0,0,0,0,0]: {float(grad_input_cpu[0, 0, 0, 0, 0]):.4f}")
        print("  CPU reference computed successfully!")

    # -------------------------------------------------------------------------
    # Step 6: GPU Execution
    # -------------------------------------------------------------------------
    print("\nStep 6: GPU Execution")
    print("-" * 40)

    from conv_utils import GpuConvRunner

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

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BACKWARD DATA CONFIG PATTERN")
    print("=" * 70)
    print("""
sig = ConvSignature()
sig.dtype("fp16")
sig.layout = "nhwc"
sig.direction = "bwd_data"  # Key difference from forward
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
