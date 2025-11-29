#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 03: 3D Convolution Forward (Python)

Demonstrates 3D forward convolution using the Signature/Algorithm/Arch pattern.

Usage:
    python3 03_conv3d_fwd.py
    python3 03_conv3d_fwd.py --verify
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Import conv utilities
from conv_utils import (
    ConvSignature,
    ConvAlgorithm,
    ArchInfo,
    ConvKernelConfig,
    ConvKernelSet,
    ConvProblem,
    create_conv3d_fwd_config,
)

# Add codegen path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codegen"))


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
    parser.add_argument("--dtype", type=str, default="fp16", help="Data type")
    parser.add_argument(
        "--arch", type=str, default="gfx942", help="Target architecture"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 03: 3D Convolution Forward (Signature/Algorithm/Arch Pattern)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Define problem using ConvProblem
    # -------------------------------------------------------------------------
    print("\nStep 1: Define ConvProblem")
    print("-" * 40)

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

    # -------------------------------------------------------------------------
    # Step 2: Define kernel config (Signature/Algorithm/Arch)
    # -------------------------------------------------------------------------
    print("\nStep 2: Define Kernel Config")
    print("-" * 40)

    # Method 1: Using convenience function
    config_simple = create_conv3d_fwd_config(
        dtype=args.dtype, tile_k=64, tile_c=64, arch=args.arch
    )
    print(f"  Simple config: {config_simple.name()}")

    # Method 2: Full explicit specification
    sig = ConvSignature()
    sig.dtype(args.dtype, args.dtype, args.dtype, "fp32")
    sig.layout = "ndhwc"
    sig.direction = "forward"
    sig.num_dims = 3
    sig.groups = G

    algo = ConvAlgorithm()
    algo.tile(1, 64, 64)  # N, K, C tile
    algo.wave(2, 2, 1)  # Warp distribution
    algo.warp(16, 16, 32)  # Warp tile sizes
    algo.pipeline = "compv3"
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

    kernel_set = ConvKernelSet("conv3d_fwd_set")
    kernel_set.add(sig, algo, arch)
    kernel_set.print()

    # -------------------------------------------------------------------------
    # Step 4: Generate test data (NDHWGC layout)
    # -------------------------------------------------------------------------
    print("\nStep 4: Generate Test Data")
    print("-" * 40)

    np_dtype = np.float16 if args.dtype == "fp16" else np.float32
    input_np = np.random.uniform(-0.5, 0.5, (N, Di, Hi, Wi, G, C)).astype(np_dtype)
    weight_np = np.random.uniform(-0.5, 0.5, (G, K, Z, Y, X, C)).astype(np_dtype)

    print(f"  Input:  {input_np.shape} ({input_np.dtype})")
    print(f"  Weight: {weight_np.shape} ({weight_np.dtype})")

    # -------------------------------------------------------------------------
    # Step 5: CPU verification (optional)
    # -------------------------------------------------------------------------
    if args.verify:
        print("\nStep 5: CPU Reference Verification")
        print("-" * 40)

        output_ref = reference_conv3d_fwd(input_np, weight_np, stride=stride, pad=pad)
        print(f"  Output shape: {output_ref.shape}")
        print(f"  Output range: [{output_ref.min():.4f}, {output_ref.max():.4f}]")
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

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3D CONV CONFIG PATTERN")
    print("=" * 70)
    print("""
sig = ConvSignature()
sig.dtype("fp16")
sig.layout = "ndhwc"
sig.direction = "forward"
sig.num_dims = 3

algo = ConvAlgorithm()
algo.tile(1, 64, 64)
algo.wave(2, 2, 1)
algo.warp(16, 16, 32)
algo.pipeline = "compv3"

arch = ArchInfo(name="gfx942")

config = ConvKernelConfig(signature=sig, algorithm=algo, arch=arch)
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
