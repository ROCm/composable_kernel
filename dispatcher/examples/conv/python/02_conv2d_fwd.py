#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 02: 2D Convolution Forward (Python)

Demonstrates generating and running 2D forward convolution using Python.
Uses conv_utils.py for Signature/Algorithm/Arch pattern.

Usage:
    python3 02_conv2d_fwd.py
    python3 02_conv2d_fwd.py --verify
    python3 02_conv2d_fwd.py -n 2 -c 64 -k 128 -hi 56 -y 3
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
    ConvValidator,
    create_conv2d_fwd_config,
)

# Add codegen path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codegen"))


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
        "--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"]
    )
    parser.add_argument(
        "--arch", type=str, default="gfx942", help="Target architecture"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 02: 2D Convolution Forward (Signature/Algorithm/Arch Pattern)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Define problem using ConvProblem
    # -------------------------------------------------------------------------
    print("\nStep 1: Define ConvProblem")
    print("-" * 40)

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

    # -------------------------------------------------------------------------
    # Step 2: Define kernel config using Signature/Algorithm/Arch
    # -------------------------------------------------------------------------
    print("\nStep 2: Define Kernel Config (Signature/Algorithm/Arch)")
    print("-" * 40)

    # Method 1: Using convenience function
    config_simple = create_conv2d_fwd_config(
        dtype=args.dtype, tile_k=128, tile_c=128, arch=args.arch
    )
    print(f"  Simple config: {config_simple.name()}")

    # Method 2: Full explicit specification
    sig = ConvSignature()
    sig.dtype(args.dtype, args.dtype, args.dtype, "fp32")
    sig.layout = "nhwc"
    sig.direction = "forward"
    sig.num_dims = 2
    sig.groups = args.g

    algo = ConvAlgorithm()
    algo.tile(1, 128, 128)  # N, K, C tile
    algo.tile_output(1, 16)  # Ho, Wo tile
    algo.wave(2, 2, 1)  # Warp distribution
    algo.warp(32, 32, 16)  # Warp tile sizes
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

    kernel_set = ConvKernelSet("conv2d_fwd_set")
    kernel_set.add(sig, algo, arch)

    # Add additional tile sizes
    for tile_k, tile_c in [(64, 64), (256, 256)]:
        algo_variant = algo.copy()
        algo_variant.tile_k = tile_k
        algo_variant.tile_c = tile_c
        kernel_set.add(sig.copy(), algo_variant, arch)

    kernel_set.print()

    # -------------------------------------------------------------------------
    # Step 4: Generate test data
    # -------------------------------------------------------------------------
    print("\nStep 4: Generate Test Data")
    print("-" * 40)

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

    print(f"  Input:  {input_np.shape} ({input_np.dtype})")
    print(f"  Weight: {weight_np.shape} ({weight_np.dtype})")

    # -------------------------------------------------------------------------
    # Step 5: CPU verification (optional)
    # -------------------------------------------------------------------------
    if args.verify:
        print("\nStep 5: CPU Reference Verification")
        print("-" * 40)

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

    # -------------------------------------------------------------------------
    # Step 5: GPU Execution
    # -------------------------------------------------------------------------
    print("\nStep 5: GPU Execution")
    print("-" * 40)

    try:
        from conv_utils import ConvDispatcherLib
        import ctypes

        lib = ConvDispatcherLib.find()
        if lib is None:
            print("  Library not found - showing config pattern only")
            print("\n  To run on GPU: Build dispatcher_conv_lib.so")
        else:
            lib.initialize()
            print(f"  Library: {lib.path}")

            # Load HIP library
            hip_lib = ctypes.CDLL("libamdhip64.so")
            hip_lib.hipMalloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
            ]
            hip_lib.hipMalloc.restype = ctypes.c_int
            hip_lib.hipFree.argtypes = [ctypes.c_void_p]
            hip_lib.hipFree.restype = ctypes.c_int
            hip_lib.hipMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            hip_lib.hipMemcpy.restype = ctypes.c_int
            hip_lib.hipDeviceSynchronize.argtypes = []
            hip_lib.hipDeviceSynchronize.restype = ctypes.c_int

            # Sizes
            input_size = input_np.nbytes
            weight_size = weight_np.nbytes
            output_size = (
                problem.N
                * problem.Ho
                * problem.Wo
                * problem.K
                * input_np.dtype.itemsize
            )

            # Allocate GPU memory
            input_dev = ctypes.c_void_p()
            weight_dev = ctypes.c_void_p()
            output_dev = ctypes.c_void_p()

            hip_lib.hipMalloc(ctypes.byref(input_dev), input_size)
            hip_lib.hipMalloc(ctypes.byref(weight_dev), weight_size)
            hip_lib.hipMalloc(ctypes.byref(output_dev), output_size)

            # Copy to device
            hip_lib.hipMemcpy(input_dev, input_np.ctypes.data, input_size, 1)
            hip_lib.hipMemcpy(weight_dev, weight_np.ctypes.data, weight_size, 1)

            print(f"  Input:  {input_np.shape} -> GPU")
            print(f"  Weight: {weight_np.shape} -> GPU")

            # Run convolution
            elapsed_ms = lib.run(
                input_dev.value, weight_dev.value, output_dev.value, problem
            )
            hip_lib.hipDeviceSynchronize()

            # Free GPU memory
            hip_lib.hipFree(input_dev)
            hip_lib.hipFree(weight_dev)
            hip_lib.hipFree(output_dev)

            if elapsed_ms > 0:
                tflops = problem.flops / (elapsed_ms * 1e9)
                print("\n  *** GPU EXECUTION SUCCESSFUL ***")
                print(f"  Time:   {elapsed_ms:.4f} ms")
                print(f"  TFLOPS: {tflops:.2f}")
            else:
                print(f"  Kernel returned: {elapsed_ms}")

            lib.cleanup()
    except Exception as e:
        print(f"  GPU execution not available: {e}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KERNEL CONFIG PATTERN")
    print("=" * 70)
    print("""
# Full Signature + Algorithm + Arch specification:

sig = ConvSignature()
sig.dtype("fp16", "fp16", "fp16", "fp32")
sig.layout = "nhwc"
sig.direction = "forward"
sig.num_dims = 2

algo = ConvAlgorithm()
algo.tile(1, 128, 128)      # N, K, C
algo.wave(2, 2, 1)          # Warp distribution
algo.warp(32, 32, 16)       # Warp tile
algo.pipeline = "compv4"
algo.scheduler = "intrawave"

arch = ArchInfo(name="gfx942")

config = ConvKernelConfig(signature=sig, algorithm=algo, arch=arch)
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
