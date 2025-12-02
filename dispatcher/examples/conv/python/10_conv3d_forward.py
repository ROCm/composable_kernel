#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 10: 3D Convolution Forward with GPU Execution

Demonstrates 3D convolution (e.g., for video or volumetric data) with GPU execution.

Usage:
    python3 10_conv3d_forward.py
"""

import sys
import ctypes
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from conv_utils import (
    ConvSignature,
    ConvAlgorithm,
    ArchInfo,
    ConvKernelSet,
    ConvProblem,
    ConvDispatcherLib,
)


def main():
    print("=" * 70)
    print("Example 10: 3D Convolution Forward with GPU Execution")
    print("=" * 70)
    print()

    # =========================================================================
    # Step 1: Define 3D kernels
    # =========================================================================
    print("Step 1: Define 3D Kernels")
    print("-" * 50)

    kernel_set = ConvKernelSet("conv3d_fwd_kernels")

    sig = ConvSignature()
    sig.dtype("fp16")
    sig.layout = "ndhwc"
    sig.direction = "forward"
    sig.num_dims = 3  # 3D convolution

    algo = ConvAlgorithm()
    algo.tile(1, 128, 128)
    algo.wave(2, 2, 1)
    algo.warp(32, 32, 16)
    algo.pipeline = "compv3"
    algo.scheduler = "intrawave"

    kernel_set.add(sig, algo, ArchInfo(name="gfx942"))

    print(f"  Kernel Set: {kernel_set.name}")
    print(f"  Configurations: {len(kernel_set.configs)}")
    for cfg in kernel_set.configs:
        print(f"    - {cfg.name()}")
    print()

    # =========================================================================
    # Step 2: Define 3D problem
    # =========================================================================
    print("Step 2: Define 3D Problem")
    print("-" * 50)

    # 3D problem: N=1, C=32, K=64, D=8, H=16, W=16, filter 3x3x3
    problem = ConvProblem(
        N=1,
        C=32,
        K=64,
        Di=8,
        Hi=16,
        Wi=16,  # 3D spatial dimensions
        Z=3,
        Y=3,
        X=3,  # 3D filter
        pad_d=1,
        pad_h=1,
        pad_w=1,
        stride_d=1,
        stride_h=1,
        stride_w=1,
        direction="forward",
    )

    print(f"  N={problem.N}, C={problem.C}, K={problem.K}")
    print(f"  Input (3D): {problem.Di}x{problem.Hi}x{problem.Wi}")
    print(f"  Filter (3D): {problem.Z}x{problem.Y}x{problem.X}")
    print(f"  Output (3D): {problem.Do}x{problem.Ho}x{problem.Wo}")
    print(f"  FLOPs: {problem.flops_3d:.2e}")
    print()

    # =========================================================================
    # Step 3: GPU Execution
    # =========================================================================
    print("Step 3: GPU Execution")
    print("-" * 50)

    lib = ConvDispatcherLib.find()

    if lib is None:
        print("  [Dispatcher library not found]")
        return 1

    if not lib.has_kernels():
        print("  [No kernels compiled]")
        return 1

    lib.initialize()
    print(f"  Library: {lib.path}")
    print(f"  Kernels: {lib.get_kernel_count()}")

    try:
        hip_lib = ctypes.CDLL("libamdhip64.so")

        # 3D tensor sizes (NDHWC layout)
        dtype = np.float16
        dtype_size = dtype().itemsize  # 2 bytes for fp16
        input_size = (
            problem.N * problem.Di * problem.Hi * problem.Wi * problem.C * dtype_size
        )
        weight_size = (
            problem.K * problem.Z * problem.Y * problem.X * problem.C * dtype_size
        )
        output_size = (
            problem.N * problem.Do * problem.Ho * problem.Wo * problem.K * dtype_size
        )

        hip_lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
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

        # Create tensors
        input_host = np.random.randn(
            problem.N, problem.Di, problem.Hi, problem.Wi, problem.C
        ).astype(np.float16)
        weight_host = np.random.randn(
            problem.K, problem.Z, problem.Y, problem.X, problem.C
        ).astype(np.float16)

        # Allocate device memory
        input_dev = ctypes.c_void_p()
        weight_dev = ctypes.c_void_p()
        output_dev = ctypes.c_void_p()

        hip_lib.hipMalloc(ctypes.byref(input_dev), input_size)
        hip_lib.hipMalloc(ctypes.byref(weight_dev), weight_size)
        hip_lib.hipMalloc(ctypes.byref(output_dev), output_size)

        hip_lib.hipMemcpy(input_dev, input_host.ctypes.data, input_size, 1)
        hip_lib.hipMemcpy(weight_dev, weight_host.ctypes.data, weight_size, 1)

        print(f"  Input (3D):  {input_host.shape} -> GPU")
        print(f"  Weight (3D): {weight_host.shape} -> GPU")

        # Run 3D convolution
        elapsed_ms = lib.run(
            input_dev.value, weight_dev.value, output_dev.value, problem
        )
        hip_lib.hipDeviceSynchronize()

        if elapsed_ms > 0:
            tflops = problem.flops_3d / (elapsed_ms * 1e9)
            print("\n  *** 3D CONV GPU EXECUTION SUCCESSFUL ***")
            print(f"  Time:   {elapsed_ms:.4f} ms")
            print(f"  TFLOPS: {tflops:.2f}")
        else:
            print(f"  [GPU execution returned {elapsed_ms}]")

        hip_lib.hipFree(input_dev)
        hip_lib.hipFree(weight_dev)
        hip_lib.hipFree(output_dev)

    except Exception as e:
        print(f"  [Error: {e}]")

    lib.cleanup()

    print()
    print("=" * 70)
    print("3D Convolution: Used for video, medical imaging, volumetric data")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
