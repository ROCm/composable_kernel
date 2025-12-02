#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 01: Basic Convolution with GPU Execution

Demonstrates the Signature/Algorithm/Arch pattern with GPU execution.
Includes validation against arch filter with auto-correction for invalid configs.

Usage:
    python3 01_basic_conv.py
"""

import sys
import ctypes
import numpy as np
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))

from conv_utils import (
    ConvSignature,
    ConvAlgorithm,
    ArchInfo,
    ConvKernelSet,
    ConvProblem,
    ConvDispatcherLib,
    validate_conv_config,
    find_matching_conv_kernel_header,
)


def hip_check(result):
    """Check HIP result and raise if error"""
    if result != 0:
        raise RuntimeError(f"HIP error: {result}")


def main():
    print("=" * 70)
    print("Example 01: Basic Convolution with GPU Execution")
    print("=" * 70)
    print()

    # =========================================================================
    # Step 1: Define kernels using the pattern
    # =========================================================================
    print("Step 1: Define Kernels (Signature/Algorithm/Arch)")
    print("-" * 50)

    kernel_set = ConvKernelSet("conv_fwd_kernels")

    sig = ConvSignature()
    sig.dtype("fp16", "fp16", "fp16", "fp32")
    sig.layout = "nhwc"
    sig.direction = "forward"
    sig.num_dims = 2

    algo = ConvAlgorithm()
    algo.tile(1, 128, 128)
    algo.wave(2, 2, 1)
    algo.warp(32, 32, 16)
    algo.pipeline = "compv3"
    algo.scheduler = "intrawave"  # Try "interwave" to see auto-correction

    arch = ArchInfo(name="gfx942")

    kernel_set.add(sig, algo, arch)

    print(f"  Kernel Set: {kernel_set.name}")
    print(f"  Configurations: {len(kernel_set.configs)}")
    for cfg in kernel_set.configs:
        print(f"    - {cfg.name()}")
    print()

    # =========================================================================
    # Step 2: Validate configuration against arch filter
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
    )
    validation.print_result()

    if not validation.is_valid:
        print("\n  Auto-correcting configuration...")
        for key, val in validation.suggested_fixes.items():
            if key == "scheduler":
                algo.scheduler = val
                print(f"    scheduler -> {val}")
            elif key == "wave_m":
                algo.wave_m = val
                print(f"    wave_m -> {val}")
            elif key == "wave_n":
                algo.wave_n = val
                print(f"    wave_n -> {val}")
            elif key == "warp_m":
                algo.warp_m = val
                print(f"    warp_m -> {val}")
            elif key == "warp_n":
                algo.warp_n = val
                print(f"    warp_n -> {val}")
    print()

    # =========================================================================
    # Step 3: Find matching kernel header
    # =========================================================================
    print("Step 3: Find Matching Kernel Header")
    print("-" * 50)

    kernel_header = find_matching_conv_kernel_header(
        dtype=sig.dtype_in,
        conv_type=sig.direction,
        ndim=sig.num_dims,
        pipeline=algo.pipeline,
        scheduler=algo.scheduler,
        tile_k=algo.tile_k,
        tile_c=algo.tile_c,
        wave_m=algo.wave_m,
        wave_n=algo.wave_n,
        wave_k=algo.wave_k,
    )

    if kernel_header:
        print(f"  Found: {kernel_header.name}")
    else:
        print("  No matching kernel found - library may have different params")
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
    )

    print(f"  N={problem.N}, C={problem.C}, K={problem.K}")
    print(f"  Input: {problem.Hi}x{problem.Wi}")
    print(f"  Filter: {problem.Y}x{problem.X}")
    print(f"  Output: {problem.Ho}x{problem.Wo}")
    print(f"  FLOPs: {problem.flops:.2e}")
    print()

    # =========================================================================
    # Step 5: Load Dispatcher Library
    # =========================================================================
    print("Step 5: Load Dispatcher Library")
    print("-" * 50)

    lib = ConvDispatcherLib.find()

    if lib is None:
        print("  [ERROR] Dispatcher library not found")
        print(
            "  Build with: cd dispatcher/build && cmake .. && make dispatcher_conv_lib"
        )
        return 1

    if not lib.has_kernels():
        print("  [ERROR] Library has no compiled kernels")
        print("  Generate kernels first:")
        print(
            "  python3 codegen/unified_conv_codegen.py --datatype fp16 --variant forward"
        )
        return 1

    lib.initialize()
    print(f"  Library: {lib.path}")
    print(f"  Version: {lib.get_version()}")
    print(f"  Has kernels: {lib.has_kernels()}")
    print()

    # =========================================================================
    # Step 6: GPU Execution
    # =========================================================================
    print("Step 6: GPU Execution")
    print("-" * 50)

    # Use ctypes to call HIP directly
    try:
        hip_lib = ctypes.CDLL("libamdhip64.so")
    except OSError:
        print("  [ERROR] Cannot load libamdhip64.so")
        print("  Make sure ROCm is installed")
        lib.cleanup()
        return 1

    # Allocate GPU memory using hipMalloc
    dtype_size = np.float16().itemsize  # 2 bytes for fp16
    input_size = problem.N * problem.C * problem.Hi * problem.Wi * dtype_size
    weight_size = problem.K * problem.C * problem.Y * problem.X * dtype_size
    output_size = problem.N * problem.K * problem.Ho * problem.Wo * dtype_size

    # hipMalloc
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

    # Create numpy arrays
    input_host = np.random.randn(problem.N, problem.Hi, problem.Wi, problem.C).astype(
        np.float16
    )
    weight_host = np.random.randn(problem.K, problem.Y, problem.X, problem.C).astype(
        np.float16
    )
    output_host = np.zeros(
        (problem.N, problem.Ho, problem.Wo, problem.K), dtype=np.float16
    )

    # Allocate device memory
    input_dev = ctypes.c_void_p()
    weight_dev = ctypes.c_void_p()
    output_dev = ctypes.c_void_p()

    hip_lib.hipMalloc(ctypes.byref(input_dev), input_size)
    hip_lib.hipMalloc(ctypes.byref(weight_dev), weight_size)
    hip_lib.hipMalloc(ctypes.byref(output_dev), output_size)

    # Copy to device (hipMemcpyHostToDevice = 1)
    hip_lib.hipMemcpy(input_dev, input_host.ctypes.data, input_size, 1)
    hip_lib.hipMemcpy(weight_dev, weight_host.ctypes.data, weight_size, 1)

    print(f"  Input:  {input_host.shape} -> GPU")
    print(f"  Weight: {weight_host.shape} -> GPU")
    print(f"  Output: {output_host.shape} (allocated)")

    # Run convolution on GPU
    elapsed_ms = lib.run(input_dev.value, weight_dev.value, output_dev.value, problem)

    hip_lib.hipDeviceSynchronize()

    if elapsed_ms > 0:
        tflops = problem.flops / (elapsed_ms * 1e9)
        print("\n  *** GPU EXECUTION SUCCESSFUL ***")
        print(f"  Time:   {elapsed_ms:.4f} ms")
        print(f"  TFLOPS: {tflops:.2f}")
    else:
        print(f"  [ERROR] GPU execution failed (returned {elapsed_ms})")

    # Cleanup
    hip_lib.hipFree(input_dev)
    hip_lib.hipFree(weight_dev)
    hip_lib.hipFree(output_dev)

    lib.cleanup()

    print()
    print("=" * 70)
    print("SUMMARY: Python example ran convolution on GPU!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
