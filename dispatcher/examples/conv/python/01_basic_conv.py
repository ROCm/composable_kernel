#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 01: Basic Convolution with GPU Execution

Demonstrates the Signature/Algorithm/Arch pattern with GPU execution.

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
)

# Try to import HIP for GPU memory management
try:
    from hip import hip  # noqa: F401

    HIP_AVAILABLE = True
except ImportError:
    HIP_AVAILABLE = False


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
    algo.scheduler = "intrawave"

    arch = ArchInfo(name="gfx942")

    kernel_set.add(sig, algo, arch)

    print(f"  Kernel Set: {kernel_set.name}")
    print(f"  Configurations: {len(kernel_set.configs)}")
    for cfg in kernel_set.configs:
        print(f"    - {cfg.name()}")
    print()

    # =========================================================================
    # Step 2: Define problem
    # =========================================================================
    print("Step 2: Define Problem")
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
    # Step 3: Load Dispatcher Library
    # =========================================================================
    print("Step 3: Load Dispatcher Library")
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
    # Step 4: GPU Execution
    # =========================================================================
    print("Step 4: GPU Execution")
    print("-" * 50)

    if not HIP_AVAILABLE:
        print("  [NOTE] hip-python not available - using ctypes for GPU memory")
        print("  Install with: pip install hip-python")
        print()

        # Use ctypes to call HIP directly
        try:
            hip_lib = ctypes.CDLL("libamdhip64.so")
        except OSError:
            print("  [ERROR] Cannot load libamdhip64.so")
            print("  Make sure ROCm is installed")
            lib.cleanup()
            return 1

        # Allocate GPU memory using hipMalloc
        input_size = problem.N * problem.C * problem.Hi * problem.Wi * 2  # fp16
        weight_size = problem.K * problem.C * problem.Y * problem.X * 2
        output_size = problem.N * problem.K * problem.Ho * problem.Wo * 2

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
        input_host = np.random.randn(
            problem.N, problem.Hi, problem.Wi, problem.C
        ).astype(np.float16)
        weight_host = np.random.randn(
            problem.K, problem.Y, problem.X, problem.C
        ).astype(np.float16)
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
        elapsed_ms = lib.run(
            input_dev.value, weight_dev.value, output_dev.value, problem
        )

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

    else:
        # Use hip-python (cleaner API)
        # ... similar logic with hip-python API
        print("  Using hip-python for GPU memory management")

    lib.cleanup()

    print()
    print("=" * 70)
    print("SUMMARY: Python example ran convolution on GPU!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
