#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 11: Backward Data Convolution

Demonstrates the backward data gradient computation (dL/dInput) API.
Used during neural network backpropagation.

Note: GPU execution requires proper backward kernel codegen (in progress).

Usage:
    python3 11_bwd_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from conv_utils import (
    ConvSignature,
    ConvAlgorithm,
    ArchInfo,
    ConvKernelSet,
    ConvProblem,
)


def main():
    print("=" * 70)
    print("Example 11: Backward Data Convolution")
    print("=" * 70)
    print()

    # =========================================================================
    # Step 1: Define backward data kernels
    # =========================================================================
    print("Step 1: Define Backward Data Kernels")
    print("-" * 50)

    kernel_set = ConvKernelSet("conv_bwd_data_kernels")

    sig = ConvSignature()
    sig.dtype("fp16")
    sig.layout = "nhwc"
    sig.direction = "bwd_data"  # Backward data direction
    sig.num_dims = 2

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
        direction="bwd_data",
    )

    print(f"  N={problem.N}, C={problem.C}, K={problem.K}")
    print(f"  Input: {problem.Hi}x{problem.Wi}")
    print(f"  Filter: {problem.Y}x{problem.X}")
    print(f"  FLOPs: {problem.flops:.2e}")
    print()

    # =========================================================================
    # Step 3: Tensor Semantics
    # =========================================================================
    print("Step 3: Backward Data Tensor Semantics")
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
    
  API Pattern:
    sig = ConvSignature()
    sig.direction = "bwd_data"
    
    algo = ConvAlgorithm()
    algo.tile(1, 128, 128)
    
    # Once codegen is complete:
    # elapsed = lib.run_bwd_data(doutput_ptr, weight_ptr, dinput_ptr, problem)
""")

    # =========================================================================
    # Step 4: GPU Execution
    # =========================================================================
    print("Step 4: GPU Execution")
    print("-" * 50)

    from conv_utils import GpuConvRunner
    import numpy as np

    # Create test problem
    prob = ConvProblem(
        N=1, C=64, K=128, Hi=14, Wi=14, Y=3, X=3, pad_h=1, pad_w=1, direction="bwd_data"
    )

    # Generate test data
    np_dtype = np.float16
    doutput = np.random.uniform(
        -0.5, 0.5, (prob.N, prob.Ho, prob.Wo, prob.G, prob.K)
    ).astype(np_dtype)
    weight = np.random.uniform(
        -0.5, 0.5, (prob.G, prob.K, prob.Y, prob.X, prob.C)
    ).astype(np_dtype)

    print(f"  dOutput: {doutput.shape} ({doutput.dtype})")
    print(f"  Weight:  {weight.shape} ({weight.dtype})")
    print()

    runner = GpuConvRunner()
    if runner.is_available():
        print(f"  Library: {runner.library_path}")

        result = runner.run(doutput, weight, prob)

        if result.get("success"):
            print("\n  *** GPU EXECUTION SUCCESSFUL ***")
            print(f"  Time:   {result['time_ms']:.4f} ms")
            print(f"  TFLOPS: {result['tflops']:.2f}")
        else:
            print(f"  Execution: {result.get('error', 'kernel not found')}")

        runner.cleanup()
    else:
        print("  GPU library not available")

    print()
    print("=" * 70)
    print("Backward Data: Computes dL/dInput for backpropagation")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
