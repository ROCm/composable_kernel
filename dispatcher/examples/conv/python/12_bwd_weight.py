#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 12: Backward Weight Convolution

Demonstrates the backward weight gradient computation (dL/dWeight) API.
Used during neural network training to update filter weights.

Note: GPU execution requires proper backward kernel codegen (in progress).

Usage:
    python3 12_bwd_weight.py
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
    print("Example 12: Backward Weight Convolution")
    print("=" * 70)
    print()

    # =========================================================================
    # Step 1: Define backward weight kernels
    # =========================================================================
    print("Step 1: Define Backward Weight Kernels")
    print("-" * 50)

    kernel_set = ConvKernelSet("conv_bwd_weight_kernels")

    sig = ConvSignature()
    sig.dtype("fp16")
    sig.layout = "nhwc"
    sig.direction = "bwd_weight"  # Backward weight direction
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
        direction="bwd_weight",
    )

    print(f"  N={problem.N}, C={problem.C}, K={problem.K}")
    print(f"  Input: {problem.Hi}x{problem.Wi}")
    print(f"  Filter: {problem.Y}x{problem.X}")
    print(f"  FLOPs: {problem.flops:.2e}")
    print()

    # =========================================================================
    # Step 3: Tensor Semantics
    # =========================================================================
    print("Step 3: Backward Weight Tensor Semantics")
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
    
  API Pattern:
    sig = ConvSignature()
    sig.direction = "bwd_weight"
    
    algo = ConvAlgorithm()
    algo.tile(1, 128, 128)
    
    # Once codegen is complete:
    # elapsed = lib.run_bwd_weight(input_ptr, doutput_ptr, dweight_ptr, problem)
""")

    # =========================================================================
    # Step 4: GPU Execution
    # =========================================================================
    print("Step 4: GPU Execution")
    print("-" * 50)

    from conv_utils import GpuConvBwdWeightRunner
    import numpy as np

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
    np_dtype = np.float16
    input_data = np.random.uniform(
        -0.5, 0.5, (prob.N, prob.Hi, prob.Wi, prob.G, prob.C)
    ).astype(np_dtype)
    doutput = np.random.uniform(
        -0.5, 0.5, (prob.N, prob.Ho, prob.Wo, prob.G, prob.K)
    ).astype(np_dtype)

    print(f"  Input:   {input_data.shape} ({input_data.dtype})")
    print(f"  dOutput: {doutput.shape} ({doutput.dtype})")
    print()

    # Use dedicated backward weight runner (separate library due to CK Tile template conflicts)
    runner = GpuConvBwdWeightRunner()
    if runner.is_available():
        print(f"  Library: {runner.library_path}")

        result = runner.run(input_data, doutput, prob)

        if result.get("success"):
            print("\n  *** BACKWARD WEIGHT GPU EXECUTION SUCCESSFUL ***")
            print(f"  Time:   {result['time_ms']:.4f} ms")
            print(f"  TFLOPS: {result['tflops']:.2f}")
        else:
            print(f"  Execution: {result.get('error', 'kernel not found')}")

        runner.cleanup()
    else:
        print("  GPU library not available (need libdispatcher_conv_bwdw_lib.so)")

    print()
    print("=" * 70)
    print("Backward Weight: Computes dL/dWeight for training")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
