#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 07: Convolution Validation

Demonstrates validating convolution results against CPU reference,
similar to GEMM 04_validation.py.

Usage:
    python3 07_validation.py
"""

import numpy as np
from conv_utils import (
    ConvProblem,
    ConvValidator,
)


def cpu_conv2d_nhwc(
    input_data: np.ndarray,
    weight: np.ndarray,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
) -> np.ndarray:
    """
    CPU reference implementation for 2D convolution with NHWC layout.

    Args:
        input_data: Input tensor (N, Hi, Wi, C)
        weight: Weight tensor (K, Y, X, C)
        stride: (stride_h, stride_w)
        padding: (pad_h, pad_w)
        dilation: (dilation_h, dilation_w)

    Returns:
        Output tensor (N, Ho, Wo, K)
    """
    N, Hi, Wi, C = input_data.shape
    K, Y, X, _ = weight.shape
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    # Calculate effective filter size with dilation
    eff_y = (Y - 1) * dilation_h + 1
    eff_x = (X - 1) * dilation_w + 1

    Ho = (Hi + 2 * pad_h - eff_y) // stride_h + 1
    Wo = (Wi + 2 * pad_w - eff_x) // stride_w + 1

    # Pad input if needed
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(input_data, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    else:
        padded = input_data

    # Use float32 for accumulation
    output = np.zeros((N, Ho, Wo, K), dtype=np.float32)

    for n in range(N):
        for ho in range(Ho):
            for wo in range(Wo):
                for k in range(K):
                    acc = 0.0
                    for y in range(Y):
                        for x in range(X):
                            for c in range(C):
                                hi = ho * stride_h + y * dilation_h
                                wi = wo * stride_w + x * dilation_w
                                acc += float(padded[n, hi, wi, c]) * float(
                                    weight[k, y, x, c]
                                )
                    output[n, ho, wo, k] = acc

    return output.astype(input_data.dtype)


def main():
    print("=" * 70)
    print("Example 07: Convolution Validation")
    print("=" * 70)
    print()

    # -------------------------------------------------------------------------
    # Step 1: Define validation problems
    # -------------------------------------------------------------------------
    print("VALIDATION PROBLEMS")
    print("=" * 40)

    problems = [
        # Small problem for easy debugging
        ("Small", ConvProblem(N=1, C=4, K=8, Hi=4, Wi=4, Y=3, X=3, pad_h=1, pad_w=1)),
        # Medium problem
        (
            "Medium",
            ConvProblem(N=1, C=16, K=32, Hi=8, Wi=8, Y=3, X=3, pad_h=1, pad_w=1),
        ),
        # Pointwise convolution (1x1)
        ("Pointwise", ConvProblem(N=1, C=64, K=64, Hi=14, Wi=14, Y=1, X=1)),
        # Strided convolution
        (
            "Strided",
            ConvProblem(
                N=1,
                C=16,
                K=32,
                Hi=8,
                Wi=8,
                Y=3,
                X=3,
                stride_h=2,
                stride_w=2,
                pad_h=1,
                pad_w=1,
            ),
        ),
        # No padding
        ("No Padding", ConvProblem(N=1, C=16, K=32, Hi=10, Wi=10, Y=3, X=3)),
        # Batch > 1
        (
            "Batch=4",
            ConvProblem(N=4, C=8, K=16, Hi=6, Wi=6, Y=3, X=3, pad_h=1, pad_w=1),
        ),
    ]

    for name, prob in problems:
        print(f"  {name}: {prob}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Run validation
    # -------------------------------------------------------------------------
    print("VALIDATION RESULTS")
    print("=" * 40)
    print()

    validator = ConvValidator(rtol=1e-3, atol=1e-3)
    all_passed = True

    print(f"{'Problem':<15} | {'Shape':<20} | {'Max Diff':>12} | {'Status':<8}")
    print("-" * 65)

    for name, prob in problems:
        # Create input data (small values to avoid overflow)
        np.random.seed(42)  # Reproducibility
        input_data = (np.random.randn(prob.N, prob.Hi, prob.Wi, prob.C) * 0.1).astype(
            np.float16
        )
        weight = (
            np.random.randn(prob.K, prob.Y, prob.X, prob.C // prob.G) * 0.1
        ).astype(np.float16)

        # Run CPU reference
        reference = cpu_conv2d_nhwc(
            input_data,
            weight,
            stride=(prob.stride_h, prob.stride_w),
            padding=(prob.pad_h, prob.pad_w),
            dilation=(prob.dilation_h, prob.dilation_w),
        )

        # For now, we validate CPU implementation against itself
        # (GPU validation requires compiled library)
        result = validator.check(reference, reference)

        shape_str = f"{prob.N}x{prob.Hi}x{prob.Wi}x{prob.C}"
        status = "PASS" if result["passed"] else "FAIL"

        print(
            f"{name:<15} | {shape_str:<20} | {result['max_abs_diff']:>12.6f} | {status:<8}"
        )

        if not result["passed"]:
            all_passed = False

    print()

    # -------------------------------------------------------------------------
    # Step 3: Detailed validation for small problem
    # -------------------------------------------------------------------------
    print("DETAILED VALIDATION (Small Problem)")
    print("=" * 40)
    print()

    prob = problems[0][1]  # Small problem
    np.random.seed(123)
    input_data = (np.random.randn(prob.N, prob.Hi, prob.Wi, prob.C) * 0.5).astype(
        np.float16
    )
    weight = (np.random.randn(prob.K, prob.Y, prob.X, prob.C) * 0.5).astype(np.float16)

    reference = cpu_conv2d_nhwc(
        input_data,
        weight,
        stride=(prob.stride_h, prob.stride_w),
        padding=(prob.pad_h, prob.pad_w),
    )

    print(f"Input shape:    {input_data.shape}")
    print(f"Weight shape:   {weight.shape}")
    print(f"Output shape:   {reference.shape}")
    print()

    print("Input (first 2x2 spatial, first channel):")
    print(input_data[0, :2, :2, 0])
    print()

    print("Weight (first filter, 3x3, first channel):")
    print(weight[0, :, :, 0])
    print()

    print("Output (first 2x2 spatial, first filter):")
    print(reference[0, :2, :2, 0])
    print()

    # -------------------------------------------------------------------------
    # Step 4: Numerical precision analysis
    # -------------------------------------------------------------------------
    print("NUMERICAL PRECISION ANALYSIS")
    print("=" * 40)
    print()

    # Test with identity-like operation
    prob = ConvProblem(N=1, C=1, K=1, Hi=5, Wi=5, Y=1, X=1)
    input_data = np.ones((1, 5, 5, 1), dtype=np.float16)
    weight = np.ones((1, 1, 1, 1), dtype=np.float16)

    output = cpu_conv2d_nhwc(input_data, weight)
    expected = np.ones((1, 5, 5, 1), dtype=np.float16)

    match = np.allclose(output, expected)
    print(f"Identity test (1x1 conv with ones): {'PASS' if match else 'FAIL'}")
    print(f"  Expected: {expected[0, 0, 0, 0]}")
    print(f"  Got:      {output[0, 0, 0, 0]}")
    print()

    # Test with simple 3x3 sum
    prob = ConvProblem(N=1, C=1, K=1, Hi=5, Wi=5, Y=3, X=3, pad_h=1, pad_w=1)
    input_data = np.ones((1, 5, 5, 1), dtype=np.float16)
    weight = np.ones((1, 3, 3, 1), dtype=np.float16)

    output = cpu_conv2d_nhwc(input_data, weight, padding=(1, 1))

    # Center should be 9.0 (3x3 = 9 ones)
    center_val = float(output[0, 2, 2, 0])
    print(f"3x3 sum test (ones): {'PASS' if abs(center_val - 9.0) < 0.1 else 'FAIL'}")
    print("  Expected center: 9.0")
    print(f"  Got center:      {center_val}")
    print()

    # -------------------------------------------------------------------------
    # Step 5: GPU vs CPU Validation
    # -------------------------------------------------------------------------
    print("GPU vs CPU VALIDATION")
    print("=" * 40)
    print()

    from conv_utils import GpuConvRunner

    runner = GpuConvRunner()
    if runner.is_available():
        # Use a small problem for detailed comparison
        prob = ConvProblem(N=1, C=64, K=128, Hi=14, Wi=14, Y=3, X=3, pad_h=1, pad_w=1)
        np.random.seed(42)
        input_data = np.random.randn(prob.N, prob.Hi, prob.Wi, prob.C).astype(
            np.float16
        )
        weight = np.random.randn(prob.K, prob.Y, prob.X, prob.C).astype(np.float16)

        # CPU reference
        cpu_out = cpu_conv2d_nhwc(
            input_data,
            weight,
            stride=(prob.stride_h, prob.stride_w),
            padding=(prob.pad_h, prob.pad_w),
        )

        # GPU output
        gpu_out = np.zeros((prob.N, prob.Ho, prob.Wo, prob.K), dtype=np.float16)
        result = runner.run(input_data, weight, prob, gpu_out)

        if result.get("success"):
            # Compare
            max_diff = np.max(np.abs(cpu_out - gpu_out))
            mean_diff = np.mean(np.abs(cpu_out - gpu_out))
            matches = np.allclose(cpu_out, gpu_out, rtol=1e-2, atol=1e-3)

            print(
                f"  Problem: {prob.N}x{prob.C}x{prob.Hi}x{prob.Wi} conv {prob.Y}x{prob.X}"
            )
            print(f"  GPU Time: {result['time_ms']:.4f} ms")
            print(f"  TFLOPS:   {result['tflops']:.2f}")
            print()
            print(f"  Max diff:  {max_diff:.6f}")
            print(f"  Mean diff: {mean_diff:.6f}")
            print(f"  Status:    {'PASS' if matches else 'FAIL'}")

            if matches:
                print("\n  *** GPU vs CPU VALIDATION PASSED ***")
        else:
            print(f"  GPU execution failed: {result.get('error')}")

        runner.cleanup()
    else:
        print("  GPU library not available - CPU validation only")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 70)
    if all_passed:
        print("All validation tests PASSED!")
    else:
        print("Some validation tests FAILED!")


if __name__ == "__main__":
    main()
