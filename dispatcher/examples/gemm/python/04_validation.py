#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 04: Validation

Validates GPU GEMM against NumPy reference using JIT compilation.

Usage:
    python3 04_validation.py
    python3 04_validation.py --help
    python3 04_validation.py --dtype bf16
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    Validator,
    Registry,
    detect_gpu_arch,
)


def main():
    parser = argparse.ArgumentParser(
        description="GEMM Validation Example - validates GPU results against NumPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 04_validation.py                    # Default FP16 validation
  python3 04_validation.py --dtype bf16       # BF16 validation
  python3 04_validation.py --rtol 1e-2        # Relaxed tolerance
        """,
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-3, help="Relative tolerance (default: 1e-3)"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-2, help="Absolute tolerance (default: 1e-2)"
    )
    parser.add_argument(
        "--arch",
        default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Example 04: Validation")
    print("=" * 60)

    # =========================================================================
    # Step 1: JIT build dispatcher
    # =========================================================================
    print("\nStep 1: JIT Build Dispatcher")

    config = KernelConfig(
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        tile_m=128,
        tile_n=128,
        tile_k=32,
        gfx_arch=args.arch,
    )

    reg = Registry(name="validation")
    reg.register_kernel(config)

    setups = reg.build(verbose=True)
    if not setups or not setups[0].success:
        error = setups[0].error if setups else "No kernels built"
        print(f"  ERROR: {error}")
        return 1

    dispatcher = setups[0].dispatcher

    # =========================================================================
    # Step 2: Run validation tests
    # =========================================================================
    print("\nStep 2: Validation Tests")

    validator = Validator(rtol=args.rtol, atol=args.atol)
    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    test_cases = [
        ("Identity", 128, 128, 128, "identity"),
        ("Small", 256, 256, 256, "random"),
        ("Medium", 512, 512, 512, "random"),
        ("Large", 1024, 1024, 1024, "random"),
        ("Non-square", 512, 1024, 256, "random"),
    ]

    passed = 0
    failed = 0

    print(f"\n  {'Test':<15} | {'Size':<15} | {'Max Err':>10} | {'Status':>8}")
    print("  " + "-" * 55)

    for name, M, N, K, pattern in test_cases:
        if not dispatcher.is_supported(M, N, K):
            print(f"  {name:<15} | {M}x{N}x{K:<5} | {'N/A':>10} | Skipped")
            continue

        np.random.seed(42)
        if pattern == "identity":
            A = np.eye(M, K, dtype=np_dtype)
            B = np.eye(K, N, dtype=np_dtype)
        else:
            A = (np.random.randn(M, K) * 0.1).astype(np_dtype)
            B = (np.random.randn(K, N) * 0.1).astype(np_dtype)

        result = dispatcher.run(A, B, M, N, K)
        if not result.success:
            print(f"  {name:<15} | {M}x{N}x{K:<5} | {'GPU Err':>10} | FAILED")
            failed += 1
            continue

        C_ref = np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(np_dtype)
        is_valid, max_err, _ = validator.check(result.output, C_ref)

        if is_valid:
            print(f"  {name:<15} | {M}x{N}x{K:<5} | {max_err:>10.2e} | PASSED")
            passed += 1
        else:
            print(f"  {name:<15} | {M}x{N}x{K:<5} | {max_err:>10.2e} | FAILED")
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed")
    print(f"Settings: dtype={args.dtype}, rtol={args.rtol}, atol={args.atol}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
