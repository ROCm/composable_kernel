#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 04: Validation

Validates GPU GEMM against NumPy reference using explicit API.

Complexity: ★★★☆☆

Usage:
    python3 04_validation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    CodegenRunner,
    DispatcherLib,
    Registry,
    Dispatcher,
    Validator,
)


def main():
    print("=" * 60)
    print("Example 04: Validation")
    print("=" * 60)

    # =========================================================================
    # Step 1: Define kernel config
    # =========================================================================
    print("\nStep 1: Define KernelConfig")

    config = KernelConfig(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        pad_m=True,
        pad_n=True,
        pad_k=True,
    )
    print(f"  Tile: {config.tile_str}")

    # =========================================================================
    # Step 2: Setup registry and dispatcher
    # =========================================================================
    print("\nStep 2: Setup")

    codegen = CodegenRunner()
    codegen.generate_from_config(config)

    lib = DispatcherLib.auto()
    if lib is None:
        print("  ERROR: Could not load library")
        return 1

    registry = Registry(name="validation", lib=lib)
    registry.register_kernel(config)

    dispatcher = Dispatcher(registry=registry, lib=lib)
    print(f"  {dispatcher}")

    # =========================================================================
    # Step 3: Run validation tests
    # =========================================================================
    print("\nStep 3: Validation Tests")

    validator = Validator(rtol=1e-3, atol=1e-2)

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

        # Create inputs
        np.random.seed(42)
        if pattern == "identity":
            A = np.eye(M, K, dtype=np.float16)
            B = np.eye(K, N, dtype=np.float16)
        else:
            A = (np.random.randn(M, K) * 0.1).astype(np.float16)
            B = (np.random.randn(K, N) * 0.1).astype(np.float16)

        # Run GPU
        result = dispatcher.run(A, B, M, N, K)
        if not result.success:
            print(f"  {name:<15} | {M}x{N}x{K:<5} | {'GPU Err':>10} | FAILED")
            failed += 1
            continue

        # Compute reference
        C_ref = np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(np.float16)

        # Validate
        is_valid, max_err, _ = validator.check(result.output, C_ref)

        if is_valid:
            print(f"  {name:<15} | {M}x{N}x{K:<5} | {max_err:>10.2e} | PASSED")
            passed += 1
        else:
            print(f"  {name:<15} | {M}x{N}x{K:<5} | {max_err:>10.2e} | FAILED")
            failed += 1

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
